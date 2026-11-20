import base64
import hashlib
import hmac
import io
import math
import os
import re
import secrets as pysecrets
from datetime import datetime, timedelta

import cv2
import librosa
import mediapipe as mp
import numpy as np
import pandas as pd
import requests
import streamlit as st
from PIL import Image, ImageDraw
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer

try:
    import parselmouth
    from parselmouth.praat import call
    HAS_PRAAT = True
except Exception:
    HAS_PRAAT = False

try:
    import reportlab  # noqa: F401
    HAS_REPORTLAB = True
except Exception:
    HAS_REPORTLAB = False

try:
    from sklearn.ensemble import IsolationForest
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False


# =========================================================
# CONFIG
# =========================================================
st.set_page_config(page_title="VOICEGUARD AI", page_icon="🎙️", layout="wide")

VOICE_FILE = "voice_history.csv"
NECK_FILE = "neck_history.csv"

N_BASE = 3                 # recordings used to build the personal baseline
Z_THRESHOLD = 2.5          # demo threshold (not clinical)
PCT_THRESHOLD = 10.0       # demo threshold (not clinical)
ASYMMETRY_THRESHOLD = 25.0 # demo threshold (not clinical)
NECK_DROP_THRESHOLD = 20.0 # demo threshold (not clinical)

TEST_TYPES = ["Sustained vowel (aaa)", "Standard sentence", "Pitch glide (low to high)"]

FEATURES_BY_TEST = {
    "Sustained vowel (aaa)": {
        "Phonation_sec": "Phonation duration",
        "Median_Pitch_Hz": "Median pitch",
        "Pitch_Variability_percent": "Pitch variability",
        "Jitter_percent": "Jitter",
        "Shimmer_percent": "Shimmer",
        "HNR_dB": "Harmonics-to-noise ratio",
    },
    "Standard sentence": {
        "Median_Pitch_Hz": "Median pitch",
        "Pitch_Variability_percent": "Pitch variability",
        "Mean_RMS": "Loudness (RMS)",
    },
    "Pitch glide (low to high)": {
        "Pitch_Range_Hz": "Pitch range",
        "Median_Pitch_Hz": "Median pitch",
    },
}

NECK_FEATURES = {
    "Flexion": "Neck flexion (°)",
    "Extension": "Neck extension (°)",
    "Right_Rotation": "Right rotation (°)",
    "Left_Rotation": "Left rotation (°)",
    "Tilt_Right": "Tilt right (°)",
    "Tilt_Left": "Tilt left (°)",
}

# 3D reference face model (image-style axes: x right, y down, z away from camera).
# Generic model: nose tip, chin, eye outer corners, mouth corners.
MODEL_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),        # nose tip      (landmark 1)
        (0.0, 330.0, 65.0),     # chin          (landmark 152)
        (-225.0, -170.0, 135.0),  # eye corner  (landmark 33)
        (225.0, -170.0, 135.0),   # eye corner  (landmark 263)
        (-150.0, 150.0, 125.0),   # mouth       (landmark 61)
        (150.0, 150.0, 125.0),    # mouth       (landmark 291)
    ],
    dtype=np.float64,
)
LANDMARK_IDS = [1, 152, 33, 263, 61, 291]
PROC_W, PROC_H = 400, 300  # resolution used for face analysis


# =========================================================
# HELPERS: ANGLES
# =========================================================
def wrap180(a):
    """Wrap an angle (or array) in degrees to [-180, 180)."""
    return (np.asarray(a) + 180.0) % 360.0 - 180.0


def head_angles(face, w, h):
    """Return (pitch, yaw, roll) in degrees from 6 face landmarks, or None."""
    pts = np.array(
        [[face.landmark[i].x * w, face.landmark[i].y * h] for i in LANDMARK_IDS],
        dtype=np.float64,
    )
    cam = np.array([[w, 0, w / 2.0], [0, w, h / 2.0], [0, 0, 1]], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(
        MODEL_POINTS, pts, cam, np.zeros((4, 1)), flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        return None
    rot, _ = cv2.Rodrigues(rvec)
    angles = cv2.RQDecomp3x3(rot)[0]
    return float(angles[0]), float(angles[1]), float(angles[2])


def roll_from_eyes(face, w, h):
    """Stable head-tilt (roll) in degrees from the line between the outer eye corners.
    Positive = tilt toward the person's right shoulder (verify with the swap option)."""
    x_r, y_r = face.landmark[33].x * w, face.landmark[33].y * h    # person's right eye
    x_l, y_l = face.landmark[263].x * w, face.landmark[263].y * h  # person's left eye
    return float(np.degrees(np.arctan2(y_r - y_l, x_l - x_r)))


def head_rotation(face, w, h):
    """3x3 head rotation matrix built from the 3D face mesh (columns = head x, y, z axes
    in camera coordinates: x right, y down, z away from the camera)."""
    def P(i):
        lm = face.landmark[i]
        return np.array([lm.x * w, lm.y * h, lm.z * w], dtype=np.float64)

    x_raw = (P(454) - P(234)) + (P(356) - P(127))  # across the face, two cheek pairs
    y_raw = P(152) - P(10)                          # forehead to chin
    nx, ny = np.linalg.norm(x_raw), np.linalg.norm(y_raw)
    if nx < 1e-6 or ny < 1e-6:
        return None
    x = x_raw / nx
    z = np.cross(x, y_raw)
    nz = np.linalg.norm(z)
    if nz < 1e-6:
        return None
    z /= nz
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def rot_to_pitch_yaw(rel):
    """Pitch (+ = looking down) and yaw (+ = turning to the person's right), in degrees."""
    yaw = np.degrees(np.arctan2(rel[0, 2], rel[2, 2]))
    pitch = np.degrees(np.arcsin(np.clip(-rel[1, 2], -1.0, 1.0)))
    return float(pitch), float(yaw)


# =========================================================
# NECK MOVEMENT VIDEO PROCESSOR
# =========================================================
class NeckMovementProcessor(VideoProcessorBase):
    CALIBRATION_FRAMES = 15

    def __init__(self):
        self.frame_count = 0
        self.movement_data = []

        # UI-controlled sign options (set from the main Streamlit thread)
        self.swap_left_right = False
        self.swap_flex_ext = False

        self.reset_calibration()
        self.reset_maxima()

        self.assessment_active = False
        self.assessment_completed = False

        self.current_pitch = 0.0
        self.current_yaw = 0.0
        self.current_roll = 0.0
        self.face_visible = False

        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    # ---- state management -------------------------------------------------
    def reset_calibration(self):
        self.calibration_samples = []
        self.neutral = None
        self.neutral_rot = None
        self.neutral_roll = 0.0
        self.is_calibrated = False
        self.last_rel = None
        self.skipped = 0

    def reset_maxima(self):
        self.max_flexion = 0.0
        self.max_extension = 0.0
        self.max_right_rotation = 0.0
        self.max_left_rotation = 0.0
        self.max_tilt_right = 0.0
        self.max_tilt_left = 0.0

    def start_assessment(self):
        self.reset_maxima()
        self.movement_data = []
        self.assessment_active = True
        self.assessment_completed = False

    def stop_assessment(self):
        self.assessment_active = False
        self.assessment_completed = True

    # ---- frame loop -------------------------------------------------------
    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        self._tick = getattr(self, "_tick", 0) + 1
        if self._tick % 2 == 0:  # analyse every 2nd frame to keep the server fast
            self._analyze(image)  # measurements always use the unflipped image
        return frame

    def _analyze(self, image):
        self.frame_count += 1

        small = cv2.resize(image, (PROC_W, PROC_H), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            self.face_visible = False
            return

        self.face_visible = True
        face = results.multi_face_landmarks[0]
        rot = head_rotation(face, PROC_W, PROC_H)
        if rot is None:
            return
        roll = roll_from_eyes(face, PROC_W, PROC_H)

        if not self.is_calibrated:
            self.calibration_samples.append((rot, roll))
            if len(self.calibration_samples) >= self.CALIBRATION_FRAMES:
                mean_rot = np.mean([s[0] for s in self.calibration_samples], axis=0)
                u, _, vt = np.linalg.svd(mean_rot)
                if np.linalg.det(u @ vt) < 0:
                    u[:, -1] *= -1
                self.neutral_rot = u @ vt  # nearest true rotation to the average
                self.neutral_roll = float(np.median([s[1] for s in self.calibration_samples]))
                self.is_calibrated = True
            return

        # Rotation relative to the person's own neutral pose (removes camera/face-shape offsets).
        rel_rot = rot @ self.neutral_rot.T
        rel_pitch, rel_yaw = rot_to_pitch_yaw(rel_rot)
        rel_roll = float(wrap180(roll - self.neutral_roll))

        if self.swap_flex_ext:
            rel_pitch = -rel_pitch
        if self.swap_left_right:
            rel_yaw = -rel_yaw
            rel_roll = -rel_roll

        # Reject impossible values and one-frame glitches (e.g. a flipped pose
        # solution that shows up as ~180 degrees), then smooth lightly.
        rel = np.array([rel_pitch, rel_yaw, rel_roll])
        if np.any(np.abs(rel) > 90):
            return
        if self.last_rel is not None:
            jump = np.abs(rel - self.last_rel)
            if np.any(jump > 30) and self.skipped < 10:
                self.skipped += 1
                return
            if np.any(jump > 30):
                self.last_rel = rel  # accept a real fast change after 10 frames
            rel = 0.6 * rel + 0.4 * self.last_rel
        self.skipped = 0
        self.last_rel = rel
        rel_pitch, rel_yaw, rel_roll = (float(v) for v in rel)

        self.current_pitch = rel_pitch
        self.current_yaw = rel_yaw
        self.current_roll = rel_roll

        # Convention (verify by testing, then use the sidebar swap options):
        #   pitch > 0 -> flexion (looking down), pitch < 0 -> extension
        #   yaw   > 0 -> right rotation,         yaw   < 0 -> left rotation
        #   roll  > 0 -> tilt right,             roll  < 0 -> tilt left
        if self.assessment_active:
            if rel_pitch > 0:
                self.max_flexion = max(self.max_flexion, rel_pitch)
            else:
                self.max_extension = max(self.max_extension, -rel_pitch)
            if rel_yaw > 0:
                self.max_right_rotation = max(self.max_right_rotation, rel_yaw)
            else:
                self.max_left_rotation = max(self.max_left_rotation, -rel_yaw)
            if rel_roll > 0:
                self.max_tilt_right = max(self.max_tilt_right, rel_roll)
            else:
                self.max_tilt_left = max(self.max_tilt_left, -rel_roll)

            if self.frame_count % 5 == 0:
                self.movement_data.append(
                    {"pitch": rel_pitch, "yaw": rel_yaw, "roll": rel_roll}
                )

        return

    def __del__(self):
        try:
            if getattr(self, "face_mesh", None) is not None:
                self.face_mesh.close()
        except Exception:
            pass


# =========================================================
# HELPERS: STORAGE
# =========================================================
PROFILE_FILE = "profile_local.csv"
KIND = {VOICE_FILE: "voice", NECK_FILE: "neck", PROFILE_FILE: "profile"}
PATIENTS_FILE = "patients_local.csv"
NON_NUMERIC = ("Patient_ID", "Date", "Test_Type", "Name")


def _supabase_cfg():
    """(url, key) if a [supabase] secret is configured, else None (local CSV fallback)."""
    try:
        cfg = dict(st.secrets["supabase"])
        if cfg.get("url") and cfg.get("key"):
            url = cfg["url"].strip().rstrip("/")
            if url.endswith("/rest/v1"):  # tolerate a URL pasted with the API path
                url = url[: -len("/rest/v1")]
            return url, cfg["key"].strip()
    except Exception:
        pass
    return None


def _headers(key, extra=None):
    h = {"apikey": key, "Content-Type": "application/json"}
    if not key.startswith("sb_"):
        # Legacy JWT keys go in both headers. The newer sb_secret_... / sb_publishable_...
        # keys are not JWTs and must be sent in the apikey header only.
        h["Authorization"] = f"Bearer {key}"
    if extra:
        h.update(extra)
    return h


def _clean(v):
    if isinstance(v, (float, np.floating)):
        return None if pd.isna(v) else float(v)
    if isinstance(v, np.integer):
        return int(v)
    return v


@st.cache_data(ttl=60, show_spinner=False)
def _fetch_remote(url, key, kind, patient_id):
    r = requests.get(
        f"{url}/rest/v1/records",
        headers=_headers(key),
        params={
            "patient_id": f"eq.{patient_id}",
            "kind": f"eq.{kind}",
            "select": "data",
            "order": "date.asc",
            "limit": "5000",
        },
        timeout=15,
    )
    r.raise_for_status()
    return [x["data"] for x in r.json()]


def save_row(path, row):
    cfg = _supabase_cfg()
    if cfg:
        url, key = cfg
        body = {
            "patient_id": str(row["Patient_ID"]),
            "kind": KIND[path],
            "date": str(row["Date"]),
            "data": {k: _clean(v) for k, v in row.items()},
        }
        r = requests.post(
            f"{url}/rest/v1/records",
            headers=_headers(key, {"Prefer": "return=minimal"}),
            json=body,
            timeout=15,
        )
        r.raise_for_status()
        _fetch_remote.clear()
        return
    df = pd.DataFrame([row])
    if os.path.exists(path):
        df = pd.concat([pd.read_csv(path), df], ignore_index=True)
    df.to_csv(path, index=False)


def load_history(path, patient_id):
    cfg = _supabase_cfg()
    if cfg:
        try:
            rows = _fetch_remote(cfg[0], cfg[1], KIND[path], str(patient_id))
        except Exception as error:
            st.session_state["storage_error"] = str(error)
            return pd.DataFrame()
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        for c in df.columns:
            if c not in NON_NUMERIC:
                df[c] = pd.to_numeric(df[c], errors="coerce")
    else:
        if not os.path.exists(path):
            return pd.DataFrame()
        try:
            df = pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
        if "Patient_ID" not in df.columns:
            return pd.DataFrame()
        df = df[df["Patient_ID"].astype(str) == str(patient_id)].copy()
    if df.empty:
        return df
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)


def delete_patient_rows(path, patient_id):
    cfg = _supabase_cfg()
    if cfg:
        url, key = cfg
        r = requests.delete(
            f"{url}/rest/v1/records",
            headers=_headers(key),
            params={"patient_id": f"eq.{patient_id}", "kind": f"eq.{KIND[path]}"},
            timeout=15,
        )
        r.raise_for_status()
        _fetch_remote.clear()
        return
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    if "Patient_ID" in df.columns:
        df = df[df["Patient_ID"].astype(str) != str(patient_id)]
        df.to_csv(path, index=False)


# ---------------------------------------------------------------- accounts (ID + PIN)
def hash_pin(pin, salt_hex):
    return hashlib.pbkdf2_hmac("sha256", pin.encode(), bytes.fromhex(salt_hex), 150_000).hex()


def get_patient(pid):
    cfg = _supabase_cfg()
    if cfg:
        url, key = cfg
        r = requests.get(
            f"{url}/rest/v1/patients",
            headers=_headers(key),
            params={"patient_id": f"eq.{pid}", "select": "salt,pin_hash"},
            timeout=15,
        )
        r.raise_for_status()
        rows = r.json()
        return rows[0] if rows else None
    if not os.path.exists(PATIENTS_FILE):
        return None
    df = pd.read_csv(PATIENTS_FILE, dtype=str)
    hit = df[df["patient_id"] == pid]
    return None if hit.empty else hit.iloc[0].to_dict()


def create_patient(pid, pin):
    salt = pysecrets.token_hex(16)
    record = {"patient_id": pid, "salt": salt, "pin_hash": hash_pin(pin, salt)}
    cfg = _supabase_cfg()
    if cfg:
        url, key = cfg
        r = requests.post(
            f"{url}/rest/v1/patients",
            headers=_headers(key, {"Prefer": "return=minimal"}),
            json=record,
            timeout=15,
        )
        r.raise_for_status()
        return
    df = pd.DataFrame([record])
    if os.path.exists(PATIENTS_FILE):
        df = pd.concat([pd.read_csv(PATIENTS_FILE, dtype=str), df], ignore_index=True)
    df.to_csv(PATIENTS_FILE, index=False)


def clean_name(name):
    name = re.sub(r"[*_`#<>\[\]\\]", "", name or "")
    return re.sub(r"\s+", " ", name).strip()[:40]


def get_display_name(pid):
    if pid == "DEMO":
        return "Demo Patient"
    df = load_history(PROFILE_FILE, pid)
    if df.empty or "Name" not in df.columns:
        return None
    names = df["Name"].dropna().astype(str)
    return names.iloc[-1] if not names.empty else None


def save_display_name(pid, name):
    save_row(
        PROFILE_FILE,
        {"Patient_ID": pid, "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Name": name},
    )


def sign_in_or_create(pid, pin):
    """Returns (ok, message). A new ID creates an account; an existing ID needs its PIN."""
    if not re.fullmatch(r"[A-Z0-9_-]{3,30}", pid):
        return False, "ID must be 3 to 30 characters: letters, digits, - or _."
    if pid == "DEMO":
        return False, "DEMO is reserved. Use the demo button instead."
    if not re.fullmatch(r"\d{4,8}", pin):
        return False, "PIN must be 4 to 8 digits."
    if st.session_state.get("pin_fails", 0) >= 5:
        return False, "Too many wrong PINs. Reload the page to try again."
    try:
        acct = get_patient(pid)
        if acct is None:
            create_patient(pid, pin)
            return True, "New account created. Remember your ID and PIN."
        if hmac.compare_digest(hash_pin(pin, acct["salt"]), acct["pin_hash"]):
            return True, "Welcome back."
    except Exception as error:
        return False, f"Storage error: {error}"
    st.session_state["pin_fails"] = st.session_state.get("pin_fails", 0) + 1
    return False, "Wrong PIN for this ID."


def load_demo_data():
    """Create a simulated patient: stable baseline, then worsening voice + reduced left rotation."""
    pid = "DEMO"
    delete_patient_rows(VOICE_FILE, pid)
    delete_patient_rows(NECK_FILE, pid)
    rng = np.random.default_rng(42)
    start = datetime.now() - timedelta(days=14)

    phon = [14.5, 14.8, 14.2, 13.9, 11.5, 9.8, 8.9]
    jit = [0.55, 0.60, 0.52, 0.58, 0.95, 1.20, 1.35]
    shim = [3.1, 3.3, 3.0, 3.2, 4.8, 5.6, 6.1]
    hnr = [21.0, 20.5, 21.3, 20.8, 17.0, 15.5, 14.8]
    for i in range(7):
        save_row(
            VOICE_FILE,
            {
                "Patient_ID": pid,
                "Date": (start + timedelta(days=2 * i)).strftime("%Y-%m-%d %H:%M:%S"),
                "Test_Type": TEST_TYPES[0],
                "Duration_sec": phon[i] + 1.5,
                "Phonation_sec": phon[i],
                "Median_Pitch_Hz": 190 + rng.normal(0, 2),
                "Pitch_Range_Hz": 12 + rng.normal(0, 1),
                "Pitch_Variability_percent": 2.0 + rng.normal(0, 0.1),
                "Mean_RMS": 0.05 + rng.normal(0, 0.002),
                "Jitter_percent": jit[i],
                "Shimmer_percent": shim[i],
                "HNR_dB": hnr[i],
            },
        )
    neck = [(40, 45, 60, 58), (41, 44, 59, 57), (40, 46, 58, 44), (39, 44, 57, 38)]
    for i, (fl, ex, rr, lr) in enumerate(neck):
        save_row(
            NECK_FILE,
            {
                "Patient_ID": pid,
                "Date": (start + timedelta(days=4 * i)).strftime("%Y-%m-%d %H:%M:%S"),
                "Flexion": fl,
                "Extension": ex,
                "Right_Rotation": rr,
                "Left_Rotation": lr,
                "Tilt_Right": 30,
                "Tilt_Left": 29,
            },
        )


# =========================================================
# HELPERS: VOICE
# =========================================================
def longest_run(mask, hop_sec):
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best * hop_sec


def extract_voice_features(y, sr):
    """Returns (features dict or None, pitch array, warning list)."""
    warnings = []

    peak = float(np.max(np.abs(y))) if y.size else 0.0
    rms_all = float(np.sqrt(np.mean(y**2))) if y.size else 0.0
    if rms_all < 0.005:
        return None, None, ["Recording is too quiet. Move closer to the microphone."]
    if peak >= 0.99 and np.mean(np.abs(y) >= 0.99) > 0.001:
        warnings.append("Possible clipping (too loud). Move slightly away from the microphone.")

    hop = 512
    f0, voiced, _ = librosa.pyin(y, fmin=65, fmax=400, sr=sr, hop_length=hop)
    voiced = np.nan_to_num(voiced, nan=0).astype(bool)
    valid = voiced & np.isfinite(f0)
    vf0 = f0[valid]

    voiced_fraction = float(np.mean(valid)) if valid.size else 0.0
    if len(vf0) < 10 or voiced_fraction < 0.1:
        return None, None, ["Not enough voiced sound detected. Record again in a quiet room."]

    feats = {
        "Duration_sec": float(len(y) / sr),
        "Phonation_sec": float(longest_run(valid, hop / sr)),
        "Median_Pitch_Hz": float(np.median(vf0)),
        "Pitch_Range_Hz": float(np.max(vf0) - np.min(vf0)),
        "Pitch_Variability_percent": float(np.std(vf0) / np.mean(vf0) * 100),
        "Mean_RMS": float(np.mean(librosa.feature.rms(y=y)[0])),
        "Jitter_percent": np.nan,
        "Shimmer_percent": np.nan,
        "HNR_dB": np.nan,
    }

    if HAS_PRAAT:
        try:
            snd = parselmouth.Sound(y.astype(np.float64), sampling_frequency=sr)
            pp = call(snd, "To PointProcess (periodic, cc)", 65, 400)
            feats["Jitter_percent"] = float(
                call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3) * 100
            )
            feats["Shimmer_percent"] = float(
                call([snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6) * 100
            )
            harm = call(snd, "To Harmonicity (cc)", 0.01, 65, 0.1, 1.0)
            feats["HNR_dB"] = float(call(harm, "Get mean", 0, 0))
        except Exception:
            warnings.append("Jitter/shimmer could not be computed for this recording.")
    else:
        warnings.append("praat-parselmouth is not installed: jitter/shimmer/HNR unavailable.")

    return feats, vf0, warnings


# =========================================================
# HELPERS: ANALYSIS
# =========================================================
def change_vs_baseline(df, col, n_base=N_BASE, min_rel_sd=0.05):
    """Latest value vs mean of the first n_base recordings -> (z, percent change)."""
    if col not in df.columns or len(df) <= n_base:
        return None
    series = df[col]
    base = series.iloc[:n_base].dropna()
    cur = series.iloc[-1]
    if len(base) < n_base or pd.isna(cur):
        return None
    mu = float(base.mean())
    sd = max(float(base.std(ddof=1)), abs(mu) * min_rel_sd, 1e-9)  # floor: few samples
    pct = (cur - mu) / mu * 100 if mu else 0.0
    return (cur - mu) / sd, pct


def compute_findings(patient_id, use_ml=False):
    findings, notes, z_all = [], [], []

    voice = load_history(VOICE_FILE, patient_id)
    if voice.empty:
        notes.append("No voice data saved for this patient yet.")
    else:
        if "Test_Type" not in voice.columns:
            voice["Test_Type"] = TEST_TYPES[0]
        for test, grp in voice.groupby("Test_Type"):
            grp = grp.sort_values("Date").reset_index(drop=True)
            if len(grp) <= N_BASE:
                notes.append(
                    f"Voice baseline still being built for '{test}' "
                    f"({len(grp)}/{N_BASE + 1} recordings needed)."
                )
                continue
            for col, label in FEATURES_BY_TEST.get(test, {}).items():
                r = change_vs_baseline(grp, col)
                if r is None:
                    continue
                z, pct = r
                if test == TEST_TYPES[0]:
                    z_all.append(float(np.clip(z, -10, 10)))
                if abs(z) > Z_THRESHOLD and abs(pct) > PCT_THRESHOLD:
                    findings.append(
                        f"Voice ({test}): {label} changed {pct:+.0f}% compared with the personal baseline."
                    )

    neck = load_history(NECK_FILE, patient_id)
    if neck.empty:
        notes.append("No neck movement data saved for this patient yet.")
    else:
        last = neck.iloc[-1]
        hi = max(last["Right_Rotation"], last["Left_Rotation"])
        if hi > 0:
            asym = abs(last["Right_Rotation"] - last["Left_Rotation"]) / hi * 100
            if asym > ASYMMETRY_THRESHOLD:
                weaker = "left" if last["Left_Rotation"] < last["Right_Rotation"] else "right"
                findings.append(
                    f"Neck: rotation differs by {asym:.0f}% between sides (lower on the {weaker})."
                )
        if len(neck) >= 2:
            for col in ["Flexion", "Extension", "Right_Rotation", "Left_Rotation"]:
                prev = neck[col].iloc[-2]
                cur = neck[col].iloc[-1]
                if prev > 0 and (cur - prev) / prev * 100 < -NECK_DROP_THRESHOLD:
                    findings.append(
                        f"Neck: {col.replace('_', ' ').lower()} dropped "
                        f"{abs((cur - prev) / prev * 100):.0f}% compared with the previous assessment."
                    )

    ml_note = None
    if use_ml and HAS_SKLEARN and len(z_all) >= 3:
        rng = np.random.default_rng(0)
        sim_normal = rng.normal(0, 1, size=(500, len(z_all)))  # SIMULATED normal variation
        model = IsolationForest(random_state=0, contamination=0.05).fit(sim_normal)
        is_anomaly = model.predict(np.array([z_all]))[0] == -1
        ml_note = (
            "ML check (Isolation Forest trained on simulated normal variation): "
            + ("unusual overall pattern versus baseline." if is_anomaly
               else "no unusual overall pattern.")
        )
        if is_anomaly:
            findings.append("ML: the combined voice features look unusual compared with the baseline.")

    return findings, notes, ml_note


def recovery_profile(df, features, n_base=N_BASE):
    """Baseline / Previous / Current table (the 'Patient Recovery Profile')."""
    rows = []
    for col, label in features.items():
        if col not in df.columns:
            continue
        s = df[col].dropna()
        if s.empty:
            continue
        cur = float(s.iloc[-1])
        prev = float(s.iloc[-2]) if len(s) >= 2 else np.nan
        base = float(s.iloc[:n_base].mean()) if len(s) >= n_base else np.nan
        change = (cur - base) / base * 100 if pd.notna(base) and base else np.nan
        rows.append(
            {
                "Measure": label,
                "Baseline": base,
                "Previous": prev,
                "Current": cur,
                "Change vs baseline (%)": change,
            }
        )
    return pd.DataFrame(rows)


def get_profiles(patient_id):
    """Returns ({test_type: profile_df}, neck_profile_df)."""
    voice = load_history(VOICE_FILE, patient_id)
    neck = load_history(NECK_FILE, patient_id)
    voice_profiles = {}
    if not voice.empty:
        if "Test_Type" not in voice.columns:
            voice["Test_Type"] = TEST_TYPES[0]
        for test, grp in voice.groupby("Test_Type"):
            grp = grp.sort_values("Date").reset_index(drop=True)
            prof = recovery_profile(grp, FEATURES_BY_TEST.get(test, {}), n_base=N_BASE)
            if not prof.empty:
                voice_profiles[test] = prof
    neck_profile = (
        recovery_profile(neck, NECK_FEATURES, n_base=1) if not neck.empty else pd.DataFrame()
    )
    return voice_profiles, neck_profile


def show_table(df):
    """Display a profile table with clean number formatting."""
    shown = df.copy()
    for c in shown.columns[1:]:
        shown[c] = shown[c].map(lambda v: "—" if pd.isna(v) else f"{v:.2f}")
    st.dataframe(shown, use_container_width=True, hide_index=True)


VOICE_PDF_COLUMNS = [
    ("Date", "Date", "date"),
    ("Test_Type", "Test", "text"),
    ("Phonation_sec", "Phonation (s)", "num2"),
    ("Median_Pitch_Hz", "Median pitch (Hz)", "num1"),
    ("Pitch_Range_Hz", "Pitch range (Hz)", "num1"),
    ("Pitch_Variability_percent", "Pitch variability (%)", "num2"),
    ("Jitter_percent", "Jitter (%)", "num2"),
    ("Shimmer_percent", "Shimmer (%)", "num2"),
    ("HNR_dB", "HNR (dB)", "num1"),
]
NECK_PDF_COLUMNS = [
    ("Date", "Date", "date"),
    ("Flexion", "Flexion (°)", "num1"),
    ("Extension", "Extension (°)", "num1"),
    ("Right_Rotation", "Right rotation (°)", "num1"),
    ("Left_Rotation", "Left rotation (°)", "num1"),
    ("Tilt_Right", "Tilt right (°)", "num1"),
    ("Tilt_Left", "Tilt left (°)", "num1"),
]


def history_pdf(title, patient_name, patient_id, df, columns):
    """Build a clean, printable PDF table of a patient's history. Returns bytes."""
    from xml.sax.saxutils import escape

    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    navy, teal, ink = colors.HexColor("#153A5B"), colors.HexColor("#167D9A"), colors.HexColor("#263445")
    page = landscape(A4)
    margin = 14 * mm
    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=styles["Title"], textColor=navy, fontSize=20, alignment=0, spaceAfter=4)
    info = ParagraphStyle("info", parent=styles["Normal"], fontSize=10, textColor=ink, leading=14)
    cell = ParagraphStyle("cell", parent=styles["Normal"], fontSize=8.5, leading=10.5, textColor=ink)
    head = ParagraphStyle("head", parent=cell, textColor=colors.white, fontName="Helvetica-Bold")

    def latin_only(text):
        return text if text and all(ord(ch) < 256 for ch in text) else None

    def fmt(value, kind):
        if value is None or pd.isna(value):
            return "-"
        if kind == "date":
            return pd.to_datetime(value).strftime("%Y-%m-%d %H:%M")
        if kind == "num1":
            return f"{float(value):.1f}"
        if kind == "num2":
            return f"{float(value):.2f}"
        return escape(str(value))

    cols = [c for c in columns if c[0] in df.columns]
    weights = [1.5 if k == "date" else 2.0 if k == "text" else 1.0 for _, _, k in cols]
    usable = page[0] - 2 * margin
    widths = [usable * w / sum(weights) for w in weights]

    data = [[Paragraph(escape(h), head) for _, h, _ in cols]]
    for _, row in df.iterrows():
        data.append([Paragraph(fmt(row[c], k), cell) for c, _, k in cols])

    table = Table(data, colWidths=widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), navy),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F4F7FB")]),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#DCE5EF")),
                ("LINEBELOW", (0, 0), (-1, 0), 1.2, teal),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )

    shown_name = latin_only(patient_name)
    who = f"{escape(shown_name)} (ID: {escape(str(patient_id))})" if shown_name else f"ID: {escape(str(patient_id))}"
    story = [
        Paragraph(escape(title), h1),
        Paragraph(f"<b>Patient:</b> {who}", info),
        Paragraph(
            f"<b>Records:</b> {len(df)} &nbsp;&nbsp; <b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            info,
        ),
        Spacer(1, 8 * mm),
        table,
    ]

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#5B6B7C"))
        canvas.drawString(margin, 8 * mm, "VOICEGUARD AI - exploratory monitoring data. Not a medical diagnosis.")
        canvas.drawRightString(page[0] - margin, 8 * mm, f"Page {doc.page}")
        canvas.restoreState()

    buf = io.BytesIO()
    SimpleDocTemplate(
        buf, pagesize=page, leftMargin=margin, rightMargin=margin, topMargin=margin,
        bottomMargin=16 * mm, title=title,
    ).build(story, onFirstPage=footer, onLaterPages=footer)
    return buf.getvalue()


def report_pdf(patient_name, patient_id, findings, notes, voice_profiles, neck_profile, ml_note):
    """Printable PDF version of the recovery report. Returns bytes."""
    from xml.sax.saxutils import escape

    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    navy, teal, ink = colors.HexColor("#153A5B"), colors.HexColor("#167D9A"), colors.HexColor("#263445")
    margin = 16 * mm
    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=styles["Title"], textColor=navy, fontSize=20, alignment=0, spaceAfter=4)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], textColor=navy, fontSize=13, spaceBefore=12, spaceAfter=6)
    body = ParagraphStyle("body", parent=styles["Normal"], fontSize=10, leading=14, textColor=ink)
    small = ParagraphStyle("small", parent=body, fontSize=8.5, leading=11, textColor=colors.HexColor("#5B6B7C"))
    cell = ParagraphStyle("cell", parent=body, fontSize=9, leading=11)
    head = ParagraphStyle("head", parent=cell, textColor=colors.white, fontName="Helvetica-Bold")

    def latin_only(text):
        return text if text and all(ord(ch) < 256 for ch in text) else None

    def num(v, signed=False):
        if v is None or pd.isna(v):
            return "-"
        return f"{v:+.1f}" if signed else f"{v:.2f}"

    def profile_table(prof):
        data = [[Paragraph(escape(str(c)), head) for c in prof.columns]]
        for _, row in prof.iterrows():
            cells = [Paragraph(escape(str(row.iloc[0])), cell)]
            for col in prof.columns[1:]:
                cells.append(Paragraph(num(row[col], signed=col.startswith("Change")), cell))
            data.append(cells)
        usable = A4[0] - 2 * margin
        widths = [usable * w for w in (0.32, 0.15, 0.15, 0.15, 0.23)]
        t = Table(data, colWidths=widths, repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), navy),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F4F7FB")]),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#DCE5EF")),
            ("LINEBELOW", (0, 0), (-1, 0), 1.2, teal),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        return t

    shown = latin_only(patient_name)
    who = f"{escape(shown)} (ID: {escape(str(patient_id))})" if shown else f"ID: {escape(str(patient_id))}"
    story = [
        Paragraph("Patient Recovery Report", h1),
        Paragraph(f"<b>Patient:</b> {who}", body),
        Paragraph(f"<b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}", body),
        Spacer(1, 6 * mm),
    ]

    if findings:
        bg, fg, label = colors.HexColor("#FFF4D6"), colors.HexColor("#7A5200"), "Review suggested"
    else:
        bg, fg, label = colors.HexColor("#E3F5EA"), colors.HexColor("#1E6B3A"), "No notable change from baseline"
    banner = Table(
        [[Paragraph(f"<b>{label}</b>", ParagraphStyle("b", parent=body, textColor=fg, fontSize=12))]],
        colWidths=[A4[0] - 2 * margin],
    )
    banner.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), bg), ("TOPPADDING", (0, 0), (-1, -1), 8),
                                ("BOTTOMPADDING", (0, 0), (-1, -1), 8), ("LEFTPADDING", (0, 0), (-1, -1), 10)]))
    story.append(banner)
    story.append(Spacer(1, 4 * mm))

    for f in findings:
        story.append(Paragraph(f"&bull; {escape(f)}", body))
    if findings:
        story.append(Spacer(1, 3 * mm))
        story.append(Paragraph(
            "<b>Suggested next step:</b> review the findings with the treating clinician and repeat the "
            "standardized assessment under the same recording conditions.", body))
    for n in notes:
        story.append(Paragraph(f"<i>Note: {escape(n)}</i>", small))
    if ml_note:
        story.append(Paragraph(escape(ml_note), small))

    for test, prof in voice_profiles.items():
        story.append(KeepTogether([Paragraph(f"Voice profile: {escape(test)}", h2), profile_table(prof)]))
    if not neck_profile.empty:
        story.append(KeepTogether([
            Paragraph("Neck movement profile (baseline = first assessment)", h2), profile_table(neck_profile)]))

    story += [
        Spacer(1, 8 * mm),
        Paragraph(
            "Exploratory monitoring report, not a medical diagnosis or treatment prescription. Thresholds are "
            "demo values and have not been validated on post-thyroidectomy patients. Camera-based neck angles are "
            "estimates, not clinical range-of-motion measurements. Persistent hoarseness or voice change needs "
            "medical evaluation. Seek urgent care for difficulty breathing, stridor, rapidly increasing neck "
            "swelling, or new or severe difficulty swallowing.", small),
    ]

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#5B6B7C"))
        canvas.drawString(margin, 8 * mm, "VOICEGUARD AI")
        canvas.drawRightString(A4[0] - margin, 8 * mm, f"Page {doc.page}")
        canvas.restoreState()

    buf = io.BytesIO()
    SimpleDocTemplate(buf, pagesize=A4, leftMargin=margin, rightMargin=margin, topMargin=margin,
                      bottomMargin=16 * mm, title="Patient Recovery Report").build(
        story, onFirstPage=footer, onLaterPages=footer)
    return buf.getvalue()


@st.cache_data(ttl=3600, show_spinner=False)
def _metered_ice_servers(app_name, api_key):
    r = requests.get(
        f"https://{app_name}.metered.live/api/v1/turn/credentials",
        params={"apiKey": api_key},
        timeout=10,
    )
    r.raise_for_status()
    return r.json()


def get_rtc_configuration():
    """STUN servers by default, plus a TURN relay when one is configured in the app secrets
    ([metered] or [turn]). A TURN relay fixes 'Connection is taking longer than expected'."""
    servers = [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
    ]
    try:  # Option 1: Metered.ca free TURN (credentials fetched automatically)
        m = dict(st.secrets["metered"])
        if m.get("app") and m.get("api_key"):
            servers.extend(_metered_ice_servers(m["app"].strip(), m["api_key"].strip()))
    except Exception:
        pass
    try:  # Option 2: any TURN server entered by hand
        turn = dict(st.secrets["turn"])
        urls = turn.get("urls")
        if isinstance(urls, str):
            urls = [urls]
        if urls and turn.get("username") and turn.get("credential"):
            servers.append(
                {"urls": list(urls), "username": turn["username"], "credential": turn["credential"]}
            )
    except Exception:
        pass
    return {"iceServers": servers}


# =========================================================
# EXERCISE ANIMATIONS (drawn in code with Pillow: no image files needed)
# =========================================================
EX_SIZE = 360
EX_K = 2  # draw at 2x and shrink for smooth edges
EX_INK = (21, 58, 91)
EX_FILL = (214, 230, 242)
EX_BACK = (190, 208, 225)
EX_ACCENT = (22, 125, 154)
EX_BG = (234, 242, 248)
EX_GUIDE = (150, 175, 200)


def _ex_p(pt):
    return (pt[0] * EX_K, pt[1] * EX_K)


def _ex_rot(pt, pivot, deg):
    a = math.radians(deg)
    dx, dy = pt[0] - pivot[0], pt[1] - pivot[1]
    return (pivot[0] + dx * math.cos(a) - dy * math.sin(a), pivot[1] + dx * math.sin(a) + dy * math.cos(a))


def _ex_circle(d, c, r, fill, outline=EX_INK, w=3):
    d.ellipse(
        [(c[0] - r) * EX_K, (c[1] - r) * EX_K, (c[0] + r) * EX_K, (c[1] + r) * EX_K],
        fill=fill, outline=outline, width=int(w * EX_K),
    )


def _ex_limb(d, pts, w, fill=EX_FILL):
    """A rounded, outlined thick line (neck, arms, torso)."""
    for width, color in ((w + 6, EX_INK), (w, fill)):
        d.line([_ex_p(p) for p in pts], fill=color, width=int(width * EX_K), joint="curve")
        for p in (pts[0], pts[-1]):
            r = width / 2
            d.ellipse([(p[0] - r) * EX_K, (p[1] - r) * EX_K, (p[0] + r) * EX_K, (p[1] + r) * EX_K], fill=color)


def _ex_dashed(d, p0, p1, color=EX_GUIDE, w=2, dash=9, gap=7):
    length = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
    ux, uy = (p1[0] - p0[0]) / length, (p1[1] - p0[1]) / length
    pos = 0.0
    while pos < length:
        end = min(pos + dash, length)
        d.line([_ex_p((p0[0] + ux * pos, p0[1] + uy * pos)), _ex_p((p0[0] + ux * end, p0[1] + uy * end))],
               fill=color, width=int(w * EX_K))
        pos += dash + gap


def _ex_front(d, turn=0.0, tilt=0.0, sh_off=((0, 0), (0, 0)), breath=0.0, chest=0.0,
              back=False, blade=0.0, show_level=False, roll_guides=False, breathing=False):
    cx = 180
    (ldx, ldy), (rdx, rdy) = sh_off
    lx, ly = 118 + ldx, 222 + ldy
    rx, ry = 242 + rdx, 222 + rdy
    widen = chest * 12
    torso = [(lx - widen, ly), (rx + widen, ry), (rx - 6 + widen, 372), (lx + 6 - widen, 372)]
    d.polygon([_ex_p(p) for p in torso], fill=EX_FILL, outline=EX_INK)
    d.line([_ex_p(p) for p in torso + [torso[0]]], fill=EX_INK, width=int(3 * EX_K))
    _ex_limb(d, [(lx - widen, ly + 6), (lx - 18 - widen, 340)], 16)
    _ex_limb(d, [(rx + widen, ry + 6), (rx + 18 + widen, 340)], 16)

    if show_level:
        _ex_dashed(d, (96, 222), (264, 222), EX_ACCENT, 2)
    if roll_guides:
        for (sx, sy) in ((118, 222), (242, 222)):
            d.ellipse([(sx - 13) * EX_K, (sy - 13) * EX_K, (sx + 13) * EX_K, (sy + 13) * EX_K],
                      outline=EX_ACCENT, width=int(2 * EX_K))
        for (jx, jy) in ((lx, ly), (rx, ry)):
            d.ellipse([(jx - 6) * EX_K, (jy - 6) * EX_K, (jx + 6) * EX_K, (jy + 6) * EX_K],
                      fill=EX_ACCENT, outline=EX_INK, width=int(2 * EX_K))

    base = ((ly + ry) / 2 - 6, (lx + rx) / 2)
    base_pt = (base[1], base[0])
    head0 = (cx, 128)  # the head stays put; only the shoulders move
    head = _ex_rot(head0, base_pt, tilt)
    _ex_limb(d, [base_pt, head], 22)
    _ex_circle(d, head, 42, EX_BACK if back else EX_FILL)

    if back:
        # shoulder blades + spine
        _ex_dashed(d, (cx, ly + 10), (cx, 360), EX_GUIDE, 2)
        gap = 36 - blade * 22
        for sign in (-1, 1):
            bx, by = cx + sign * gap, ly + 62
            d.ellipse([(bx - 17) * EX_K, (by - 30) * EX_K, (bx + 17) * EX_K, (by + 30) * EX_K],
                      fill=EX_ACCENT if blade > 0.5 else (170, 205, 220), outline=EX_INK, width=int(3 * EX_K))
    else:
        shift = turn * 13
        face = lambda p: _ex_rot((head0[0] + p[0], head0[1] + p[1]), base_pt, tilt)
        hc = head
        for ex in (-15, 15):
            e = _ex_rot((head0[0] + ex * (1 - 0.25 * abs(turn)) + shift, head0[1] - 6), base_pt, tilt)
            d.ellipse([(e[0] - 3.5) * EX_K, (e[1] - 3.5) * EX_K, (e[0] + 3.5) * EX_K, (e[1] + 3.5) * EX_K], fill=EX_INK)
        n0 = _ex_rot((head0[0] + shift * 1.25, head0[1] - 2), base_pt, tilt)
        n1 = _ex_rot((head0[0] + shift * 1.25 + 4, head0[1] + 12), base_pt, tilt)
        d.line([_ex_p(n0), _ex_p(n1)], fill=EX_INK, width=int(3 * EX_K))
        m0 = _ex_rot((head0[0] - 10 + shift, head0[1] + 25), base_pt, tilt)
        m1 = _ex_rot((head0[0] + 10 + shift, head0[1] + 25), base_pt, tilt)
        d.line([_ex_p(m0), _ex_p(m1)], fill=EX_INK, width=int(3 * EX_K))

    if breathing:
        rx_b, ry_b = 30 + breath * 24, 22 + breath * 12
        d.ellipse([(cx - rx_b) * EX_K, (300 - ry_b) * EX_K, (cx + rx_b) * EX_K, (300 + ry_b) * EX_K],
                  fill=(190, 225, 235), outline=EX_ACCENT, width=int(3 * EX_K))
        for sign in (-1, 1):
            hx = cx + sign * (rx_b + 14)
            _ex_circle(d, (hx, 300), 9, EX_FILL, EX_INK, 2)


def _ex_side(d, head_dx=0.0, pitch=0.0, sh_dx=0.0, plumb=False):
    hip = (178, 352)
    sh = (176 + sh_dx, 232)
    nb = (176 + sh_dx + head_dx * 0.25, 206)
    _ex_limb(d, [nb, (176 + sh_dx * 0.6, 262), (176 + sh_dx * 0.3, 300), hip], 46)
    _ex_circle(d, (sh[0] + 2, sh[1] + 14), 16, EX_FILL, EX_INK, 3)
    head0 = (176 + head_dx, 138)
    head = _ex_rot(head0, nb, pitch)
    _ex_limb(d, [nb, head], 20)
    _ex_circle(d, head, 40, EX_FILL)
    tip = _ex_rot((head0[0] + 52, head0[1] + 6), nb, pitch)
    top = _ex_rot((head0[0] + 38, head0[1] - 4), nb, pitch)
    bot = _ex_rot((head0[0] + 38, head0[1] + 14), nb, pitch)
    d.polygon([_ex_p(top), _ex_p(tip), _ex_p(bot)], fill=EX_FILL, outline=EX_INK)
    d.line([_ex_p(top), _ex_p(tip), _ex_p(bot)], fill=EX_INK, width=int(3 * EX_K))
    eye = _ex_rot((head0[0] + 14, head0[1] - 8), nb, pitch)
    d.ellipse([(eye[0] - 3.5) * EX_K, (eye[1] - 3.5) * EX_K, (eye[0] + 3.5) * EX_K, (eye[1] + 3.5) * EX_K], fill=EX_INK)
    ear = _ex_rot((head0[0] - 6, head0[1] + 2), nb, pitch)
    d.ellipse([(ear[0] - 6) * EX_K, (ear[1] - 8) * EX_K, (ear[0] + 6) * EX_K, (ear[1] + 8) * EX_K],
              fill=EX_BACK, outline=EX_INK, width=int(2 * EX_K))
    if plumb:
        _ex_dashed(d, (176, 70), (176, 372), EX_ACCENT, 2)


def _ex_bar(d, progress):
    """Small bar at the top that fills while breathing in and empties while breathing out."""
    x0, x1, y = 70, 290, 26
    d.rounded_rectangle([x0 * EX_K, (y - 8) * EX_K, x1 * EX_K, (y + 8) * EX_K], radius=8 * EX_K,
                        fill=(255, 255, 255), outline=EX_INK, width=int(2 * EX_K))
    if progress > 0.02:
        d.rounded_rectangle([x0 * EX_K, (y - 8) * EX_K, (x0 + (x1 - x0) * progress) * EX_K, (y + 8) * EX_K],
                            radius=8 * EX_K, fill=EX_ACCENT)


EXERCISE_ANIMATIONS = {
    "neck_flexion": dict(frames=36, ms=90),
    "neck_rotation": dict(frames=40, ms=90),
    "neck_lateral": dict(frames=40, ms=90),
    "shoulder_roll": dict(frames=36, ms=90),
    "shoulder_shrug": dict(frames=36, ms=90),
    "scapular_squeeze": dict(frames=36, ms=90),
    "chin_tuck": dict(frames=36, ms=90),
    "tall_posture": dict(frames=36, ms=90),
    "belly_breath": dict(frames=48, ms=110),
    "rib_breath": dict(frames=48, ms=110),
}


def _ex_draw(key, t):
    s = (1 - math.cos(2 * math.pi * t)) / 2      # 0 -> 1 -> 0
    w = math.sin(2 * math.pi * t)                 # -1 .. 1
    img = Image.new("RGB", (EX_SIZE * EX_K, EX_SIZE * EX_K), EX_BG)
    d = ImageDraw.Draw(img)
    if key == "neck_flexion":
        _ex_side(d, pitch=38 * s)
    elif key == "neck_rotation":
        _ex_front(d, turn=w)
    elif key == "neck_lateral":
        _ex_front(d, tilt=24 * w, show_level=True)
    elif key == "shoulder_roll":
        th = 2 * math.pi * t - math.pi / 2
        off = (11 * math.cos(th), 11 * math.sin(th))
        _ex_front(d, sh_off=(off, (-off[0], off[1])), roll_guides=True)
    elif key == "shoulder_shrug":
        _ex_front(d, sh_off=((0, -20 * s), (0, -20 * s)), show_level=True)
    elif key == "scapular_squeeze":
        _ex_front(d, back=True, blade=s)
    elif key == "chin_tuck":
        _ex_side(d, head_dx=30 * (1 - s), plumb=True)
    elif key == "tall_posture":
        _ex_side(d, head_dx=32 * (1 - s), sh_dx=16 * (1 - s), plumb=True)
    elif key == "belly_breath":
        _ex_front(d, breath=s, breathing=True)
        _ex_bar(d, s)
    elif key == "rib_breath":
        _ex_front(d, breath=0.3 * s, chest=s, breathing=True)
        _ex_bar(d, s)
    return img.resize((300, 300), Image.LANCZOS)


@st.cache_data(show_spinner=False)
def exercise_gif(key):
    """Animated GIF (bytes) for an exercise key."""
    cfg = EXERCISE_ANIMATIONS[key]
    frames = [_ex_draw(key, i / cfg["frames"]) for i in range(cfg["frames"])]
    palette = frames[0].quantize(colors=64, method=Image.MEDIANCUT)
    none = getattr(Image, "Dither", Image).NONE
    quant = [f.quantize(palette=palette, dither=none) for f in frames]
    buf = io.BytesIO()
    quant[0].save(buf, format="GIF", save_all=True, append_images=quant[1:],
                  duration=cfg["ms"], loop=0)
    return buf.getvalue()


@st.cache_data(show_spinner=False)
def exercise_gif_b64(key):
    return base64.b64encode(exercise_gif(key)).decode()


EXERCISE_LIBRARY = {
    "Gentle Neck ROM": {
        "goal": "Gentle, pain-free neck movements to improve mobility. Only within the range your surgeon or physiotherapist approved.",
        "exercises": [
            {
                "key": "neck_flexion",
                "name": "Chin to chest (gentle forward bend)",
                "steps": [
                    "Sit tall with your shoulders relaxed and your eyes looking forward.",
                    "Slowly lower your chin toward your chest until you feel a gentle stretch.",
                    "Hold for 2 to 3 seconds, breathing normally.",
                    "Slowly lift back to the starting position.",
                ],
                "dose": "About 5 slow repetitions, or as your physiotherapist advises.",
                "caution": "Never pull with your hands. Stop if you feel pain, pulling at the scar, dizziness or tingling. Do not tilt your head far backward unless your surgeon says it is fine.",
            },
            {
                "key": "neck_rotation",
                "name": "Turn your head side to side",
                "steps": [
                    "Sit tall and look straight ahead with your shoulders level.",
                    "Slowly turn your head to one side as far as is comfortable, without moving your shoulders.",
                    "Hold for 2 seconds, then return to the centre.",
                    "Repeat to the other side.",
                ],
                "dose": "About 5 times to each side, moving slowly.",
                "caution": "Move only as far as is comfortable. Stop if you feel pain, dizziness or tingling.",
            },
            {
                "key": "neck_lateral",
                "name": "Ear toward shoulder",
                "steps": [
                    "Sit tall with both shoulders relaxed and level (use the dotted line as a guide).",
                    "Slowly tilt your head so your ear moves toward your shoulder.",
                    "Keep the shoulder down. Do not lift it to meet your ear.",
                    "Hold for 2 seconds, return to the centre, then repeat to the other side.",
                ],
                "dose": "About 5 times to each side.",
                "caution": "Stop if you feel pain or pulling at the scar.",
            },
        ],
    },
    "Shoulder Mobility": {
        "goal": "Reduce shoulder and upper-back stiffness that can follow surgery.",
        "exercises": [
            {
                "key": "shoulder_roll",
                "name": "Shoulder rolls",
                "steps": [
                    "Stand or sit tall with your arms relaxed by your sides.",
                    "Lift your shoulders up, then roll them back and down in a smooth circle.",
                    "Keep your neck relaxed and keep breathing normally.",
                ],
                "dose": "5 to 10 slow circles. Then try the other direction.",
                "caution": "Keep the circles small and smooth. Stop if you feel pain.",
            },
            {
                "key": "shoulder_shrug",
                "name": "Shoulder shrugs",
                "steps": [
                    "Sit or stand with your arms relaxed.",
                    "Lift both shoulders up toward your ears (the dotted line shows where they started).",
                    "Hold for 2 seconds, then slowly lower them and let go of the tension.",
                ],
                "dose": "5 to 10 slow repetitions.",
                "caution": "Do not force the movement. Stop if you feel pain or pulling in your neck.",
            },
            {
                "key": "scapular_squeeze",
                "name": "Shoulder blade squeeze",
                "steps": [
                    "Sit or stand tall with your arms relaxed (this view is from behind).",
                    "Gently squeeze your shoulder blades back and down, as if holding a pencil between them.",
                    "Hold for 3 seconds, then relax.",
                ],
                "dose": "5 to 10 repetitions.",
                "caution": "Keep your shoulders down, away from your ears. Do not arch your lower back.",
            },
        ],
    },
    "Postural Correction": {
        "goal": "Keep the head stacked over the shoulders and avoid the forward-head posture that adds strain.",
        "exercises": [
            {
                "key": "chin_tuck",
                "name": "Chin tuck",
                "steps": [
                    "Sit or stand tall, looking straight ahead (the dotted line is your guide).",
                    "Slide your head straight backward, as if making a gentle double chin. Keep your eyes level.",
                    "Hold for 3 seconds, then relax forward.",
                ],
                "dose": "5 to 8 repetitions.",
                "caution": "Do not tilt your head down or up. This should be a small, comfortable movement.",
            },
            {
                "key": "tall_posture",
                "name": "Stand tall (posture reset)",
                "steps": [
                    "Start slouched, then lift your chest slightly and relax your shoulders back.",
                    "Line up your ear, shoulder and hip along the dotted line.",
                    "Hold while breathing normally, then check yourself in a mirror.",
                ],
                "dose": "Hold 10 to 20 seconds, a few times during the day.",
                "caution": "Stay relaxed. Do not stiffen your neck or push your chest out too far.",
            },
        ],
    },
    "Breathing Exercises": {
        "goal": "Slow, relaxed breathing to improve breath control, which also supports the voice. The bar at the top shows when to breathe in (fills) and out (empties).",
        "exercises": [
            {
                "key": "belly_breath",
                "name": "Belly (diaphragmatic) breathing",
                "steps": [
                    "Sit or lie comfortably with one or both hands on your belly.",
                    "Breathe in slowly through your nose for about 3 to 4 seconds. Let your belly rise.",
                    "Breathe out slowly through slightly pursed lips for about 4 to 6 seconds.",
                    "Keep your shoulders and neck relaxed.",
                ],
                "dose": "5 to 10 breaths, a few times a day.",
                "caution": "Stop and rest if you feel light-headed. Do not force the breath.",
            },
            {
                "key": "rib_breath",
                "name": "Side rib breathing",
                "steps": [
                    "Place your hands on the sides of your lower ribs.",
                    "Breathe in slowly through your nose and feel your ribs widen sideways under your hands.",
                    "Breathe out slowly and feel them gently come back in.",
                ],
                "dose": "5 to 10 breaths.",
                "caution": "Keep your shoulders down. Stop if you feel light-headed.",
            },
        ],
    },
}

# Optional: paste your own video link (YouTube or a direct .mp4 URL) for any exercise key,
# for example:  "neck_rotation": "https://www.youtube.com/watch?v=XXXX"
EXERCISE_VIDEOS = {}


# =========================================================
# CUSTOM DESIGN
# =========================================================
st.markdown(
    """
    <style>
    .stApp { background-color: #F4F7FB; color: #263445; }
    .stApp p, .stApp label, .stApp h1, .stApp h2, .stApp h3,
    .stApp span, .stApp div[data-testid="stMetricValue"] { color: #263445; }
    h1, h2, h3 { color: #153A5B !important; }
    [data-testid="stSidebar"] { background-color: #153A5B; }
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span, [data-testid="stSidebar"] h2 { color: white; }
    [data-testid="stMetric"] {
        background-color: white; padding: 16px; border-radius: 12px;
        border: 1px solid #DCE5EF;
    }
    .stButton button { background-color: #167D9A; color: white !important;
        border-radius: 8px; border: none; }
    .stApp .stDownloadButton button, .stApp [data-testid="stDownloadButton"] button {
        background-color: #167D9A; color: white !important; border-radius: 8px; border: none; }
    .stApp .stButton button p, .stApp .stButton button span,
    .stApp .stDownloadButton button p, .stApp .stDownloadButton button span,
    .stApp [data-testid="stDownloadButton"] button p,
    .stApp [data-testid="stDownloadButton"] button span { color: white !important; }
    [data-testid="stAlert"] { border-radius: 10px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# HEADER + SIDEBAR
# =========================================================
st.title("🎙️ VOICEGUARD AI")
st.caption("AI-Powered Post-Thyroidectomy Recovery Monitoring (exploratory prototype)")
st.divider()

def login_sidebar():
    """Sidebar sign-in. Returns the signed-in patient ID or None."""
    sb = st.sidebar
    sb.markdown("### Patient sign-in")
    if st.session_state.get("patient_id"):
        shown = st.session_state.get("patient_name") or st.session_state.patient_id
        sb.success(f"Signed in: {shown}")
        if st.session_state.get("patient_name") and st.session_state.patient_id != "DEMO":
            with sb.expander("Change my name"):
                new_name = st.text_input("New name", key="rename_input", max_chars=40)
                if st.button("Update name"):
                    cleaned = clean_name(new_name)
                    if len(cleaned) < 2:
                        st.error("Please type at least 2 characters.")
                    else:
                        try:
                            save_display_name(st.session_state.patient_id, cleaned)
                            st.session_state.patient_name = cleaned
                            st.rerun()
                        except Exception as error:
                            st.error(f"Could not save the name: {error}")
        if sb.button("Sign out"):
            for k in ("patient_id", "patient_name", "voice_result", "pitch_data", "voice_saved"):
                st.session_state.pop(k, None)
            st.rerun()
    else:
        pid_in = sb.text_input("Patient ID").strip().upper()
        pin_in = sb.text_input("PIN (4 to 8 digits)", type="password")
        sb.caption(
            "First time? Choose an ID and a PIN and press the button: your account is created. "
            "Next time, use the same ID and PIN to see your saved data. "
            "A forgotten PIN cannot be recovered."
        )
        if sb.button("Sign in / Create account"):
            ok, msg = sign_in_or_create(pid_in, pin_in)
            if ok:
                st.session_state.patient_id = pid_in
                st.session_state.login_msg = msg
                st.rerun()
            else:
                sb.error(msg)
        if sb.button("Try the demo patient"):
            try:
                load_demo_data()
                st.session_state.patient_id = "DEMO"
                st.session_state.login_msg = "Demo data loaded (simulated)."
                st.rerun()
            except Exception as error:
                sb.error(f"Could not load demo data: {error}")

    if st.session_state.get("login_msg"):
        sb.caption(st.session_state.pop("login_msg"))

    if _supabase_cfg():
        sb.caption("Storage: cloud database (data is kept between sessions).")
    else:
        sb.warning("Storage: temporary local file. Data may be erased when the app restarts.")
    if st.session_state.get("storage_error"):
        sb.error("Database problem: " + st.session_state["storage_error"][:150])
    return st.session_state.get("patient_id")


patient_id = login_sidebar()
if not patient_id:
    st.info("Sign in from the sidebar (or open the menu on a phone) to start.")
    st.stop()

patient_name = st.session_state.get("patient_name")
if not patient_name:
    try:
        patient_name = get_display_name(patient_id)
    except Exception:
        patient_name = None
    if not patient_name:
        st.header("Welcome! What is your name?")
        st.write("Your name will appear in the app instead of your ID.")
        typed = st.text_input("Your name", max_chars=40)
        if st.button("Save my name"):
            cleaned = clean_name(typed)
            if len(cleaned) < 2:
                st.error("Please type your name (at least 2 characters).")
            else:
                try:
                    save_display_name(patient_id, cleaned)
                    st.session_state.patient_name = cleaned
                    st.rerun()
                except Exception as error:
                    st.error(f"Could not save the name: {error}")
        st.stop()
    st.session_state.patient_name = patient_name

page = st.sidebar.radio(
    "MAIN MENU",
    [
        "Dashboard",
        "Voice Assessment",
        "Neck Movement",
        "AI Analysis",
        "Recovery Progress",
        "Exercise Library",
    ],
)

st.sidebar.divider()
st.sidebar.caption(
    "Seek urgent medical care for difficulty breathing, stridor, rapidly "
    "increasing neck swelling, or new/severe difficulty swallowing."
)



# =========================================================
# DASHBOARD
# =========================================================
if page == "Dashboard":
    st.header(f"Welcome, {patient_name} 👋")
    st.write(f"Patient: **{patient_name}** (ID: {patient_id})")

    voice = load_history(VOICE_FILE, patient_id)
    neck = load_history(NECK_FILE, patient_id)
    findings, _, _ = compute_findings(patient_id)

    if voice.empty and neck.empty:
        status = "Not assessed"
    elif findings:
        status = "Review suggested"
    else:
        status = "No notable change"

    c1, c2, c3 = st.columns(3)
    c1.metric("Voice Assessments", len(voice))
    c2.metric("Movement Assessments", len(neck))
    c3.metric("Recovery Status", status)

    if findings:
        st.warning("Changes were detected. Open 'AI Analysis' for details.")
    else:
        st.info("Start by completing a voice or neck movement assessment.")

    st.caption("Only perform assessments that your surgeon or clinician has cleared.")


# =========================================================
# VOICE ASSESSMENT
# =========================================================
elif page == "Voice Assessment":
    st.header("🎤 Voice Assessment")
    st.info(
        "Only do these tests if your clinician has cleared you to speak. "
        "Record in a quiet room, with the same microphone and distance each time."
    )

    test_type = st.selectbox("Test type", TEST_TYPES)
    prompts = {
        TEST_TYPES[0]: "Take a breath and say 'aaa' at a comfortable, steady pitch for as long as is comfortable.",
        TEST_TYPES[1]: "Read the same short sentence every time, at a normal speaking volume.",
        TEST_TYPES[2]: "Glide your voice gently from a low note to a high note (only if comfortable).",
    }
    st.write(prompts[test_type])

    audio = st.audio_input("Record your voice")

    if audio is not None:
        st.audio(audio)
        if st.button("Analyze Voice"):
            try:
                y, sr = librosa.load(io.BytesIO(audio.getvalue()), sr=None, mono=True)
                if y.size == 0 or sr <= 0:
                    st.error("The recording is empty. Please record again.")
                else:
                    with st.spinner("Analyzing..."):
                        feats, vf0, warns = extract_voice_features(y, sr)
                    for w in warns:
                        st.warning(w)
                    if feats is not None:
                        feats.update(
                            {
                                "Patient_ID": patient_id,
                                "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "Test_Type": test_type,
                            }
                        )
                        st.session_state.voice_result = feats
                        st.session_state.pitch_data = vf0.tolist()
                        st.session_state.voice_saved = False
                    else:
                        st.session_state.pop("voice_result", None)
            except Exception as error:
                st.error(f"Analysis failed: {error}")

    if "voice_result" in st.session_state:
        r = st.session_state.voice_result
        st.success("Analysis completed!")
        st.subheader(f"Results: {r['Test_Type']}")

        a, b, c = st.columns(3)
        a.metric("Phonation (longest voiced run)", f"{r['Phonation_sec']:.2f} s")
        b.metric("Median Pitch", f"{r['Median_Pitch_Hz']:.1f} Hz")
        c.metric("Pitch Range", f"{r['Pitch_Range_Hz']:.1f} Hz")

        d, e, f, g = st.columns(4)
        d.metric("Pitch Variability", f"{r['Pitch_Variability_percent']:.2f}%")
        e.metric("Jitter", "n/a" if pd.isna(r["Jitter_percent"]) else f"{r['Jitter_percent']:.2f}%")
        f.metric("Shimmer", "n/a" if pd.isna(r["Shimmer_percent"]) else f"{r['Shimmer_percent']:.2f}%")
        g.metric("HNR", "n/a" if pd.isna(r["HNR_dB"]) else f"{r['HNR_dB']:.1f} dB")

        st.subheader("Pitch contour")
        st.line_chart(st.session_state.pitch_data)

        if st.button("Save Results", disabled=st.session_state.get("voice_saved", False)):
            try:
                save_row(VOICE_FILE, r)
                st.session_state.voice_saved = True
                st.success("Assessment saved successfully!")
            except Exception as error:
                st.error(f"Could not save assessment: {error}")

        if st.session_state.get("voice_saved", False):
            st.caption("This assessment has been saved.")

        st.caption(
            "Exploratory acoustic measurements only. Jitter and shimmer are affected by "
            "microphone quality and noise. Not a clinical diagnosis."
        )


# =========================================================
# NECK MOVEMENT
# =========================================================
elif page == "Neck Movement":
    st.header("📷 Neck Movement Assessment")
    st.info(
        "Only do these movements if your surgeon/physiotherapist has cleared them. "
        "Sit facing the camera, keep a comfortable neutral position while calibrating, "
        "then move slowly: look down, look up, turn right, turn left, tilt to each shoulder. "
        "Stop if you feel pain, dizziness, or tingling."
    )

    st.markdown(
        "**Movement order (about 3 seconds each, return to neutral between movements):** "
        "1) look down, 2) look up, 3) turn right, 4) turn left, 5) tilt right, 6) tilt left."
    )

    with st.expander("Direction settings (check these once by testing)"):
        swap_lr = st.checkbox("Swap left/right (use if turning RIGHT shows as LEFT)")
        swap_fe = st.checkbox("Swap flexion/extension (use if looking DOWN shows as extension)")

    rtc_cfg = get_rtc_configuration()
    has_turn = any("username" in s for s in rtc_cfg["iceServers"])
    if not has_turn:
        st.caption(
            "Camera relay (TURN) is not set up. If the video stays on 'Connection is taking longer "
            "than expected', the app owner needs to add a TURN server in the app Secrets."
        )

    ctx = webrtc_streamer(
        key="neck_movement",
        video_processor_factory=NeckMovementProcessor,
        media_stream_constraints={
            "video": {
                "facingMode": "user",
                "width": {"ideal": 640},
                "height": {"ideal": 480},
                "frameRate": {"ideal": 24},
            },
            "audio": False,
        },
        rtc_configuration=rtc_cfg,
        async_processing=True,
        video_html_attrs={
            "style": {"width": "100%", "transform": "scaleX(-1)"},  # selfie (mirrored) view
            "controls": False,
            "autoPlay": True,
            "muted": True,
        },
    )

    playing = bool(ctx.state.playing)
    if playing and not st.session_state.get("neck_was_playing", False):
        # The camera was (re)started: begin a fresh assessment and drop the old results.
        st.session_state.pop("neck_result", None)
        st.session_state.pop("neck_saved", None)
    st.session_state["neck_was_playing"] = playing

    st.write("Camera Status:", "Running" if playing else "Not running")
    processor = ctx.video_processor

    def neck_snapshot(p):
        return {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Flexion": float(p.max_flexion),
            "Extension": float(p.max_extension),
            "Right_Rotation": float(p.max_right_rotation),
            "Left_Rotation": float(p.max_left_rotation),
            "Tilt_Right": float(p.max_tilt_right),
            "Tilt_Left": float(p.max_tilt_left),
            "movement": pd.DataFrame(p.movement_data),
        }

    if processor is None:
        st.info("Waiting for the camera. Allow camera permission and press Start.")
    else:
        processor.swap_left_right = swap_lr
        processor.swap_flex_ext = swap_fe

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            if st.button("Start Assessment", disabled=processor.assessment_active):
                if processor.is_calibrated:
                    processor.start_assessment()
                    st.session_state.pop("neck_result", None)
                    st.session_state.pop("neck_saved", None)
                    st.rerun()
                else:
                    st.warning("Still calibrating. Wait for 'Neutral position calibrated' below, then press Start again.")
        with col_b:
            if st.button("Stop Assessment", disabled=not processor.assessment_active):
                processor.stop_assessment()
                st.session_state["neck_result"] = neck_snapshot(processor)
                st.session_state["neck_saved"] = False
                st.rerun()
        with col_c:
            if st.button("Recalibrate", disabled=processor.assessment_active):
                processor.reset_calibration()
                st.rerun()

        st.write(
            "Assessment status:",
            "Active" if processor.assessment_active
            else "Completed" if processor.assessment_completed
            else "Not started",
        )

        @st.fragment(run_every=1)
        def live_results(p):
            if not p.is_calibrated:
                left = max(0, p.CALIBRATION_FRAMES - len(p.calibration_samples))
                if p.face_visible:
                    st.warning(f"Calibrating: about {left} more frames. Keep your head still.")
                else:
                    st.error("No face detected. Face the camera in good light, with your whole face visible.")
                return
            st.success("Neutral position calibrated. You can start the assessment.")
            st.caption(
                f"Face detected: {'yes' if p.face_visible else 'NO'} | "
                f"live: pitch {p.current_pitch:+.1f}°, yaw {p.current_yaw:+.1f}°, roll {p.current_roll:+.1f}°"
            )
            if p.assessment_active:
                # Keep a running copy, so the results survive even if the camera is closed.
                st.session_state["neck_result"] = neck_snapshot(p)
                st.subheader("Assessment in progress: maximum movement so far")
                c1, c2, c3 = st.columns(3)
                c1.metric("Flexion", f"{p.max_flexion:.1f}°")
                c1.metric("Extension", f"{p.max_extension:.1f}°")
                c2.metric("Right rotation", f"{p.max_right_rotation:.1f}°")
                c2.metric("Left rotation", f"{p.max_left_rotation:.1f}°")
                c3.metric("Tilt right", f"{p.max_tilt_right:.1f}°")
                c3.metric("Tilt left", f"{p.max_tilt_left:.1f}°")

        live_results(processor)

    # ---- results of the last assessment: stay on screen until the camera is started again
    result = st.session_state.get("neck_result")
    assessment_running = processor is not None and processor.assessment_active
    if result is not None and not assessment_running:
        st.divider()
        st.subheader("Assessment results")
        st.caption(
            f"Recorded {result['time']}. These results stay here until you start the camera again, "
            "so you can save them."
        )
        r1, r2, r3 = st.columns(3)
        r1.metric("Flexion", f"{result['Flexion']:.1f}°")
        r1.metric("Extension", f"{result['Extension']:.1f}°")
        r2.metric("Right rotation", f"{result['Right_Rotation']:.1f}°")
        r2.metric("Left rotation", f"{result['Left_Rotation']:.1f}°")
        r3.metric("Tilt right", f"{result['Tilt_Right']:.1f}°")
        r3.metric("Tilt left", f"{result['Tilt_Left']:.1f}°")
        if not result["movement"].empty:
            st.markdown("**Recorded movement (last 100 samples)**")
            st.line_chart(result["movement"].tail(100))

        already_saved = st.session_state.get("neck_saved", False)
        if st.button("Save Neck Assessment", disabled=already_saved):
            try:
                save_row(
                    NECK_FILE,
                    {
                        "Patient_ID": patient_id,
                        "Date": result["time"],
                        "Flexion": result["Flexion"],
                        "Extension": result["Extension"],
                        "Right_Rotation": result["Right_Rotation"],
                        "Left_Rotation": result["Left_Rotation"],
                        "Tilt_Right": result["Tilt_Right"],
                        "Tilt_Left": result["Tilt_Left"],
                    },
                )
                st.session_state["neck_saved"] = True
                st.success("Neck assessment saved.")
            except Exception as error:
                st.error(f"Could not save assessment: {error}")
        if already_saved:
            st.caption("This assessment has been saved.")

    st.caption(
        "Camera-based estimates relative to your neutral position. They are not validated "
        "clinical range-of-motion measurements and do not replace a goniometer or a physiotherapist."
    )


# =========================================================
# AI ANALYSIS
# =========================================================
elif page == "AI Analysis":
    st.header("🧠 AI Recovery Analysis")
    st.write(f"Patient: **{patient_name}** (ID: {patient_id})")

    st.subheader("Safety check")
    red_flags = [
        st.checkbox("Difficulty breathing or noisy breathing (stridor)"),
        st.checkbox("Rapidly increasing neck swelling"),
        st.checkbox("New or severe difficulty swallowing"),
    ]
    if any(red_flags):
        st.error(
            "These symptoms need urgent medical assessment. "
            "Do not wait for this app or a physiotherapy session."
        )
        st.stop()

    use_ml = st.checkbox(
        "Include ML anomaly check (demo: trained on simulated normal variation)",
        value=False,
        disabled=not HAS_SKLEARN,
    )

    findings, notes, ml_note = compute_findings(patient_id, use_ml=use_ml)

    st.subheader("Recovery report")
    for n in notes:
        st.info(n)

    if findings:
        st.warning("Review suggested")
        for f in findings:
            st.write("•", f)
        st.write(
            "**Suggested next step:** review the findings with the treating clinician and "
            "repeat the standardized assessment under the same recording conditions."
        )
    elif not notes:
        st.success("No notable change from baseline detected.")

    if ml_note:
        st.caption(ml_note)

    voice_profiles, neck_profile = get_profiles(patient_id)
    if voice_profiles or not neck_profile.empty:
        st.subheader("Patient Recovery Profile")
        for test, prof in voice_profiles.items():
            st.markdown(f"**Voice: {test}**")
            show_table(prof)
        if not neck_profile.empty:
            st.markdown("**Neck movement** (baseline = first assessment)")
            show_table(neck_profile)

        if HAS_REPORTLAB:
            st.download_button(
                "Download report (PDF)",
                report_pdf(patient_name, patient_id, findings, notes, voice_profiles, neck_profile, ml_note),
                file_name=f"voiceguard_report_{patient_id}.pdf",
                mime="application/pdf",
            )
        else:
            st.info("PDF export needs the 'reportlab' package (add it to requirements.txt).")

    st.caption(
        "Exploratory monitoring tool, not a diagnosis. Persistent hoarseness or voice change "
        "needs medical evaluation (e.g., laryngoscopy). Thresholds are demo values and have "
        "not been validated on post-thyroidectomy patients."
    )


# =========================================================
# RECOVERY PROGRESS
# =========================================================
elif page == "Recovery Progress":
    st.header("📈 Recovery Progress")
    st.write(f"Patient: **{patient_name}** (ID: {patient_id})")

    voice = load_history(VOICE_FILE, patient_id)
    neck = load_history(NECK_FILE, patient_id)

    if voice.empty and neck.empty:
        st.info("No saved assessments yet for this patient.")

    if not voice.empty or not neck.empty:
        st.caption(
            "Tip: on the free cloud server saved data can be erased when the app restarts. "
            "Download your data to keep a copy."
        )
        if HAS_REPORTLAB:
            p1, p2 = st.columns(2)
            if not voice.empty:
                p1.download_button(
                    "Download voice history (PDF)",
                    history_pdf("Voice History", patient_name, patient_id, voice, VOICE_PDF_COLUMNS),
                    file_name=f"voice_history_{patient_id}.pdf",
                    mime="application/pdf",
                )
            if not neck.empty:
                p2.download_button(
                    "Download neck history (PDF)",
                    history_pdf("Neck Movement History", patient_name, patient_id, neck, NECK_PDF_COLUMNS),
                    file_name=f"neck_history_{patient_id}.pdf",
                    mime="application/pdf",
                )
        else:
            st.info("PDF export needs the 'reportlab' package (add it to requirements.txt).")

    if not voice.empty:
        if "Test_Type" not in voice.columns:
            voice["Test_Type"] = TEST_TYPES[0]
        st.subheader("Voice assessments")
        test = st.selectbox("Test type", sorted(voice["Test_Type"].unique()))
        sub = voice[voice["Test_Type"] == test].reset_index(drop=True)
        st.dataframe(sub, use_container_width=True)

        for col, label in FEATURES_BY_TEST.get(test, {}).items():
            if col in sub.columns and sub[col].notna().any():
                st.markdown(f"**{label} over time**")
                st.line_chart(sub.set_index("Date")[col])

    if not neck.empty:
        st.subheader("Neck movement assessments")
        st.dataframe(neck, use_container_width=True)
        cols = [c for c in ["Flexion", "Extension", "Right_Rotation", "Left_Rotation"] if c in neck.columns]
        st.line_chart(neck.set_index("Date")[cols])


# =========================================================
# EXERCISE LIBRARY
# =========================================================
elif page == "Exercise Library":
    st.header("🏋️ Exercise Library")
    st.warning(
        "Exercises must only be started after clearance from your surgeon and assessment by a "
        "physiotherapist. Not every thyroidectomy patient needs the same exercises. "
        "Specialized voice therapy belongs to a Speech-Language Pathologist."
    )
    st.error(
        "Do not exercise, and get urgent medical care, if you have difficulty breathing, noisy "
        "breathing (stridor), rapidly increasing neck swelling, or new or severe difficulty swallowing."
    )

    with st.spinner("Preparing the exercise animations..."):
        for _cat in EXERCISE_LIBRARY.values():
            for _ex in _cat["exercises"]:
                exercise_gif_b64(_ex["key"])

    tabs = st.tabs(list(EXERCISE_LIBRARY.keys()))
    for tab, (category, info) in zip(tabs, EXERCISE_LIBRARY.items()):
        with tab:
            st.caption(info["goal"])
            for ex in info["exercises"]:
                with st.container(border=True):
                    left, right = st.columns([1, 1.5])
                    with left:
                        st.markdown(
                            f'<img src="data:image/gif;base64,{exercise_gif_b64(ex["key"])}" '
                            f'alt="{ex["name"]}" style="width:100%;max-width:320px;border-radius:12px;">',
                            unsafe_allow_html=True,
                        )
                    with right:
                        st.markdown(f"**{ex['name']}**")
                        for i, step in enumerate(ex["steps"], start=1):
                            st.write(f"{i}. {step}")
                        st.caption(f"**How much:** {ex['dose']}")
                        st.caption(f"**Be careful:** {ex['caution']}")
                    video = EXERCISE_VIDEOS.get(ex["key"])
                    if video:
                        st.video(video)

    st.caption(
        "Stop any exercise if you feel pain, dizziness, tingling or pulling at the scar. "
        "The repetitions shown are general examples; follow your own physiotherapist's plan."
    )
