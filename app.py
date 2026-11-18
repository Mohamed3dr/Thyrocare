import io
import os
from datetime import datetime, timedelta

import cv2
import librosa
import mediapipe as mp
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer

try:
    import parselmouth
    from parselmouth.praat import call
    HAS_PRAAT = True
except Exception:
    HAS_PRAAT = False

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
        self.neutral = None  # (pitch, yaw, roll)
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
        if self._tick % 2 == 0:  # analyse every 2nd frame to keep video smooth
            self._analyze(image)  # measurements always use the unflipped image
        return frame

    def _analyze(self, image):
        self.frame_count += 1

        small = cv2.resize(image, (320, 240), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            self.face_visible = False
            return

        self.face_visible = True
        face = results.multi_face_landmarks[0]
        pnp = head_angles(face, 320, 240)
        if pnp is None:
            return

        pitch, yaw, _ = pnp
        roll = roll_from_eyes(face, 320, 240)  # more stable than the solvePnP roll
        angles = (pitch, yaw, roll)

        if not self.is_calibrated:
            self.calibration_samples.append(angles)
            if len(self.calibration_samples) >= self.CALIBRATION_FRAMES:
                # Wrap-safe mean: average differences from the first sample.
                ref = np.array(self.calibration_samples[0])
                diffs = wrap180(np.array(self.calibration_samples) - ref)
                self.neutral = tuple(ref + diffs.mean(axis=0))
                self.is_calibrated = True
            return

        rel_pitch = float(wrap180(pitch - self.neutral[0]))
        rel_yaw = float(wrap180(yaw - self.neutral[1]))
        rel_roll = float(wrap180(roll - self.neutral[2]))

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
            rel = 0.5 * rel + 0.5 * self.last_rel
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
def save_row(path, row):
    df = pd.DataFrame([row])
    if os.path.exists(path):
        df = pd.concat([pd.read_csv(path), df], ignore_index=True)
    df.to_csv(path, index=False)


def load_history(path, patient_id):
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
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    if "Patient_ID" in df.columns:
        df = df[df["Patient_ID"].astype(str) != str(patient_id)]
        df.to_csv(path, index=False)


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

patient_id = st.sidebar.text_input("Patient ID", value="P001").strip() or "P001"

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

if st.sidebar.button("Load demo patient (DEMO)"):
    load_demo_data()
    st.sidebar.success("Demo data created. Enter Patient ID: DEMO")


# =========================================================
# DASHBOARD
# =========================================================
if page == "Dashboard":
    st.header("Patient Dashboard")
    st.write(f"Patient: **{patient_id}**")

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

    with st.expander("Direction settings (check these once by testing)"):
        swap_lr = st.checkbox("Swap left/right (use if turning RIGHT shows as LEFT)")
        swap_fe = st.checkbox("Swap flexion/extension (use if looking DOWN shows as extension)")

    ctx = webrtc_streamer(
        key="neck_movement",
        video_processor_factory=NeckMovementProcessor,
        media_stream_constraints={
            "video": {
                "facingMode": "user",
                "width": {"ideal": 480},
                "height": {"ideal": 360},
                "frameRate": {"ideal": 30, "min": 15},
            },
            "audio": False,
        },
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        async_processing=True,
        video_html_attrs={
            "style": {"width": "100%", "transform": "scaleX(-1)"},  # selfie (mirrored) view
            "controls": False,
            "autoPlay": True,
            "muted": True,
        },
    )

    st.write("Camera Status:", "Running" if ctx.state.playing else "Not running")
    processor = ctx.video_processor

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
                    st.rerun()
                else:
                    st.warning("Still calibrating. Wait for 'Neutral position calibrated' below, then press Start again.")
        with col_b:
            if st.button("Stop Assessment", disabled=not processor.assessment_active):
                processor.stop_assessment()
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
            st.subheader("Maximum movement (degrees from neutral)")
            c1, c2, c3 = st.columns(3)
            c1.metric("Flexion", f"{p.max_flexion:.1f}°")
            c1.metric("Extension", f"{p.max_extension:.1f}°")
            c2.metric("Right rotation", f"{p.max_right_rotation:.1f}°")
            c2.metric("Left rotation", f"{p.max_left_rotation:.1f}°")
            c3.metric("Tilt right", f"{p.max_tilt_right:.1f}°")
            c3.metric("Tilt left", f"{p.max_tilt_left:.1f}°")

        live_results(processor)

        st.caption(
            "Camera-based estimates relative to your neutral position. They are not validated "
            "clinical range-of-motion measurements and do not replace a goniometer or a physiotherapist."
        )

        if processor.movement_data:
            df = pd.DataFrame(processor.movement_data)
            st.subheader("Recorded movement (last 100 samples)")
            st.line_chart(df.tail(100))

        if processor.assessment_completed and not processor.assessment_active:
            if st.button("Save Neck Assessment"):
                save_row(
                    NECK_FILE,
                    {
                        "Patient_ID": patient_id,
                        "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Flexion": processor.max_flexion,
                        "Extension": processor.max_extension,
                        "Right_Rotation": processor.max_right_rotation,
                        "Left_Rotation": processor.max_left_rotation,
                        "Tilt_Right": processor.max_tilt_right,
                        "Tilt_Left": processor.max_tilt_left,
                    },
                )
                st.success("Neck assessment saved.")


# =========================================================
# AI ANALYSIS
# =========================================================
elif page == "AI Analysis":
    st.header("🧠 AI Recovery Analysis")
    st.write(f"Patient: **{patient_id}**")

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
    st.write(f"Patient: **{patient_id}**")

    voice = load_history(VOICE_FILE, patient_id)
    neck = load_history(NECK_FILE, patient_id)

    if voice.empty and neck.empty:
        st.info("No saved assessments yet for this patient.")

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

    library = {
        "Gentle Neck ROM": "Slow, pain-free movements within the range your clinician approved.",
        "Shoulder Mobility": "Gentle shoulder rolls and arm raises to reduce stiffness.",
        "Postural Correction": "Keep the head stacked over the shoulders; avoid forward-head posture.",
        "Breathing Exercises": "Slow diaphragmatic breathing to support breath control.",
    }
    for name, desc in library.items():
        with st.expander(name):
            st.write(desc)
            st.caption("Add a video or animation here. Stop if you feel pain, dizziness or tingling.")
