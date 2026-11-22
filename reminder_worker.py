"""VOICEGUARD AI - optional Telegram reminder worker.

Runs outside the Streamlit app (for example from GitHub Actions every 10 minutes). It reads the
patients' medicines and appointments from the Supabase database and sends a Telegram message when a
dose or an appointment is due. A message is sent only once (it is logged in the database).

Environment variables: SUPABASE_URL, SUPABASE_KEY (the secret/service key), TELEGRAM_BOT_TOKEN
Optional: WINDOW_MIN (how many minutes after the scheduled time a dose reminder may still be sent; default 20)
"""
import os
import re
from datetime import datetime, timedelta

import requests

try:
    from zoneinfo import ZoneInfo
    TZ = ZoneInfo("Africa/Cairo")
except Exception:  # pragma: no cover
    TZ = None

WINDOW_MIN = int(os.environ.get("WINDOW_MIN", "20"))


def now_local():
    return datetime.now(TZ).replace(tzinfo=None) if TZ else datetime.now()


def parse_times(text):
    out = []
    for h, m in re.findall(r"(\d{1,2}):(\d{2})", str(text or "")):
        if 0 <= int(h) < 24 and 0 <= int(m) < 60:
            out.append(f"{int(h):02d}:{m}")
    return sorted(set(out))


def to_date(text):
    try:
        return datetime.strptime(str(text)[:10], "%Y-%m-%d").date()
    except Exception:
        return None


def latest(rows, *keys):
    """Keep the newest record for each combination of data[keys] (rows are oldest first)."""
    out = {}
    for r in rows:
        out[tuple(r["data"].get(k) for k in keys)] = r
    return out


MESSAGES = {
    "en": {
        "dose": "💊 Time to take {name} {dose} (scheduled {time}). Open VOICEGUARD AI to mark it as taken.",
        "appt24": "📅 Reminder: {title} tomorrow at {time}.",
        "appt2": "📅 Reminder: {title} today at {time}.",
    },
    "ar": {
        "dose": "💊 حان وقت تناول {name} {dose} (الموعد {time}). افتح VOICEGUARD AI لتسجيل أنك تناولته.",
        "appt24": "📅 تذكير: {title} غدًا الساعة {time}.",
        "appt2": "📅 تذكير: {title} اليوم الساعة {time}.",
    },
}


def due_messages(medications, doses, appointments, profiles, reminded, now):
    """Pure function. Inputs are lists of records {"patient_id", "data"} (oldest first).
    Returns a list of (patient_id, chat_id, key, text) that should be sent now."""
    chat, lang = {}, {}
    for r in profiles:
        d = r["data"]
        if d.get("Telegram"):
            chat[r["patient_id"]] = str(d["Telegram"]).strip()
        if d.get("Lang") in ("ar", "en"):
            lang[r["patient_id"]] = d["Lang"]
    done = {}
    for r in doses:
        d = r["data"]
        done[(r["patient_id"], d.get("Med_ID"), d.get("Scheduled"))] = d.get("Status")

    out = []
    meds = {}
    for r in medications:
        meds[(r["patient_id"], r["data"].get("Med_ID"))] = r
    for (pid, mid), r in meds.items():
        d = r["data"]
        if str(d.get("Active", "yes")).lower() == "no" or pid not in chat:
            continue
        start, end = to_date(d.get("Start")), to_date(d.get("End"))
        today = now.date()
        if (start and today < start) or (end and today > end):
            continue
        for tm in parse_times(d.get("Times")):
            slot = datetime.combine(today, datetime.strptime(tm, "%H:%M").time())
            minutes_late = (now - slot).total_seconds() / 60
            sched = f"{slot:%Y-%m-%d %H:%M}"
            if not (0 <= minutes_late <= WINDOW_MIN):
                continue
            if done.get((pid, mid, sched)) in ("taken", "skipped"):
                continue
            key = f"dose|{mid}|{sched}"
            if (pid, key) in reminded:
                continue
            text = MESSAGES[lang.get(pid, "en")]["dose"].format(name=d.get("Name", ""), dose=d.get("Dose", ""), time=tm)
            out.append((pid, chat[pid], key, text.replace("  ", " ")))

    appts = {}
    for r in appointments:
        appts[(r["patient_id"], r["data"].get("Appt_ID"))] = r
    for (pid, aid), r in appts.items():
        d = r["data"]
        if d.get("Status") != "upcoming" or pid not in chat:
            continue
        try:
            when = datetime.strptime(str(d.get("When"))[:16], "%Y-%m-%d %H:%M")
        except Exception:
            continue
        title = d.get("Title") or d.get("Type") or "appointment"
        hours = (when - now).total_seconds() / 3600
        for tag, lo, hi in (("appt24", 23, 24.5), ("appt2", 1.5, 2.5)):
            key = f"{tag}|{aid}"
            if lo <= hours <= hi and (pid, key) not in reminded:
                text = MESSAGES[lang.get(pid, "en")][tag].format(title=title, time=f"{when:%H:%M}")
                out.append((pid, chat[pid], key, text))
    return out


# ------------------------------------------------------------------ database + Telegram (not used by the tests)
def fetch(url, key, kind):
    rows, offset = [], 0
    while True:
        r = requests.get(
            f"{url}/rest/v1/records",
            headers={"apikey": key, "Authorization": f"Bearer {key}"} if not key.startswith("sb_") else {"apikey": key},
            params={"kind": f"eq.{kind}", "select": "patient_id,date,data", "order": "date.asc,id.asc",
                    "limit": "1000", "offset": str(offset)},
            timeout=30,
        )
        r.raise_for_status()
        chunk = r.json()
        rows.extend(chunk)
        if len(chunk) < 1000:
            return rows
        offset += 1000


def main():
    url, key, bot = os.environ["SUPABASE_URL"].rstrip("/"), os.environ["SUPABASE_KEY"], os.environ["TELEGRAM_BOT_TOKEN"]
    now = now_local()
    headers = {"apikey": key, "Content-Type": "application/json", "Prefer": "return=minimal"}
    if not key.startswith("sb_"):
        headers["Authorization"] = f"Bearer {key}"
    meds, doses = fetch(url, key, "medication"), fetch(url, key, "dose")
    appts, profiles, reminded_rows = fetch(url, key, "appointment"), fetch(url, key, "profile"), fetch(url, key, "reminder")
    reminded = {(r["patient_id"], r["data"].get("Key")) for r in reminded_rows}
    to_send = due_messages(meds, doses, appts, profiles, reminded, now)
    sent = 0
    for pid, chat_id, rkey, text in to_send:
        resp = requests.post(f"https://api.telegram.org/bot{bot}/sendMessage", json={"chat_id": chat_id, "text": text}, timeout=30)
        if resp.ok:
            requests.post(f"{url}/rest/v1/records", headers=headers, timeout=30, json={
                "patient_id": pid, "kind": "reminder", "date": f"{now:%Y-%m-%d %H:%M:%S}",
                "data": {"Patient_ID": pid, "Date": f"{now:%Y-%m-%d %H:%M:%S}", "Key": rkey}})
            sent += 1
        else:
            print("Telegram error for", pid, resp.status_code, resp.text[:200])
    print(f"{now:%Y-%m-%d %H:%M} checked; {len(to_send)} due, {sent} sent")


if __name__ == "__main__":
    main()
