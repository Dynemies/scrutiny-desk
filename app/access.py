# ======================= PATHS (edit if needed) =======================
LOG_FILE = "./access_log.jsonl"          # who opened what, and notifications queued (one JSON line each)
# =======================================================================
"""Access log and notification queue. Every application opened is logged with the role and time;
notifications are queued here (no e-mail/SMS gateway is connected in the prototype)."""
import datetime as dt
import json
from pathlib import Path

import streamlit as st

HERE = Path(__file__).resolve().parent
LOG = HERE / LOG_FILE


def _write(entry):
    entry = {"at": dt.datetime.now().isoformat(timespec="seconds"), **entry}
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def record(who, app_id):
    """Log an application view once per (role, application) change in this browser session."""
    key = (who, app_id)
    if st.session_state.get("_last_access") != key:
        st.session_state["_last_access"] = key
        _write({"kind": "view", "who": who, "what": app_id})


def notify(who, items):
    for it in items:
        _write({"kind": "notification", "who": who, "what": it["application"], "to": it["to"], "text": it["text"]})


def read(limit=200):
    if not LOG.exists():
        return []
    lines = LOG.read_text(encoding="utf-8").splitlines()[-limit:]
    return [json.loads(l) for l in reversed(lines) if l.strip()]
