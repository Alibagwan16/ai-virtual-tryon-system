"""
sheets_db.py
------------
A tiny "database" layer backed by a Google Sheet, used by auth.py and
credits.py instead of (or alongside) local JSON files.

WHY THIS EXISTS: on Streamlit Community Cloud (and similar), the live
server's local filesystem is NOT persistent across redeploys — every time
you push new code to GitHub, the server is rebuilt fresh from the repo.
Any data that was only ever written to a local JSON file on the OLD server
(a shop signing up, a credit purchase, usage counters) is gone after that
rebuild, because it was never part of the git repo. A Google Sheet lives
completely outside of GitHub/Streamlit Cloud, so it survives redeploys,
restarts, and sleep/wake cycles untouched.

Each "table" is one worksheet (tab) in a single Google Sheet. The exact
tables/headers are defined by the callers (auth.py, credits.py) — this
module only knows how to read/write arbitrary named tables generically.

SAFE BY DEFAULT: every public function here returns None (for reads) or
False (for writes) if Sheets hasn't been configured yet, or if a request
to Google fails for any reason (network hiccup, bad credentials, etc).
Callers are expected to fall back to their OLD local-JSON-file behavior
in that case — so the app works exactly as it did before, right up until
you finish the one-time Sheets setup (see README.md), and it keeps
working even if Sheets is briefly unreachable afterwards.

SETUP (see README.md for the full walkthrough with screensh? — short
version):
    1. Create a Google Cloud service account, enable the Sheets + Drive
       APIs, download its JSON key.
    2. Create a Google Sheet, share it (Editor access) with the service
       account's email address (ends in ...gserviceaccount.com).
    3. Put the JSON key's contents under [gcp_service_account] in
       .streamlit/secrets.toml (or GOOGLE_SERVICE_ACCOUNT_JSON in .env),
       and the sheet's ID (from its URL) in GOOGLE_SHEET_ID in .env.
    4. (Optional, recommended) run `python migrate_to_sheets.py` once to
       copy any existing local shops.json/usage.json/credits.json/
       payment_requests.json data into the new sheet.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_cache: Dict[str, tuple] = {}  # table name -> (fetched_at, records)
_CACHE_TTL_SECONDS = 8  # short enough that admin approvals show up almost
                         # immediately, long enough to avoid hammering the
                         # Sheets API on every Streamlit rerun

_client = None
_sheet = None
_init_attempted = False
_init_error: Optional[str] = None


# --------------------------------------------------------------------------
# Connection setup
# --------------------------------------------------------------------------
def _get_credentials_info() -> Optional[dict]:
    """Looks for Google service-account credentials, in order:
    1. st.secrets["gcp_service_account"] (recommended on Streamlit Cloud).
    2. GOOGLE_SERVICE_ACCOUNT_JSON env var — the full JSON key as one string.
    3. GOOGLE_SERVICE_ACCOUNT_FILE env var — a path to the JSON key file.
    Returns None if none of these are set."""
    try:
        import streamlit as st
        if "gcp_service_account" in st.secrets:
            return dict(st.secrets["gcp_service_account"])
    except Exception:  # noqa: BLE001
        pass

    raw = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip()
    if raw:
        try:
            return json.loads(raw)
        except Exception:  # noqa: BLE001
            logger.warning("GOOGLE_SERVICE_ACCOUNT_JSON is set but isn't valid JSON.")
            return None

    path = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE", "").strip()
    if path and os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:  # noqa: BLE001
            logger.warning("GOOGLE_SERVICE_ACCOUNT_FILE=%s couldn't be read as JSON.", path)
            return None

    return None


def _get_sheet_identifier() -> Optional[dict]:
    """Returns {'by': 'id'|'name', 'value': ...} for which sheet to open,
    or None if neither GOOGLE_SHEET_ID nor GOOGLE_SHEET_NAME is set."""
    sheet_id = os.getenv("GOOGLE_SHEET_ID", "").strip()
    if sheet_id:
        return {"by": "id", "value": sheet_id}
    sheet_name = os.getenv("GOOGLE_SHEET_NAME", "").strip()
    if sheet_name:
        return {"by": "name", "value": sheet_name}
    return None


def is_configured() -> bool:
    """True once both credentials AND a sheet reference are set — does NOT
    verify they actually work (that only happens on first real use)."""
    return _get_credentials_info() is not None and _get_sheet_identifier() is not None


def init_error() -> Optional[str]:
    """The last connection error message, if any — handy for an
    admin-facing diagnostic line. None if never attempted or if it
    succeeded."""
    return _init_error


def _init() -> bool:
    """Lazily connects to Google Sheets once per process. Returns True if
    ready to use, False if not configured or the connection failed (in
    which case every public function below safely no-ops)."""
    global _client, _sheet, _init_attempted, _init_error
    if _sheet is not None:
        return True
    if _init_attempted:
        return False  # already tried and failed earlier this process run

    creds_info = _get_credentials_info()
    sheet_ref = _get_sheet_identifier()
    if not creds_info or not sheet_ref:
        _init_attempted = True
        return False  # not configured yet — not an error, just not set up

    try:
        import gspread
        from google.oauth2.service_account import Credentials

        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive.file",
        ]
        creds = Credentials.from_service_account_info(creds_info, scopes=scopes)
        client = gspread.authorize(creds)
        sheet = (
            client.open_by_key(sheet_ref["value"])
            if sheet_ref["by"] == "id"
            else client.open(sheet_ref["value"])
        )
        _client = client
        _sheet = sheet
        _init_attempted = True
        return True
    except Exception as e:  # noqa: BLE001
        _init_error = str(e)
        _init_attempted = True
        logger.warning(
            "Google Sheets storage is configured but couldn't connect (%s) — "
            "falling back to local JSON files for this run.", e,
        )
        return False


def _get_or_create_worksheet(name: str, headers: List[str]):
    assert _sheet is not None
    try:
        ws = _sheet.worksheet(name)
    except Exception:  # noqa: BLE001
        ws = _sheet.add_worksheet(title=name, rows=200, cols=max(10, len(headers)))
        ws.append_row(headers)
        return ws
    if not ws.row_values(1):
        ws.append_row(headers)
    return ws


def _invalidate(table: str) -> None:
    with _lock:
        _cache.pop(table, None)


# --------------------------------------------------------------------------
# Public read/write API — generic, table-name + headers driven
# --------------------------------------------------------------------------
def get_rows(table: str, headers: List[str], force_refresh: bool = False) -> Optional[List[Dict[str, Any]]]:
    """Returns every row of `table` as a list of {column: value} dicts, or
    None if Sheets isn't configured/reachable — callers should fall back to
    their local JSON file in that case. Cached briefly so a page full of
    Streamlit widgets re-running doesn't re-hit the Sheets API every time."""
    if not _init():
        return None

    if not force_refresh:
        with _lock:
            cached = _cache.get(table)
        if cached and (time.time() - cached[0] < _CACHE_TTL_SECONDS):
            return [dict(r) for r in cached[1]]

    try:
        ws = _get_or_create_worksheet(table, headers)
        records = ws.get_all_records()
    except Exception as e:  # noqa: BLE001
        logger.warning("Sheets read failed for table %r: %s", table, e)
        return None

    with _lock:
        _cache[table] = (time.time(), records)
    return [dict(r) for r in records]


def append_row(table: str, headers: List[str], row: Dict[str, Any]) -> bool:
    """Appends one new row. Returns True on success, False if Sheets isn't
    configured/reachable (caller should fall back to its local JSON file)."""
    if not _init():
        return False
    try:
        ws = _get_or_create_worksheet(table, headers)
        ws.append_row([row.get(h, "") for h in headers])
        _invalidate(table)
        return True
    except Exception as e:  # noqa: BLE001
        logger.warning("Sheets append failed for table %r: %s", table, e)
        return False


def upsert_row(table: str, headers: List[str], match: Dict[str, Any], updates: Dict[str, Any]) -> bool:
    """Finds the first row where EVERY column in `match` equals its given
    value, and merges `updates` into it; appends a brand-new row
    (match + updates) if no row matched. Returns True on success, False if
    Sheets isn't configured/reachable."""
    if not _init():
        return False
    try:
        ws = _get_or_create_worksheet(table, headers)
        records = ws.get_all_records()
        row_index = None
        for i, r in enumerate(records):
            if all(str(r.get(k, "")) == str(v) for k, v in match.items()):
                row_index = i + 2  # +1 header row, +1 for 1-indexing
                break
        if row_index is None:
            new_row = dict(match)
            new_row.update(updates)
            ws.append_row([new_row.get(h, "") for h in headers])
        else:
            current = dict(records[row_index - 2])
            current.update(updates)
            ws.update(f"A{row_index}", [[current.get(h, "") for h in headers]])
        _invalidate(table)
        return True
    except Exception as e:  # noqa: BLE001
        logger.warning("Sheets upsert failed for table %r: %s", table, e)
        return False
