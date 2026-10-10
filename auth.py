"""
auth.py
--------
Lightweight multi-shop login + per-shop daily usage limiting.

Why this exists: when ONE Gemini API key is shared across many shops, a
single shop generating hundreds of times could burn through your whole
budget. This module gives every shop its own username/password and a daily
generation cap you control.

PERSISTENCE: shops you pre-configure yourself (via manage_shops.py, then
git-commit shops.json) are safe across redeploys because they're part of
your git repo. But shops that SELF-SIGN-UP through the app, and daily usage
counters, are written live while the app is running — if they only ever
land in a local JSON file, they're lost the next time you push new code
and Streamlit Cloud rebuilds the server from your repo. To fix that, this
module writes self-signups and usage to a Google Sheet (via sheets_db.py)
whenever one is configured — see README.md for setup. Until you set that
up (or if Sheets is briefly unreachable), everything transparently falls
back to the local JSON files exactly as before, so nothing breaks either
way.
"""

from __future__ import annotations

import json
import hashlib
import threading
from datetime import date
from pathlib import Path
from typing import Optional, Dict, Any

import sheets_db

BASE_DIR = Path(__file__).parent
SHOPS_FILE = BASE_DIR / "shops.json"
USAGE_FILE = BASE_DIR / "usage.json"

SHOPS_HEADERS = ["id", "name", "password_hash", "daily_limit", "admin", "created_at"]
USAGE_HEADERS = ["shop_id", "date", "count"]

GLOBAL_KEY = "__global__"  # pseudo shop-id used for organization-wide usage tracking
SIGNUP_DEFAULT_LIMIT = 0   # self-signed-up shops get ZERO free daily quota —
                           # they must buy credits before they can generate.
                           # This exists so people can't dodge paying by
                           # creating unlimited throwaway shop accounts.

_lock = threading.Lock()


# --------------------------------------------------------------------------
# Password hashing
# --------------------------------------------------------------------------
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# Shops config (who is allowed to log in)
# --------------------------------------------------------------------------
def _shop_row_to_dict(row: Dict[str, Any]) -> Dict[str, Any]:
    shop = {
        "id": row.get("id", ""),
        "name": row.get("name", ""),
        "password_hash": row.get("password_hash", ""),
        "daily_limit": int(row.get("daily_limit") or 0),
    }
    if str(row.get("admin", "")).strip().lower() in ("1", "true", "yes"):
        shop["admin"] = True
    return shop


def _load_shops() -> Dict[str, Any]:
    """Merges shops from three sources, each used for a different purpose:
    1. shops.json (the local/git-committed file) — shops you pre-configured
       yourself via manage_shops.py before deploying. Safe across redeploys
       because it's part of your git repo.
    2. Google Sheets "Shops" table (if configured) — shops that SELF-SIGNED
       UP through the app's Sign Up tab. These are written live while the
       app runs, so they need storage that survives a redeploy — see
       sheets_db.py. If Sheets isn't configured, self-signups still land in
       the SAME local shops.json file as a fallback (old behavior).
    3. st.secrets — persistent, set by the app owner; takes priority if the
       same id exists in more than one place.
    """
    shops: Dict[str, Any] = {}

    if SHOPS_FILE.exists():
        try:
            data = json.loads(SHOPS_FILE.read_text(encoding="utf-8"))
            for s in data.get("shops", []):
                shops[s["id"]] = s
        except Exception:
            pass

    sheet_rows = sheets_db.get_rows("Shops", SHOPS_HEADERS)
    if sheet_rows is not None:
        for row in sheet_rows:
            if row.get("id"):
                shops[row["id"]] = _shop_row_to_dict(row)

    try:
        import streamlit as st
        if "shops" in st.secrets:
            # secrets take priority if the same id exists in both places
            for s in st.secrets["shops"]:
                shops[s["id"]] = dict(s)
    except Exception:
        pass

    return shops


def verify_login(shop_id: str, password: str) -> Optional[Dict[str, Any]]:
    shops = _load_shops()
    shop = shops.get(shop_id.strip())
    if not shop:
        return None
    if shop.get("password_hash") != hash_password(password):
        return None
    return shop


def shop_exists(shop_id: str) -> bool:
    return shop_id.strip() in _load_shops()


def is_admin(shop: Dict[str, Any]) -> bool:
    return bool(shop.get("admin"))


def list_all_shops() -> list:
    """Returns all shop records (for the admin dashboard) — _load_shops()
    already merges shops.json + Google Sheets self-signups + st.secrets."""
    return list(_load_shops().values())


class SignupError(Exception):
    pass


def signup_shop(shop_id: str, name: str, password: str) -> Dict[str, Any]:
    """Self-service shop registration. New shops start on a small trial
    daily limit (SIGNUP_DEFAULT_LIMIT) — the app owner can raise it later
    via `python manage_shops.py add <id> <name> <same_password> <limit>`.

    Writes to Google Sheets when configured (sheets_db.py) so the signup
    survives the next redeploy; falls back to the local shops.json file
    (the old behavior) if Sheets isn't set up or is briefly unreachable —
    either way, signup can't modify st.secrets, which is read-only at
    runtime."""
    shop_id = shop_id.strip()
    name = name.strip()
    if not shop_id or not name or not password:
        raise SignupError("Shop ID, name and password are all required.")
    if len(password) < 6:
        raise SignupError("Password must be at least 6 characters.")
    if not shop_id.replace("_", "").replace("-", "").isalnum():
        raise SignupError("Shop ID can only contain letters, numbers, _ and -.")
    if shop_exists(shop_id):
        raise SignupError("That Shop ID is already taken — pick another one.")

    new_shop = {
        "id": shop_id,
        "name": name,
        "password_hash": hash_password(password),
        "daily_limit": SIGNUP_DEFAULT_LIMIT,
    }

    saved_to_sheets = sheets_db.append_row(
        "Shops", SHOPS_HEADERS,
        {**new_shop, "admin": "", "created_at": date.today().isoformat()},
    )
    if not saved_to_sheets:
        with _lock:
            if SHOPS_FILE.exists():
                try:
                    data = json.loads(SHOPS_FILE.read_text(encoding="utf-8"))
                except Exception:
                    data = {"shops": []}
            else:
                data = {"shops": []}
            data.setdefault("shops", []).append(new_shop)
            SHOPS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")

    return new_shop


# --------------------------------------------------------------------------
# Daily usage tracking
# --------------------------------------------------------------------------
def _load_usage() -> Dict[str, Any]:
    if USAGE_FILE.exists():
        try:
            return json.loads(USAGE_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_usage(data: Dict[str, Any]) -> None:
    try:
        USAGE_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception:
        pass  # best-effort — don't crash generation over a logging failure


def today_key() -> str:
    return date.today().isoformat()


def get_usage_today(shop_id: str) -> int:
    today = today_key()
    sheet_rows = sheets_db.get_rows("Usage", USAGE_HEADERS)
    if sheet_rows is not None:
        for r in sheet_rows:
            if r.get("shop_id") == shop_id and str(r.get("date")) == today:
                return int(r.get("count") or 0)
        return 0
    with _lock:
        data = _load_usage()
        return int(data.get(shop_id, {}).get(today, 0))


def increment_usage(shop_id: str) -> int:
    """Increments today's count for a shop and returns the new count."""
    today = today_key()

    sheet_rows = sheets_db.get_rows("Usage", USAGE_HEADERS)
    if sheet_rows is not None:
        current = 0
        for r in sheet_rows:
            if r.get("shop_id") == shop_id and str(r.get("date")) == today:
                current = int(r.get("count") or 0)
                break
        new_count = current + 1
        sheets_db.upsert_row(
            "Usage", USAGE_HEADERS,
            match={"shop_id": shop_id, "date": today},
            updates={"count": new_count},
        )
        return new_count

    with _lock:
        data = _load_usage()
        shop_usage = data.setdefault(shop_id, {})
        shop_usage[today] = int(shop_usage.get(today, 0)) + 1
        # Keep the file small — only retain the last 14 days per shop.
        if len(shop_usage) > 14:
            for old_day in sorted(shop_usage.keys())[:-14]:
                shop_usage.pop(old_day, None)
        _save_usage(data)
        return shop_usage[today]


def remaining_quota(shop_id: str, daily_limit: int) -> int:
    """daily_limit convention: a NEGATIVE number (e.g. -1) means unlimited.
    0 means a real zero free-per-day quota (used for self-signup shops that
    must buy credits before generating anything). Any positive number is a
    real daily cap."""
    if daily_limit < 0:
        return 10 ** 9  # negative = unlimited
    used = get_usage_today(shop_id)
    return max(0, daily_limit - used)


def usage_summary_today() -> Dict[str, int]:
    """Returns {shop_id: generations_today} for every shop that has usage
    logged today — used by the admin dashboard."""
    today = today_key()

    sheet_rows = sheets_db.get_rows("Usage", USAGE_HEADERS)
    if sheet_rows is not None:
        result: Dict[str, int] = {}
        for r in sheet_rows:
            sid = r.get("shop_id")
            if sid and sid != GLOBAL_KEY and str(r.get("date")) == today:
                result[sid] = int(r.get("count") or 0)
        return result

    with _lock:
        data = _load_usage()
        return {sid: int(days.get(today, 0)) for sid, days in data.items() if sid != GLOBAL_KEY}
