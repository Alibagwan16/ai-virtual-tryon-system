"""
migrate_to_sheets.py
---------------------
Run this ONCE, locally, after you've set up Google Sheets storage (see
README.md), to copy whatever is in your existing local JSON files
(shops.json, usage.json, credits.json, payment_requests.json) into the new
Google Sheet — so none of that history is lost when you switch over.

Usage:
    python migrate_to_sheets.py

Safe to re-run: it upserts (updates-or-inserts) rather than blindly
appending, so running it twice won't create duplicate rows.
"""

from __future__ import annotations

import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # so GOOGLE_SHEET_ID / GOOGLE_SERVICE_ACCOUNT_JSON are picked up

import sheets_db  # noqa: E402  (import after load_dotenv on purpose)

BASE_DIR = Path(__file__).parent
SHOPS_FILE = BASE_DIR / "shops.json"
USAGE_FILE = BASE_DIR / "usage.json"
CREDITS_FILE = BASE_DIR / "credits.json"
REQUESTS_FILE = BASE_DIR / "payment_requests.json"

SHOPS_HEADERS = ["id", "name", "password_hash", "daily_limit", "admin", "created_at"]
USAGE_HEADERS = ["shop_id", "date", "count"]
CREDITS_HEADERS = ["shop_id", "balance"]
REQUESTS_HEADERS = [
    "id", "shop_id", "shop_name", "credits_requested", "amount",
    "utr_reference", "status", "created_at", "decided_at",
]


def _load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        print(f"  ⚠️  Couldn't read {path.name}: {e} — skipping it.")
        return default


def migrate_shops() -> int:
    data = _load_json(SHOPS_FILE, {"shops": []})
    shops = data.get("shops", [])
    count = 0
    for s in shops:
        row = {
            "id": s.get("id", ""),
            "name": s.get("name", ""),
            "password_hash": s.get("password_hash", ""),
            "daily_limit": s.get("daily_limit", 0),
            "admin": "true" if s.get("admin") else "",
            "created_at": s.get("created_at", ""),
        }
        if not row["id"]:
            continue
        ok = sheets_db.upsert_row("Shops", SHOPS_HEADERS, match={"id": row["id"]}, updates=row)
        if ok:
            count += 1
    return count


def migrate_usage() -> int:
    data = _load_json(USAGE_FILE, {})
    count = 0
    for shop_id, days in data.items():
        if not isinstance(days, dict):
            continue
        for day, cnt in days.items():
            ok = sheets_db.upsert_row(
                "Usage", USAGE_HEADERS,
                match={"shop_id": shop_id, "date": day},
                updates={"count": cnt},
            )
            if ok:
                count += 1
    return count


def migrate_credits() -> int:
    data = _load_json(CREDITS_FILE, {})
    count = 0
    for shop_id, balance in data.items():
        ok = sheets_db.upsert_row(
            "Credits", CREDITS_HEADERS,
            match={"shop_id": shop_id}, updates={"balance": balance},
        )
        if ok:
            count += 1
    return count


def migrate_requests() -> int:
    data = _load_json(REQUESTS_FILE, [])
    count = 0
    for r in data:
        row = {
            "id": r.get("id", ""),
            "shop_id": r.get("shop_id", ""),
            "shop_name": r.get("shop_name", ""),
            "credits_requested": r.get("credits_requested", 0),
            "amount": r.get("amount", 0),
            "utr_reference": r.get("utr_reference", ""),
            "status": r.get("status", "pending"),
            "created_at": r.get("created_at", ""),
            "decided_at": r.get("decided_at") or "",
        }
        if not row["id"]:
            continue
        ok = sheets_db.upsert_row("PaymentRequests", REQUESTS_HEADERS, match={"id": row["id"]}, updates=row)
        if ok:
            count += 1
    return count


def main():
    print("Checking Google Sheets configuration...")
    if not sheets_db.is_configured():
        print(
            "❌ Google Sheets isn't configured yet (GOOGLE_SHEET_ID and/or the "
            "service account key is missing). Set those up first — see "
            "README.md — then run this script again."
        )
        return

    print("Connecting...")
    # Trigger a real connection attempt now so we can show a clear error.
    probe = sheets_db.get_rows("Shops", SHOPS_HEADERS)
    if probe is None:
        print(f"❌ Couldn't connect to the Sheet: {sheets_db.init_error()}")
        print("   Double-check the Sheet ID and that you shared it with the")
        print("   service account's email (Editor access).")
        return
    print("✅ Connected.\n")

    print("Migrating shops.json -> 'Shops' tab...")
    n = migrate_shops()
    print(f"  {n} shop(s) migrated.\n")

    print("Migrating usage.json -> 'Usage' tab...")
    n = migrate_usage()
    print(f"  {n} usage row(s) migrated.\n")

    print("Migrating credits.json -> 'Credits' tab...")
    n = migrate_credits()
    print(f"  {n} credit balance(s) migrated.\n")

    print("Migrating payment_requests.json -> 'PaymentRequests' tab...")
    n = migrate_requests()
    print(f"  {n} payment request(s) migrated.\n")

    print("✅ Done! Open your Google Sheet to double-check everything looks right.")
    print("   From now on, the live app will read/write this Sheet automatically.")


if __name__ == "__main__":
    main()
