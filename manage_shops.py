"""
manage_shops.py
-----------------
Simple command-line tool to manage which shops can log into the app.

Usage:
    python manage_shops.py add <shop_id> <display_name> <password> [daily_limit]
    python manage_shops.py remove <shop_id>
    python manage_shops.py list
    python manage_shops.py admin <shop_id>       # make a shop an admin (sees usage dashboard)
    python manage_shops.py unadmin <shop_id>     # remove admin rights

Examples:
    python manage_shops.py add sharma_cloth "Sharma Cloth House" MyPass123 20
    python manage_shops.py add gupta_textiles "Gupta Textiles" Secret456 15
    python manage_shops.py admin sharma_cloth
    python manage_shops.py list
    python manage_shops.py remove gupta_textiles

This writes to shops.json (next to app.py), which auth.py reads at login
time. daily_limit is the max number of try-on generations that shop can run
per day (default 20). Use -1 for unlimited, or 0 for a real zero free-daily
quota (shop must rely entirely on purchased credits — this is what
self-signup accounts get automatically, see auth.SIGNUP_DEFAULT_LIMIT).

Note: shops that sign themselves up via the app's "Sign Up" tab also land
in this same shops.json file — `list` and `admin` work on them too.
"""

import sys
import json
from pathlib import Path

from auth import hash_password  # reuse the same hashing as the app

SHOPS_FILE = Path(__file__).parent / "shops.json"


def load() -> dict:
    if SHOPS_FILE.exists():
        return json.loads(SHOPS_FILE.read_text(encoding="utf-8"))
    return {"shops": []}


def save(data: dict) -> None:
    SHOPS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Saved to {SHOPS_FILE}")


def cmd_add(shop_id: str, display_name: str, password: str, daily_limit: int = 20):
    data = load()
    shops = data["shops"]
    existing = next((s for s in shops if s["id"] == shop_id), None)
    is_admin = bool(existing.get("admin")) if existing else False  # preserve admin flag on update
    shops[:] = [s for s in shops if s["id"] != shop_id]  # replace if exists
    entry = {
        "id": shop_id,
        "name": display_name,
        "password_hash": hash_password(password),
        "daily_limit": daily_limit,
    }
    if is_admin:
        entry["admin"] = True
    shops.append(entry)
    save(data)
    print(f"✅ Shop '{shop_id}' added/updated — daily limit: {'unlimited' if daily_limit < 0 else daily_limit}")


def cmd_remove(shop_id: str):
    data = load()
    before = len(data["shops"])
    data["shops"] = [s for s in data["shops"] if s["id"] != shop_id]
    save(data)
    after = len(data["shops"])
    if before == after:
        print(f"⚠️  No shop found with id '{shop_id}'")
    else:
        print(f"✅ Shop '{shop_id}' removed")


def cmd_admin(shop_id: str, make_admin: bool = True):
    data = load()
    found = False
    for s in data["shops"]:
        if s["id"] == shop_id:
            if make_admin:
                s["admin"] = True
            else:
                s.pop("admin", None)
            found = True
    if not found:
        print(f"⚠️  No shop found with id '{shop_id}' — add it first with 'add'.")
        return
    save(data)
    print(f"✅ Shop '{shop_id}' {'is now an admin' if make_admin else 'admin rights removed'}.")


def cmd_list():
    data = load()
    if not data["shops"]:
        print("No shops configured yet.")
        return
    print(f"{'ID':<20} {'Name':<28} {'Daily limit':<12} {'Admin':<6}")
    print("-" * 70)
    for s in data["shops"]:
        limit = s.get("daily_limit", 20)
        limit_display = "unlimited" if limit < 0 else limit
        admin_flag = "yes" if s.get("admin") else ""
        print(f"{s['id']:<20} {s['name']:<28} {limit_display!s:<12} {admin_flag:<6}")


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return

    cmd = args[0]
    if cmd == "add":
        if len(args) < 4:
            print("Usage: python manage_shops.py add <shop_id> <display_name> <password> [daily_limit]")
            return
        shop_id, display_name, password = args[1], args[2], args[3]
        daily_limit = int(args[4]) if len(args) > 4 else 20
        cmd_add(shop_id, display_name, password, daily_limit)
    elif cmd == "remove":
        if len(args) < 2:
            print("Usage: python manage_shops.py remove <shop_id>")
            return
        cmd_remove(args[1])
    elif cmd == "admin":
        if len(args) < 2:
            print("Usage: python manage_shops.py admin <shop_id>")
            return
        cmd_admin(args[1], make_admin=True)
    elif cmd == "unadmin":
        if len(args) < 2:
            print("Usage: python manage_shops.py unadmin <shop_id>")
            return
        cmd_admin(args[1], make_admin=False)
    elif cmd == "list":
        cmd_list()
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
