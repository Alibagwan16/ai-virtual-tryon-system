"""
credits.py
----------
Pay-as-you-go credit top-ups for shops, on top of the existing free daily
limit in auth.py.

How it fits together:
- Each shop still has its normal `daily_limit` (free generations/day, reset
  every day) — unchanged, still enforced by auth.py.
- On top of that, each shop has a `credit_balance` (purchased credits that
  do NOT expire/reset). Once a shop's free daily quota runs out, they can
  keep generating by spending 1 credit per generation instead of being
  blocked — see spend_credit_if_needed() in app.py's generation loop.
- A shop buys credits from inside the app: picks how many credits they
  want, sees the auto-calculated price (credits x PRICE_PER_CREDIT), and
  gets a UPI QR code (generated locally — no payment gateway account
  needed) pre-filled with your UPI ID, the exact amount, and a reference
  note so you can match the payment to the shop.
- IMPORTANT LIMITATION: a plain UPI QR code has no way to tell this app
  that the payment actually succeeded — UPI doesn't call back a webhook
  the way a payment gateway (Razorpay/Cashfree/PhonePe Business etc.) does.
  So the shop submits their UPI transaction ID after paying, which creates
  a PENDING request here. You (the app owner / admin) check your bank/UPI
  app, confirm the money actually landed, and approve the request from the
  Admin Dashboard — only then are credits added to their balance. This is
  a manual-approval flow, not instant auto-credit. If you want instant
  auto-credit later, that requires integrating a real payment gateway with
  webhook support (a bigger change — ask if you want that built).

Storage: a Google Sheet (via sheets_db.py) when one is configured — see
README.md for setup — so balances and pending requests survive redeploys,
since they're written live while the app runs and would otherwise be lost
the next time you push new code (Streamlit Cloud rebuilds the server fresh
from your git repo on every deploy; anything only in a local file on the
OLD server is gone). Falls back to two local JSON files, next to
shops.json/usage.json — credits.json (shop_id -> balance) and
payment_requests.json (list of requests, pending/approved/rejected) — if
Sheets isn't configured yet or is briefly unreachable.
"""

from __future__ import annotations

import io
import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

import sheets_db

BASE_DIR = Path(__file__).parent
CREDITS_FILE = BASE_DIR / "credits.json"
REQUESTS_FILE = BASE_DIR / "payment_requests.json"

CREDITS_HEADERS = ["shop_id", "balance"]
REQUESTS_HEADERS = [
    "id", "shop_id", "shop_name", "credits_requested", "amount",
    "utr_reference", "status", "created_at", "decided_at",
]

_lock = threading.Lock()

# --------------------------------------------------------------------------
# Pricing / payee config — set these in .env (see .env.example)
# --------------------------------------------------------------------------
def _get_price_per_credit() -> float:
    """Reads PRICE_PER_CREDIT from .env as a plain number (e.g. 15, not
    "15rs" or "₹15"). Falls back to 15 and logs a warning instead of
    crashing the whole app if someone puts a non-numeric value in .env."""
    raw = os.getenv("PRICE_PER_CREDIT", "15").strip()
    try:
        return float(raw)
    except ValueError:
        import logging
        logging.getLogger(__name__).warning(
            "PRICE_PER_CREDIT=%r in .env isn't a plain number (e.g. 15) — "
            "falling back to 15. Remove any currency symbol or suffix.",
            raw,
        )
        return 15.0


PRICE_PER_CREDIT = _get_price_per_credit()  # ₹ per credit
OWNER_UPI_ID = os.getenv("OWNER_UPI_ID", "").strip()            # e.g. "yourname@okhdfcbank"
OWNER_UPI_NAME = os.getenv("OWNER_UPI_NAME", "AI Try-On Studio").strip()


# --------------------------------------------------------------------------
# Credit balances
# --------------------------------------------------------------------------
def _load_credits() -> Dict[str, float]:
    if CREDITS_FILE.exists():
        try:
            return json.loads(CREDITS_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_credits(data: Dict[str, float]) -> None:
    try:
        CREDITS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception:
        pass  # best-effort — don't crash generation over a logging failure


def get_credit_balance(shop_id: str) -> int:
    sheet_rows = sheets_db.get_rows("Credits", CREDITS_HEADERS)
    if sheet_rows is not None:
        for r in sheet_rows:
            if r.get("shop_id") == shop_id:
                return int(r.get("balance") or 0)
        return 0
    with _lock:
        data = _load_credits()
        return int(data.get(shop_id, 0))


def add_credits(shop_id: str, amount: int) -> int:
    """Adds `amount` credits to a shop's balance. Returns the new balance."""
    with _lock:
        sheet_rows = sheets_db.get_rows("Credits", CREDITS_HEADERS)
        if sheet_rows is not None:
            current = 0
            for r in sheet_rows:
                if r.get("shop_id") == shop_id:
                    current = int(r.get("balance") or 0)
                    break
            new_balance = current + int(amount)
            sheets_db.upsert_row(
                "Credits", CREDITS_HEADERS,
                match={"shop_id": shop_id}, updates={"balance": new_balance},
            )
            return new_balance

        data = _load_credits()
        data[shop_id] = int(data.get(shop_id, 0)) + int(amount)
        _save_credits(data)
        return data[shop_id]


def spend_credit(shop_id: str, amount: int = 1) -> bool:
    """Tries to spend `amount` credits. Returns True and deducts if the
    shop has enough; returns False (no change) if they don't."""
    with _lock:
        sheet_rows = sheets_db.get_rows("Credits", CREDITS_HEADERS)
        if sheet_rows is not None:
            current = 0
            for r in sheet_rows:
                if r.get("shop_id") == shop_id:
                    current = int(r.get("balance") or 0)
                    break
            if current < amount:
                return False
            new_balance = current - amount
            sheets_db.upsert_row(
                "Credits", CREDITS_HEADERS,
                match={"shop_id": shop_id}, updates={"balance": new_balance},
            )
            return True

        data = _load_credits()
        current = int(data.get(shop_id, 0))
        if current < amount:
            return False
        data[shop_id] = current - amount
        _save_credits(data)
        return True


# --------------------------------------------------------------------------
# UPI QR code generation (local — no payment gateway account needed)
# --------------------------------------------------------------------------
def build_upi_link(amount: float, note: str) -> str:
    """Builds a standard UPI deep-link string. Scanning/opening it in any
    UPI app (GPay, PhonePe, Paytm, BHIM...) pre-fills your UPI ID, the
    amount, and this note — the shop just has to confirm and pay."""
    from urllib.parse import quote

    params = (
        f"pa={quote(OWNER_UPI_ID)}"
        f"&pn={quote(OWNER_UPI_NAME)}"
        f"&am={amount:.2f}"
        f"&cu=INR"
        f"&tn={quote(note)}"
    )
    return f"upi://pay?{params}"


def generate_upi_qr(amount: float, note: str) -> Image.Image:
    """Returns a PIL Image of a QR code encoding the UPI payment link."""
    import qrcode

    link = build_upi_link(amount, note)
    qr = qrcode.QRCode(border=2, box_size=8)
    qr.add_data(link)
    qr.make(fit=True)
    return qr.make_image(fill_color="black", back_color="white").convert("RGB")


def is_configured() -> bool:
    """True once the app owner has set OWNER_UPI_ID in .env — until then,
    the buy-credits UI stays hidden instead of showing a broken QR."""
    return bool(OWNER_UPI_ID)


# --------------------------------------------------------------------------
# Payment requests (pending manual approval)
# --------------------------------------------------------------------------
def _load_requests() -> List[Dict[str, Any]]:
    if REQUESTS_FILE.exists():
        try:
            return json.loads(REQUESTS_FILE.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def _save_requests(data: List[Dict[str, Any]]) -> None:
    try:
        REQUESTS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception:
        pass


def create_payment_request(shop_id: str, shop_name: str, credits_requested: int,
                            amount: float, utr_reference: str) -> str:
    """Shop submits this after paying via the QR code. Status starts as
    'pending' until an admin approves/rejects it. Returns the request id."""
    req_id = uuid.uuid4().hex[:10]
    new_row = {
        "id": req_id,
        "shop_id": shop_id,
        "shop_name": shop_name,
        "credits_requested": int(credits_requested),
        "amount": round(float(amount), 2),
        "utr_reference": utr_reference.strip(),
        "status": "pending",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "decided_at": "",
    }

    saved = sheets_db.append_row("PaymentRequests", REQUESTS_HEADERS, new_row)
    if not saved:
        with _lock:
            data = _load_requests()
            data.append({**new_row, "decided_at": None})
            _save_requests(data)

    return req_id


def list_pending_requests() -> List[Dict[str, Any]]:
    sheet_rows = sheets_db.get_rows("PaymentRequests", REQUESTS_HEADERS)
    if sheet_rows is not None:
        return [r for r in sheet_rows if r.get("status") == "pending"]
    with _lock:
        return [r for r in _load_requests() if r.get("status") == "pending"]


def list_requests_for_shop(shop_id: str, limit: int = 5) -> List[Dict[str, Any]]:
    sheet_rows = sheets_db.get_rows("PaymentRequests", REQUESTS_HEADERS)
    if sheet_rows is not None:
        data = [r for r in sheet_rows if r.get("shop_id") == shop_id]
        return list(reversed(data))[:limit]
    with _lock:
        data = [r for r in _load_requests() if r.get("shop_id") == shop_id]
    return list(reversed(data))[:limit]


def approve_request(request_id: str) -> Optional[Dict[str, Any]]:
    """Marks the request approved AND credits the shop's balance. Returns
    the updated request, or None if it wasn't found / already decided."""
    sheet_rows = sheets_db.get_rows("PaymentRequests", REQUESTS_HEADERS)
    if sheet_rows is not None:
        target = next((r for r in sheet_rows if r.get("id") == request_id and r.get("status") == "pending"), None)
        if target is None:
            return None
        updates = {"status": "approved", "decided_at": time.strftime("%Y-%m-%d %H:%M:%S")}
        sheets_db.upsert_row("PaymentRequests", REQUESTS_HEADERS, match={"id": request_id}, updates=updates)
        target = {**target, **updates}
    else:
        with _lock:
            data = _load_requests()
            target = next((r for r in data if r["id"] == request_id and r["status"] == "pending"), None)
            if target is None:
                return None
            target["status"] = "approved"
            target["decided_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            _save_requests(data)

    add_credits(target["shop_id"], int(target["credits_requested"]))
    return target


def reject_request(request_id: str) -> Optional[Dict[str, Any]]:
    sheet_rows = sheets_db.get_rows("PaymentRequests", REQUESTS_HEADERS)
    if sheet_rows is not None:
        target = next((r for r in sheet_rows if r.get("id") == request_id and r.get("status") == "pending"), None)
        if target is None:
            return None
        updates = {"status": "rejected", "decided_at": time.strftime("%Y-%m-%d %H:%M:%S")}
        sheets_db.upsert_row("PaymentRequests", REQUESTS_HEADERS, match={"id": request_id}, updates=updates)
        return {**target, **updates}

    with _lock:
        data = _load_requests()
        target = next((r for r in data if r["id"] == request_id and r["status"] == "pending"), None)
        if target is None:
            return None
        target["status"] = "rejected"
        target["decided_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        _save_requests(data)
        return target
