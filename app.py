"""
AI Virtual Try-On Studio
------------------------
A Streamlit app that lets a user upload their own photo + a garment/fabric
photo and uses Google Gemini's image model (Nano Banana / gemini-2.5-flash-image)
to generate a realistic image of the person wearing that garment.

API key & model are configured on the BACKEND ONLY (via .env / st.secrets) —
nothing sensitive is shown or editable in the UI.

Run:
    streamlit run app.py
"""

import os
import io
import time
from datetime import datetime

import streamlit as st
from PIL import Image, ImageOps
from dotenv import load_dotenv

# Load .env BEFORE importing our own modules — several of them (credits,
# rate_limiter) read env vars like OWNER_UPI_ID / MAX_CONCURRENT_GENERATIONS
# at import time, so .env must already be loaded or those reads come back
# empty even when the file is set up correctly.
load_dotenv()

from tryon_engine import (
    generate_tryon,
    generate_full_outfit_tryon,
    GeminiTryOnError,
)
import concurrent.futures
import auth
import rate_limiter
import credits

# --------------------------------------------------------------------------
# Backend-only configuration (NEVER shown in the UI)
# --------------------------------------------------------------------------
def _get_api_key() -> str:
    """Reads the Gemini API key from Streamlit secrets first, then env vars.
    This is intentionally backend-only — never exposed as a UI text field."""
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return str(st.secrets["GEMINI_API_KEY"]).strip()
    except Exception:
        pass
    return os.getenv("GEMINI_API_KEY", "").strip()


API_KEY = _get_api_key()
SERVICE_CONFIGURED = bool(API_KEY)  # true if the Gemini API key is set
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-image").strip()
DEBUG_MODE = os.getenv("TRYON_DEBUG", "false").strip().lower() in ("1", "true", "yes")


def _generate_with_key_rotation(fn, **kwargs):
    """Calls fn(api_key=..., **kwargs) using the single configured Gemini
    API key."""
    return fn(api_key=API_KEY, **kwargs)

# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Virtual Try-On Studio",
    page_icon="🪄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# PWA support — lets people "Add to Home Screen" on Android/iOS so the app
# opens full-screen with its own icon, like a native app, with zero extra
# backend changes. Requires enableStaticServing = true in
# .streamlit/config.toml (already set) and the files under ./static/
# (manifest.json, service-worker.js, icon-*.png).
# --------------------------------------------------------------------------
st.markdown(
    """
    <link rel="manifest" href="./app/static/manifest.json">
    <meta name="theme-color" content="#22d3a6">
    <link rel="apple-touch-icon" href="./app/static/icon-192.png">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
    <meta name="apple-mobile-web-app-title" content="Try-On Studio">
    <script>
    if ('serviceWorker' in navigator) {
        navigator.serviceWorker.register('./app/static/service-worker.js')
            .catch(function (err) { console.warn('SW registration failed:', err); });
    }
    </script>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
}

:root {
    --tryon-bg: #0a0e14;
    --tryon-surface: #10151f;
    --tryon-border: rgba(255,255,255,0.08);
    --tryon-accent: #22d3a6;
    --tryon-accent-2: #2dd4bf;
    --tryon-text: #e8edf3;
    --tryon-text-dim: #96a1b3;
}

.stApp {
    background: var(--tryon-bg);
    background-image:
        linear-gradient(180deg, #0d1320 0%, #0a0e14 100%),
        radial-gradient(circle at 100% 0%, rgba(34,211,166,0.08) 0%, transparent 35%);
}

/* Hide default footer / menu clutter */
#MainMenu, footer {visibility: hidden;}

.hero {
    padding: 1.6rem 1.9rem;
    border-radius: 16px;
    background: var(--tryon-surface);
    border: 1px solid var(--tryon-border);
    border-left: 3px solid var(--tryon-accent);
    box-shadow: 0 4px 20px rgba(0,0,0,0.35);
    margin-bottom: 1.5rem;
}
.hero h1 {
    font-weight: 800;
    font-size: 1.9rem;
    letter-spacing: -0.02em;
    margin-bottom: 0.3rem;
    color: var(--tryon-text);
}
.hero h1::before {
    content: "";
}
.hero p {
    color: var(--tryon-text-dim);
    font-size: 0.98rem;
    margin: 0;
    line-height: 1.5;
}

.card {
    background: var(--tryon-surface);
    border: 1px solid var(--tryon-border);
    border-radius: 14px;
    padding: 1.2rem 1.2rem 1.4rem 1.2rem;
    box-shadow: 0 2px 10px rgba(0,0,0,0.25);
    margin-bottom: 1.1rem;
    transition: border-color 0.15s ease;
}
.card:hover {
    border-color: rgba(34,211,166,0.35);
}
.card h3 {
    margin-top: 0;
    color: var(--tryon-text);
    font-weight: 700;
    font-size: 1.02rem;
    letter-spacing: -0.005em;
}
.step-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 24px; height: 24px;
    border-radius: 7px;
    background: rgba(34,211,166,0.14);
    border: 1px solid rgba(34,211,166,0.4);
    color: var(--tryon-accent);
    font-weight: 700;
    font-size: 0.8rem;
    margin-right: 8px;
}

div.stButton > button, div.stDownloadButton > button {
    background: var(--tryon-accent);
    color: #06251d;
    border: none;
    border-radius: 10px;
    padding: 0.62rem 1.2rem;
    font-weight: 700;
    font-size: 0.94rem;
    width: 100%;
    transition: transform 0.12s ease, box-shadow 0.12s ease, filter 0.12s ease;
    box-shadow: 0 2px 12px rgba(34,211,166,0.25);
}
div.stButton > button:hover, div.stDownloadButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(34,211,166,0.35);
    filter: brightness(1.05);
    color: #06251d;
}
div.stButton > button:active, div.stDownloadButton > button:active {
    transform: translateY(0);
}
div.stButton > button:disabled {
    opacity: 0.35;
    box-shadow: none;
    transform: none;
}
/* Secondary/unselected toggle buttons — quiet outline so the selected
   (primary/accent) one visibly stands out. */
div.stButton > button[kind="secondary"] {
    background: transparent;
    color: var(--tryon-text-dim);
    border: 1px solid var(--tryon-border);
    box-shadow: none;
}
div.stButton > button[kind="secondary"]:hover {
    background: rgba(255,255,255,0.04);
    border-color: rgba(34,211,166,0.4);
    color: var(--tryon-text);
    box-shadow: none;
}

[data-testid="stSidebar"] {
    background: #0c1119;
    border-right: 1px solid var(--tryon-border);
}

.result-frame {
    border-radius: 14px;
    overflow: hidden;
    border: 1px solid var(--tryon-border);
    box-shadow: 0 8px 30px rgba(0,0,0,0.45);
}

.badge-pill {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 6px;
    background: rgba(34,211,166,0.1);
    border: 1px solid rgba(34,211,166,0.3);
    color: var(--tryon-accent);
    font-size: 0.76rem;
    font-weight: 600;
    margin-right: 6px;
}

.status-pill-ok {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 6px;
    background: rgba(34,211,166,0.1);
    border: 1px solid rgba(34,211,166,0.3);
    color: var(--tryon-accent);
    font-size: 0.76rem;
    font-weight: 600;
}
.status-pill-bad {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 6px;
    background: rgba(255,90,90,0.1);
    border: 1px solid rgba(255,90,90,0.32);
    color: #ff9d9d;
    font-size: 0.76rem;
    font-weight: 600;
}

.history-thumb {
    border-radius: 10px;
    border: 1px solid var(--tryon-border);
}

.swatch-row {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin: 0.4rem 0 0.2rem 0;
}
.swatch {
    width: 30px;
    height: 30px;
    margin: 0 auto;
    border-radius: 8px;
    border: 2px solid rgba(255,255,255,0.15);
    cursor: default;
    box-shadow: 0 2px 6px rgba(0,0,0,0.35);
    transition: transform 0.12s ease;
}
.swatch-selected {
    border: 2px solid var(--tryon-accent);
    box-shadow: 0 0 0 3px rgba(34,211,166,0.25);
    transform: scale(1.06);
}
.swatch-original {
    background: conic-gradient(from 0deg, #22d3a6, #2dd4bf, #7EC8E3, #F4D35E, #ff5fb0, #22d3a6);
    border-radius: 50%;
}
.swatch-label {
    text-align: center;
    font-size: 0.65rem;
    color: var(--tryon-text-dim);
    margin-top: 3px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

/* st.camera_input's own capture button */
[data-testid="stCameraInput"] button {
    background: var(--tryon-accent) !important;
    color: #06251d !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
}

/* Text areas, selects and the file-uploader dropzone — match the same
   crisp, flat-surface language as the cards instead of Streamlit's default. */
.stTextArea textarea, .stTextInput input {
    background: rgba(255,255,255,0.035) !important;
    border: 1px solid var(--tryon-border) !important;
    border-radius: 10px !important;
    color: var(--tryon-text) !important;
}
.stTextArea textarea:focus, .stTextInput input:focus {
    border-color: rgba(34,211,166,0.55) !important;
    box-shadow: 0 0 0 3px rgba(34,211,166,0.15) !important;
}
[data-testid="stFileUploaderDropzone"] {
    background: rgba(255,255,255,0.025) !important;
    border: 1.5px dashed rgba(34,211,166,0.35) !important;
    border-radius: 12px !important;
}
[data-baseweb="select"] > div {
    background: rgba(255,255,255,0.035) !important;
    border-radius: 10px !important;
    border-color: var(--tryon-border) !important;
}

/* Collapsible sections (Choose Mode & Type / Garment Type / Colors) — match
   the same card language so they don't look like a bare default widget. */
[data-testid="stExpander"] {
    background: var(--tryon-surface);
    border: 1px solid var(--tryon-border);
    border-radius: 14px;
    margin-bottom: 1.1rem;
    box-shadow: 0 2px 10px rgba(0,0,0,0.25);
    overflow: hidden;
}
[data-testid="stExpander"] summary {
    font-weight: 700;
    color: var(--tryon-text);
    padding: 0.9rem 1.1rem !important;
}
[data-testid="stExpander"] summary:hover {
    color: var(--tryon-accent);
}
[data-testid="stExpander"] [data-testid="stExpanderDetails"] {
    padding: 0.2rem 1.1rem 1.2rem 1.1rem;
}

@keyframes tryon-fade-in {
    from { opacity: 0; transform: translateY(6px); }
    to { opacity: 1; transform: translateY(0); }
}
.card { animation: tryon-fade-in 0.3s ease both; }
.result-frame { animation: tryon-fade-in 0.4s ease both; }

/* Top-right "☰" menu popover trigger — a quiet square icon button instead
   of a wide accent CTA, so it reads as a menu, not another action button. */
[data-testid="stPopover"] > div > button {
    background: var(--tryon-surface) !important;
    color: var(--tryon-text) !important;
    border: 1px solid var(--tryon-border) !important;
    box-shadow: none !important;
    font-size: 1.1rem !important;
    aspect-ratio: 1 / 1;
    padding: 0.5rem !important;
}
[data-testid="stPopover"] > div > button:hover {
    border-color: rgba(34,211,166,0.5) !important;
    color: var(--tryon-accent) !important;
    transform: none !important;
}

/* Tighter gaps between columns/cards on narrow (phone) screens so the UI
   doesn't feel sparse or cause horizontal squeeze/overflow. */
@media (max-width: 640px) {
    .hero { padding: 1.15rem 1.3rem; border-radius: 14px; }
    .hero h1 { font-size: 1.4rem; }
    .hero p { font-size: 0.88rem; }
    .card { padding: 1rem 0.9rem 1.2rem 0.9rem; border-radius: 12px; }
    div.stButton > button, div.stDownloadButton > button { font-size: 0.88rem; padding: 0.55rem 0.8rem; }
    .swatch { width: 26px; height: 26px; }
    /* On narrow screens the hero/menu columns stack instead of sitting
       side-by-side, so pin the "☰" menu button to the right edge instead
       of letting it sit left-aligned at full column width. */
    [data-testid="stPopover"] { display: flex; justify-content: flex-end; margin-top: -0.5rem; }
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Shop login gate — multi-shop access with per-shop daily usage limits
# --------------------------------------------------------------------------
if "shop" not in st.session_state:
    st.session_state.shop = None

if st.session_state.shop is None:
    st.markdown(
        """
        <div class="hero" style="max-width: 420px; margin: 3rem auto 0 auto; text-align:center;">
            <h1 style="font-size:1.6rem;">🪄 AI Virtual Try-On Studio</h1>
            <p>Shop login</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    login_col = st.columns([1, 1.3, 1])[1]
    with login_col:
        login_tab, signup_tab = st.tabs(["🔑 Login", "🆕 Sign Up"])

        with login_tab:
            with st.form("login_form"):
                shop_id_input = st.text_input("Shop ID")
                password_input = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Login", use_container_width=True, type="primary")
            if submitted:
                shop = auth.verify_login(shop_id_input, password_input)
                if shop:
                    st.session_state.shop = shop
                    st.rerun()
                else:
                    st.error("Invalid Shop ID or password. Contact the app owner if you don't have access.")

        with signup_tab:
            st.caption(
                "New accounts start with **0 free generations** — buy credits "
                "right after signing up to start generating."
            )
            with st.form("signup_form"):
                su_shop_id = st.text_input("Choose a Shop ID", help="Letters, numbers, _ and - only. This is what you'll log in with.")
                su_name = st.text_input("Shop / Business Name")
                su_password = st.text_input("Choose a Password", type="password")
                su_password2 = st.text_input("Confirm Password", type="password")
                su_submitted = st.form_submit_button("Create Account", use_container_width=True, type="primary")
            if su_submitted:
                if su_password != su_password2:
                    st.error("Passwords don't match.")
                else:
                    try:
                        new_shop = auth.signup_shop(su_shop_id, su_name, su_password)
                        st.session_state.shop = new_shop
                        st.success("Account created! Buy credits from the sidebar to start generating.")
                        st.rerun()
                    except auth.SignupError as e:
                        st.error(str(e))
    st.stop()

CURRENT_SHOP = st.session_state.shop
SHOP_ID = CURRENT_SHOP["id"]
SHOP_NAME = CURRENT_SHOP.get("name", SHOP_ID)
DAILY_LIMIT = int(CURRENT_SHOP.get("daily_limit", 20) or 0)
UNLIMITED_DAILY = DAILY_LIMIT < 0  # negative = unlimited (see auth.remaining_quota)
IS_ADMIN = auth.is_admin(CURRENT_SHOP)

# --------------------------------------------------------------------------
# Admin dashboard — see usage across every shop (only for shops marked
# "admin": true in shops.json, set via: python manage_shops.py admin <id>)
# --------------------------------------------------------------------------
if IS_ADMIN:
    st.markdown(
        """
        <div class="hero">
            <h1>📊 Admin Dashboard</h1>
            <p>Usage across every shop, today.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("🚪 Logout", key="admin_logout"):
        st.session_state.shop = None
        st.rerun()

    all_shops = auth.list_all_shops()
    usage_today = auth.usage_summary_today()
    global_used = auth.get_usage_today(auth.GLOBAL_KEY)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total shops", len(all_shops))
    c2.metric("Paid-engine generations today (org-wide)", global_used)
    c3.metric(
        "Org daily cap",
        rate_limiter.MAX_DAILY_GENERATIONS if rate_limiter.MAX_DAILY_GENERATIONS > 0 else "Unlimited",
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    for shop in sorted(all_shops, key=lambda s: -usage_today.get(s["id"], 0)):
        limit = int(shop.get("daily_limit", 0) or 0)
        used = usage_today.get(shop["id"], 0)
        cols = st.columns([2, 2, 1, 1])
        cols[0].markdown(f"**{shop.get('name', shop['id'])}**")
        cols[1].caption(shop["id"])
        cols[2].caption(f"{used} used")
        cols[3].caption("∞" if limit < 0 else f"limit {limit}")
        if limit >= 0:
            st.progress(min(1.0, used / limit) if limit > 0 else 1.0)
    st.markdown("</div>", unsafe_allow_html=True)

    # ----------------------------------------------------------------
    # Pending credit payment requests — shop paid via UPI QR and submitted
    # their transaction ID; verify the money actually landed in your UPI
    # account, then Approve here to credit their balance (or Reject).
    # ----------------------------------------------------------------
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="card"><h3>💳 Pending Credit Payment Requests</h3>', unsafe_allow_html=True)
    pending_requests = credits.list_pending_requests()
    if not pending_requests:
        st.caption("No pending requests.")
    else:
        st.caption(
            "Check your UPI app / bank statement for each transaction ID below before "
            "approving — approving adds the credits immediately."
        )
        for req in pending_requests:
            r_cols = st.columns([2, 1.4, 1, 1.6, 1, 1])
            r_cols[0].markdown(f"**{req.get('shop_name', req['shop_id'])}**")
            r_cols[1].caption(req["shop_id"])
            r_cols[2].caption(f"{req['credits_requested']} cr")
            r_cols[3].caption(f"₹{req['amount']:.2f} · UTR `{req['utr_reference']}`")
            if r_cols[4].button("✅ Approve", key=f"approve_{req['id']}", use_container_width=True):
                credits.approve_request(req["id"])
                st.rerun()
            if r_cols[5].button("❌ Reject", key=f"reject_{req['id']}", use_container_width=True):
                credits.reject_request(req["id"])
                st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# --------------------------------------------------------------------------
# Garment types (mirrors the reference "AI trial room" app)
# --------------------------------------------------------------------------
GARMENT_TYPES = {
    "Shirt": "👔",
    "T-Shirt": "👕",
    "Trousers": "👖",
    "Pant Shirt": "👔",
    "Blazer": "🧥",
    "Jacket": "🧥",
    "Dress": "👗",
    "Skirt": "🩱",
    "Hoodie": "🥼",
    "Kurta": "🪡",
    "Kurta Pajama": "🪡",
    "Sherwani": "🎽",
    "Indo Western": "🎽",
    "Bandhgala": "🥻",
    "Nehru Jacket": "🥻",
    "Tuxedo": "🤵",
    "3 Piece Suit": "🤵",
    "Saree Fabric": "🧵",
    "Lehenga": "👗",
}

# NOTE: which garment types get "multi-zone design" handling (sherwani's
# contrast collar/border, etc.) and which get an auto-chosen matching bottom
# in AI-model mode is decided inside tryon_engine.py (_is_multi_zone_garment /
# _is_top_only_garment), purely from the garment_type string passed below —
# no extra wiring needed here when adding a new type to GARMENT_TYPES above.

FIT_STYLES = ["Realistic studio photo", "Outdoor natural light", "Fashion catalog shot", "Same background as user photo"]


def _open_image_correctly_oriented(file_like) -> Image.Image:
    """Opens an uploaded/captured photo and fixes sideways/upside-down
    orientation. Many phone cameras (especially iPhones) save a photo's
    pixels in one fixed orientation and store the "this is actually meant
    to be rotated 90°/180°/270°" instruction separately in EXIF metadata —
    Image.open() alone ignores that metadata, so without this the photo can
    come out sideways depending on how the phone was held when the shot was
    taken. ImageOps.exif_transpose() reads that tag and physically rotates
    the pixels to match, then we strip the (now redundant) orientation tag
    by converting to a plain RGB image."""
    img = Image.open(file_like)
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def image_input(label: str, key_prefix: str, help_text: str = "", multi: bool = False):
    """Renders an 'Upload' / 'Take Photo' toggle.

    multi=False (default): returns a single PIL Image or None.
    multi=True: lets the user add SEVERAL photos (different angles — e.g.
    front/back/border close-up of a saree, or front/side of a person) via
    repeated uploads or repeated camera shots, shows a thumbnail gallery
    with per-photo remove buttons, and returns a list of PIL Images
    (possibly empty).
    """
    mode_key = f"{key_prefix}_input_mode"
    if mode_key not in st.session_state:
        st.session_state[mode_key] = "Upload"
    gallery_key = f"{key_prefix}_gallery"
    if gallery_key not in st.session_state:
        st.session_state[gallery_key] = []  # list of PNG bytes
    cam_counter_key = f"{key_prefix}_camera_counter"
    if cam_counter_key not in st.session_state:
        st.session_state[cam_counter_key] = 0

    tab_cols = st.columns(2)
    with tab_cols[0]:
        if st.button("📁 Upload", key=f"{key_prefix}_tab_upload", use_container_width=True,
                     type="primary" if st.session_state[mode_key] == "Upload" else "secondary"):
            st.session_state[mode_key] = "Upload"
            st.rerun()
    with tab_cols[1]:
        if st.button("📸 Take Photo", key=f"{key_prefix}_tab_camera", use_container_width=True,
                     type="primary" if st.session_state[mode_key] == "Camera" else "secondary"):
            st.session_state[mode_key] = "Camera"
            st.rerun()

    def _to_bytes(img: Image.Image) -> bytes:
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="PNG")
        return buf.getvalue()

    if not multi:
        img = None
        if st.session_state[mode_key] == "Upload":
            file = st.file_uploader(
                label, type=["jpg", "jpeg", "png", "webp"],
                key=f"{key_prefix}_upload", label_visibility="collapsed",
            )
            if file is not None:
                try:
                    img = _open_image_correctly_oriented(file)
                except Exception:
                    st.error("Couldn't read that image file. Please try a different JPG/PNG.")
        else:
            st.caption(
                "⚠️ Camera not loading? If you opened this link inside WhatsApp/Instagram, "
                "tap **⋯ → Open in Chrome/Safari** — in-app browsers often block camera access. "
                "The **📁 Upload** tab always works there."
            )
            shot = st.camera_input(
                label, key=f"{key_prefix}_camera", label_visibility="collapsed",
                help="Allow camera access when your browser asks — works on both mobile and laptop.",
            )
            if shot is not None:
                try:
                    img = _open_image_correctly_oriented(shot)
                except Exception:
                    st.error("Couldn't read that photo. Please try capturing again.")
        if img is not None:
            st.image(img, use_container_width=True)
        elif help_text:
            st.info(help_text)
        return img

    # ---- multi=True: build/return a gallery of PIL Images ----
    if st.session_state[mode_key] == "Upload":
        files = st.file_uploader(
            label, type=["jpg", "jpeg", "png", "webp"],
            key=f"{key_prefix}_upload_multi", label_visibility="collapsed",
            accept_multiple_files=True,
            help="You can select multiple files at once — e.g. front, back, and close-up angles.",
        )
        if files:
            new_gallery = []
            for f in files:
                try:
                    new_gallery.append(_to_bytes(_open_image_correctly_oriented(f)))
                except Exception:
                    st.error(f"Couldn't read '{f.name}' — skipped.")
            st.session_state[gallery_key] = new_gallery
    else:
        st.caption(
            "⚠️ Camera not loading? If you opened this link inside WhatsApp/Instagram, "
            "tap **⋯ → Open in Chrome/Safari** — in-app browsers often block camera access. "
            "The **📁 Upload** tab always works there."
        )
        cam_key = f"{key_prefix}_camera_{st.session_state[cam_counter_key]}"
        shot = st.camera_input(
            label, key=cam_key, label_visibility="collapsed",
            help="Take a photo, then tap 'Add this photo' below to add another angle.",
        )
        if shot is not None:
            if st.button("➕ Add this photo", key=f"{key_prefix}_add_shot", use_container_width=True):
                try:
                    img = _open_image_correctly_oriented(shot)
                    st.session_state[gallery_key].append(_to_bytes(img))
                    st.session_state[cam_counter_key] += 1  # fresh camera widget for the next shot
                    st.rerun()
                except Exception:
                    st.error("Couldn't read that photo. Please try capturing again.")

    gallery = st.session_state[gallery_key]
    if gallery:
        st.caption(f"{len(gallery)} photo(s) added.")
        thumb_cols = st.columns(min(4, len(gallery)))
        for i, img_bytes in enumerate(gallery):
            with thumb_cols[i % len(thumb_cols)]:
                st.image(img_bytes, use_container_width=True)
                if st.button("✕ Remove", key=f"{key_prefix}_remove_{i}", use_container_width=True):
                    st.session_state[gallery_key].pop(i)
                    st.rerun()
    elif help_text:
        st.info(help_text)

    return [Image.open(io.BytesIO(b)).convert("RGB") for b in gallery]



# Color override palette — "Original" means "match the uploaded fabric exactly"
COLOR_PALETTE = [
    ("Original", None),
    ("White", "#F5F5F0"),
    ("Black", "#111111"),
    ("Navy", "#1B2A4A"),
    ("Sky Blue", "#7EC8E3"),
    ("Olive", "#6B6B3A"),
    ("Green", "#2E7D32"),
    ("Maroon", "#7B241C"),
    ("Beige", "#D8C3A5"),
    ("Grey", "#8A8A8A"),
    ("Brown", "#5C4033"),
    ("Pink", "#E8A0BF"),
    ("Yellow", "#F4D35E"),
]

# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts: {time, image_bytes, garment_type}
if "result_images" not in st.session_state:
    st.session_state.result_images = []
if "result_errors" not in st.session_state:
    st.session_state.result_errors = []
if "last_error" not in st.session_state:
    st.session_state.last_error = None
if "garment_type" not in st.session_state:
    st.session_state.garment_type = list(GARMENT_TYPES.keys())[0]
if "outfit_shirt_type" not in st.session_state:
    st.session_state.outfit_shirt_type = list(GARMENT_TYPES.keys())[0]  # top half's type in Full Outfit mode
if "color_choice" not in st.session_state:
    st.session_state.color_choice = COLOR_PALETTE[0][0]  # "Original"
if "outfit_mode" not in st.session_state:
    st.session_state.outfit_mode = False  # False = single garment, True = full outfit (shirt+trouser together)
if "shirt_color_choice" not in st.session_state:
    st.session_state.shirt_color_choice = COLOR_PALETTE[0][0]
if "trouser_color_choice" not in st.session_state:
    st.session_state.trouser_color_choice = COLOR_PALETTE[0][0]

# --------------------------------------------------------------------------
# Hero header + a top-right "☰ Menu" dropdown (st.popover) for account,
# credits, style preference and history — keeps all of that out of a
# permanent sidebar so the main column gets the full width. Matters most on
# mobile, where a Streamlit sidebar otherwise eats the whole screen just to
# show a logout button.
# --------------------------------------------------------------------------
hero_col, menu_col = st.columns([5, 1], gap="small")

with hero_col:
    st.markdown(
        """
        <div class="hero">
            <h1>🪄 AI Virtual Try-On Studio</h1>
            <p>Upload a photo of yourself and a garment or fabric — get an instant, realistic AI-generated try-on. Powered by Google Gemini (Nano Banana).</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with menu_col:
    st.markdown("<div style='height:0.55rem'></div>", unsafe_allow_html=True)
    with st.popover("☰"):
        st.markdown(f"### 🏬 {SHOP_NAME}")
        remaining = auth.remaining_quota(SHOP_ID, DAILY_LIMIT)
        if UNLIMITED_DAILY:
            st.caption("Unlimited generations today")
        elif DAILY_LIMIT == 0:
            st.caption("No free daily quota on this account — every generation uses a purchased credit.")
        else:
            st.caption(f"Today's generations left: **{remaining} / {DAILY_LIMIT}**")
            st.progress(min(1.0, max(0.0, 1 - remaining / DAILY_LIMIT)))

        current_credit_balance = credits.get_credit_balance(SHOP_ID)
        st.caption(f"Purchased credits (never expire): **{current_credit_balance}**")

        if credits.is_configured():
            with st.expander("💳 Buy Credits"):
                st.caption(
                    f"₹{credits.PRICE_PER_CREDIT:.0f} per credit. 1 credit = 1 generation, spent "
                    "automatically once your free daily quota runs out."
                )
                desired_credits = st.number_input(
                    "How many credits do you want?",
                    min_value=50, max_value=10000, value=50, step=1, key="buy_credits_qty",
                )
                purchase_amount = desired_credits * credits.PRICE_PER_CREDIT
                st.markdown(f"**Total to pay: ₹{purchase_amount:,.2f}**")

                if st.button("Generate Payment QR", use_container_width=True, key="gen_qr_btn"):
                    note = f"{SHOP_ID}-{int(desired_credits)}cr-{int(time.time())}"
                    st.session_state.pending_purchase = {
                        "credits": int(desired_credits),
                        "amount": purchase_amount,
                        "note": note,
                    }
                    st.rerun()

                pending_purchase = st.session_state.get("pending_purchase")
                if pending_purchase:
                    qr_img = credits.generate_upi_qr(pending_purchase["amount"], pending_purchase["note"])
                    st.image(qr_img, caption=f"Scan with any UPI app to pay ₹{pending_purchase['amount']:,.2f}",
                              width=220)
                    st.caption(f"UPI ID: `{credits.OWNER_UPI_ID}` · Reference: `{pending_purchase['note']}`")
                    st.caption(
                        "After paying, enter your UPI transaction ID / UTR below and submit. "
                        "Credits are added once the app owner verifies the payment."
                    )
                    utr_input = st.text_input("UPI Transaction ID / UTR", key="utr_input")
                    if st.button("✅ I've Paid — Submit for Approval", use_container_width=True,
                                 key="submit_payment_btn", type="primary"):
                        if not utr_input.strip():
                            st.warning("Please enter your UPI transaction ID / UTR reference.")
                        else:
                            credits.create_payment_request(
                                SHOP_ID, SHOP_NAME, pending_purchase["credits"],
                                pending_purchase["amount"], utr_input,
                            )
                            st.session_state.pending_purchase = None
                            st.success("Submitted! Your credits will be added once the payment is verified.")
                            st.rerun()

                recent_requests = credits.list_requests_for_shop(SHOP_ID, limit=3)
                if recent_requests:
                    st.markdown("**Recent requests**")
                    status_icon = {"pending": "⏳", "approved": "✅", "rejected": "❌"}
                    for r in recent_requests:
                        icon = status_icon.get(r["status"], "•")
                        st.caption(f"{icon} {r['credits_requested']} credits · ₹{r['amount']:.0f} · {r['status']}")
        else:
            st.caption("Credit top-ups aren't set up yet — contact the app owner.")

        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.shop = None
            st.rerun()

        st.markdown("---")
        st.markdown("### ⚙️ Preferences")

        status_html = (
            '<span class="status-pill-ok">● Service ready</span>'
            if SERVICE_CONFIGURED
            else '<span class="status-pill-bad">● Service unavailable</span>'
        )
        st.markdown(status_html, unsafe_allow_html=True)
        if not SERVICE_CONFIGURED:
            st.caption("The app owner needs to configure the Gemini API key on the server.")

        st.markdown("<br>", unsafe_allow_html=True)
        # Default to preserving the person's own background/lighting/angle
        # exactly (no surprise studio-backdrop swap) when dressing a real
        # uploaded photo — that's what a "virtual try-on" should mean by
        # default, without the user having to type "don't change the
        # background" into extra instructions every single time. The
        # AI-model mode has no original photo to preserve, so it still
        # defaults to generating a clean studio shot.
        _default_photo_mode = st.session_state.get("photo_mode", "person")
        _default_style_index = (
            FIT_STYLES.index("Same background as user photo")
            if _default_photo_mode == "person"
            else 0
        )
        style_choice = st.selectbox("Result style", FIT_STYLES, index=_default_style_index)

        st.markdown("---")
        st.markdown("### 📜 History")
        if st.session_state.history:
            for item in reversed(st.session_state.history[-6:]):
                cols = st.columns([1, 2])
                with cols[0]:
                    st.image(item["image_bytes"], use_container_width=True)
                with cols[1]:
                    st.caption(f"{item['garment_type']} · {item.get('color', 'Original')}")
                    st.caption(item["time"])
            if st.button("🗑️ Clear history", use_container_width=True):
                st.session_state.history = []
                st.rerun()
        else:
            st.caption("Your generated looks will appear here.")


if not SERVICE_CONFIGURED:
    st.error(
        "⚠️ This app isn't fully configured yet — the Gemini API key is missing on the "
        "server. If you're the app owner, add `GEMINI_API_KEY` to your `.env` file "
        "(local) or to `st.secrets` (Streamlit Cloud), then restart the app."
    )

# --------------------------------------------------------------------------
# Main layout
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# Step 1 — Choose Mode & Type. Collapsed by default (an expander) so it
# doesn't eat space once you've already picked — the label always shows
# your current choice even while closed.
# --------------------------------------------------------------------------
if "photo_mode" not in st.session_state:
    st.session_state.photo_mode = "person"

_mode_label = "📷 Use My Photo" if st.session_state.photo_mode == "person" else "🧑‍🎨 Generate AI Model"
_type_label = "👔+👖 Full Outfit" if st.session_state.outfit_mode else "👕 Single Garment"
with st.expander(f" Choose Mode & Type  —  {_mode_label} · {_type_label}", expanded=False, key="exp_mode_type"):
    mode_cols = st.columns(2)
    with mode_cols[0]:
        if st.button("📷 Use My Photo", key="mode_person", use_container_width=True,
                     type="primary" if st.session_state.photo_mode == "person" else "secondary"):
            st.session_state.photo_mode = "person"
            st.rerun()
    with mode_cols[1]:
        if st.button("🧑‍🎨 Generate AI Model", key="mode_ai", use_container_width=True,
                     type="primary" if st.session_state.photo_mode == "ai_model" else "secondary"):
            st.session_state.photo_mode = "ai_model"
            st.rerun()

    outfit_toggle_cols = st.columns(2)
    with outfit_toggle_cols[0]:
        if st.button("👕 Single Garment", key="mode_single_garment", use_container_width=True,
                     type="primary" if not st.session_state.outfit_mode else "secondary"):
            st.session_state.outfit_mode = False
            st.rerun()
    with outfit_toggle_cols[1]:
        if st.button("👔+👖 Full Outfit", key="mode_full_outfit", use_container_width=True,
                     type="primary" if st.session_state.outfit_mode else "secondary"):
            st.session_state.outfit_mode = True
            st.rerun()
    if st.session_state.outfit_mode:
        st.caption("Full Outfit mode checks a shirt fabric and a trouser fabric on you together, in a single result.")

# --------------------------------------------------------------------------
# Step 2 — Your Photo + Garment/Fabric photo(s), SIDE BY SIDE instead of
# stacked, so you can see what you uploaded and what you're trying on at
# the same time. Full Outfit mode gets a third column for the trouser fabric.
# --------------------------------------------------------------------------
person_images = []
ai_model_gender = None
garment_images = []
shirt_fabric_images = []
trouser_fabric_images = []

if st.session_state.outfit_mode:
    photo_col, shirt_col, trouser_col = st.columns(3, gap="medium")
else:
    photo_col, garment_col = st.columns(2, gap="medium")

with photo_col:
    if st.session_state.photo_mode == "person":
        st.markdown('<div class="card"><h3>Your Photo</h3>', unsafe_allow_html=True)
        person_images = image_input(
            "Upload or capture photo(s) — add more than one to generate on each",
            key_prefix="person",
            help_text="Tip: Good lighting + plain background gives the best results. "
                      "Make sure your face is clearly visible (not blocked by the phone). "
                      "Add multiple photos (e.g. different people/poses) to generate a "
                      "separate result for each one.",
            multi=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown('<div class="card"><h3><span class="step-badge">1b</span>AI Model Options</h3>', unsafe_allow_html=True)
        st.caption("No photo needed — Gemini will generate a fresh fashion model wearing your fabric.")
        if "ai_model_gender" not in st.session_state:
            st.session_state.ai_model_gender = "Female"
        gender_cols = st.columns(3)
        for i, g in enumerate(["Female", "Male", "Any"]):
            with gender_cols[i]:
                is_sel = st.session_state.ai_model_gender == g
                if st.button(g, key=f"gender_{g}", use_container_width=True,
                             type="primary" if is_sel else "secondary"):
                    st.session_state.ai_model_gender = g
                    st.rerun()
        ai_model_gender = st.session_state.ai_model_gender
        st.markdown("</div>", unsafe_allow_html=True)

if st.session_state.outfit_mode:
    with shirt_col:
        st.markdown('<div class="card"><h3>Shirt Fabric</h3>', unsafe_allow_html=True)
        _outfit_shirt_type_label = (
            f"{GARMENT_TYPES[st.session_state.outfit_shirt_type]} "
            f"{st.session_state.outfit_shirt_type}"
        )
        with st.expander(f"Top half type — {_outfit_shirt_type_label}", expanded=False):
            _outfit_shirt_names = list(GARMENT_TYPES.keys())
            _outfit_shirt_cols = st.columns(5)
            for i, name in enumerate(_outfit_shirt_names):
                with _outfit_shirt_cols[i % 5]:
                    is_sel = st.session_state.outfit_shirt_type == name
                    if st.button(
                        f"{GARMENT_TYPES[name]}\n{name}", key=f"outfit_shirt_type_{name}",
                        use_container_width=True, type="primary" if is_sel else "secondary",
                    ):
                        st.session_state.outfit_shirt_type = name
                        st.rerun()
        shirt_fabric_images = image_input(
            "Upload or capture the shirt/top fabric or garment photo(s) — add "
            "more angles for better accuracy",
            key_prefix="shirt_fabric",
            help_text="This fabric will be used for the top half of the outfit. "
                      "Add multiple photos (e.g. a wide shot + a close-up of the "
                      "weave/print, or front and back) for a more accurate result.",
            multi=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with trouser_col:
        st.markdown('<div class="card"><h3>Trouser Fabric</h3>', unsafe_allow_html=True)
        trouser_fabric_images = image_input(
            "Upload or capture the trouser/bottom fabric or garment photo(s) — "
            "add more angles for better accuracy",
            key_prefix="trouser_fabric",
            help_text="This fabric will be used for the bottom half of the outfit. "
                      "Add multiple photos (e.g. a wide shot + a close-up of the "
                      "weave/print, or front and back) for a more accurate result.",
            multi=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
else:
    with garment_col:
        st.markdown('<div class="card"><h3><span class="step-badge">2</span>Garment / Fabric Photo</h3>', unsafe_allow_html=True)
        garment_images = image_input(
            "Upload or capture the clothing item(s) or fabric(s) — add more than "
            "one to try several options",
            key_prefix="garment",
            help_text="Add multiple fabrics/garments to compare options — each one "
                      "will generate its own result. You'll be asked which fabric "
                      "goes on which photo if you've also uploaded multiple photos.",
            multi=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Step 3 — Garment Type / Colors. Also collapsible, closed by default, with
# the current choice shown right in the collapsed label.
# --------------------------------------------------------------------------
if st.session_state.outfit_mode:
    _colors_label = f"{st.session_state.shirt_color_choice} shirt · {st.session_state.trouser_color_choice} trouser"
    with st.expander(f"Shirt & Trouser Colors  —  {_colors_label}", expanded=False, key="exp_colors_outfit"):
        st.caption(
            "\"Original\" matches the exact color of each uploaded fabric photo. "
            "Pick a swatch to override it."
        )

        def _render_color_row(label: str, state_key: str, key_prefix: str) -> None:
            st.markdown(f"**{label}**")
            # Wrap into a fixed 5-column grid instead of one squished row of
            # 13 columns — keeps every swatch a comfortable tap-size on phones.
            ncols = 5
            row_cols = st.columns(ncols)
            for i, (cname, chex) in enumerate(COLOR_PALETTE):
                with row_cols[i % ncols]:
                    is_sel = st.session_state[state_key] == cname
                    swatch_class = "swatch swatch-original" if chex is None else "swatch"
                    swatch_style = "" if chex is None else f"background:{chex};"
                    sel_class = " swatch-selected" if is_sel else ""
                    st.markdown(
                        f'<div class="{swatch_class}{sel_class}" style="{swatch_style}" title="{cname}"></div>'
                        f'<div class="swatch-label">{"✓ " if is_sel else ""}{cname}</div>',
                        unsafe_allow_html=True,
                    )
                    if st.button("✓ Selected" if is_sel else "Select", key=f"{key_prefix}_{cname}",
                                 use_container_width=True, help=cname):
                        st.session_state[state_key] = cname
                        st.rerun()

        _render_color_row("Shirt color", "shirt_color_choice", "shirtcolor")
        _render_color_row("Trouser color", "trouser_color_choice", "trousercolor")
else:
    _gtype_label = f"{GARMENT_TYPES[st.session_state.garment_type]} {st.session_state.garment_type}"
    with st.expander(f" Select Garment Type  —  {_gtype_label}", expanded=False, key="exp_garment_type"):
        cols = st.columns(5)
        garment_names = list(GARMENT_TYPES.keys())

        for i, name in enumerate(garment_names):
            col = cols[i % 5]
            with col:
                is_selected = st.session_state.garment_type == name
                btn_label = f"{GARMENT_TYPES[name]}\n{name}"
                if st.button(btn_label, key=f"gtype_{name}", use_container_width=True,
                             type="primary" if is_selected else "secondary"):
                    st.session_state.garment_type = name
                    st.rerun()

        st.markdown(
            f'<span class="badge-pill">Selected: {GARMENT_TYPES[st.session_state.garment_type]} '
            f'{st.session_state.garment_type}</span>',
            unsafe_allow_html=True,
        )

    with st.expander(f"Choose Color  —  {st.session_state.color_choice}", expanded=False, key="exp_color_single"):
        st.caption("\"Original\" matches the exact color of your uploaded fabric photo. Pick a swatch to override it.")
        ncols = 5
        color_cols = st.columns(ncols)
        for i, (cname, chex) in enumerate(COLOR_PALETTE):
            with color_cols[i % ncols]:
                is_sel = st.session_state.color_choice == cname
                swatch_class = "swatch swatch-original" if chex is None else "swatch"
                swatch_style = "" if chex is None else f"background:{chex};"
                sel_class = " swatch-selected" if is_sel else ""
                st.markdown(
                    f'<div class="{swatch_class}{sel_class}" style="{swatch_style}" title="{cname}"></div>'
                    f'<div class="swatch-label">{"✓ " if is_sel else ""}{cname}</div>',
                    unsafe_allow_html=True,
                )
                if st.button("✓ Selected" if is_sel else "Select", key=f"color_{cname}",
                             use_container_width=True, help=cname):
                    st.session_state.color_choice = cname
                    st.rerun()

# ----------------------------------------------------------------
# Quick preview — a compact reminder of what you've uploaded, placed
# right next to the Generate button. On phones the page gets tall
# (upload steps are far above), so this stops you from having to
# scroll all the way back up to check your photo before generating.
# ----------------------------------------------------------------
st.markdown('<div class="card"><h3>🖼️ Your Selections</h3>', unsafe_allow_html=True)
preview_items = []  # list of (image_or_None, caption)

if st.session_state.photo_mode == "person":
    if person_images:
        for i, pimg in enumerate(person_images):
            preview_items.append((pimg, f"Photo {i + 1}"))
    else:
        preview_items.append((None, "⚠️ No photo yet"))
else:
    preview_items.append((None, f"🧑‍🎨 AI Model ({ai_model_gender})"))

if st.session_state.outfit_mode:
    if shirt_fabric_images:
        for i, simg in enumerate(shirt_fabric_images):
            label = "Shirt fabric" if len(shirt_fabric_images) == 1 else f"Shirt fabric {i + 1}"
            preview_items.append((simg, label))
    else:
        preview_items.append((None, "⚠️ No shirt fabric yet"))
    if trouser_fabric_images:
        for i, timg in enumerate(trouser_fabric_images):
            label = "Trouser fabric" if len(trouser_fabric_images) == 1 else f"Trouser fabric {i + 1}"
            preview_items.append((timg, label))
    else:
        preview_items.append((None, "⚠️ No trouser fabric yet"))
else:
    if garment_images:
        for i, gimg in enumerate(garment_images):
            preview_items.append((gimg, f"Garment {i + 1}"))
    else:
        preview_items.append((None, "⚠️ No garment yet"))

preview_cols = st.columns(min(4, max(1, len(preview_items))))
for i, (img, caption) in enumerate(preview_items[:8]):
    with preview_cols[i % len(preview_cols)]:
        if img is not None:
            st.image(img, use_container_width=True)
            st.caption(caption)
        else:
            st.markdown(f"<div style='padding-top:1.2rem;'>{caption}</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

extra_notes = st.text_area(
    "Extra instructions (optional)",
    placeholder="e.g. tuck in the shirt, full sleeves, keep my face and pose unchanged...",
    height=80,
    help="Sent directly to the AI and treated as high-priority — it will be followed "
         "unless it conflicts with keeping your face/identity unchanged.",
)

# ----------------------------------------------------------------
# Figure out how many results to generate and from which combos:
# - 1 photo + 1 fabric            -> 1 result
# - N photos + 1 fabric           -> N results (that fabric on each photo)
# - 1 photo + M fabrics           -> M results (each fabric on that photo)
# - N photos + M fabrics (both>1) -> ASK which fabric goes on which photo
# - AI Model mode + M fabrics     -> M results (a fresh model per fabric)
# - Full Outfit mode              -> 1 result per person photo, using the
#                                    single shirt fabric + single trouser
#                                    fabric together (jobs tagged "outfit")
# ----------------------------------------------------------------
jobs = []  # list of ("single", person_idx_or_None, garment_idx) or ("outfit", person_idx_or_None, None)
needs_pairing = False

if st.session_state.outfit_mode:
    if shirt_fabric_images and trouser_fabric_images:
        if st.session_state.photo_mode == "ai_model":
            jobs = [("outfit", None, None)]
        elif person_images:
            jobs = [("outfit", i, None) for i in range(len(person_images))]
elif st.session_state.photo_mode == "ai_model":
    jobs = [("single", None, g) for g in range(len(garment_images))]
elif person_images and garment_images:
    if len(person_images) > 1 and len(garment_images) > 1:
        needs_pairing = True
    elif len(person_images) > 1:
        jobs = [("single", i, 0) for i in range(len(person_images))]
    elif len(garment_images) > 1:
        jobs = [("single", 0, g) for g in range(len(garment_images))]
    else:
        jobs = [("single", 0, 0)]

if needs_pairing:
    st.markdown('<div class="card"><h3><span class="step-badge">3c</span>Match Fabric to Photo</h3>', unsafe_allow_html=True)
    st.caption(
        f"You uploaded {len(person_images)} photos and {len(garment_images)} fabrics — "
        "pick which fabric goes on each photo below."
    )
    pair_sig_key = "pairing_signature"
    current_sig = (len(person_images), len(garment_images))
    if st.session_state.get(pair_sig_key) != current_sig or "photo_fabric_pairs" not in st.session_state:
        # Reset defaults (index-aligned, wrapping) whenever the photo/fabric counts change.
        st.session_state.photo_fabric_pairs = [
            i % len(garment_images) for i in range(len(person_images))
        ]
        st.session_state[pair_sig_key] = current_sig

    for i, person_img in enumerate(person_images):
        cols = st.columns([1, 2])
        with cols[0]:
            st.image(person_img, caption=f"Photo {i + 1}", use_container_width=True)
        with cols[1]:
            chosen = st.selectbox(
                f"Fabric for Photo {i + 1}",
                options=list(range(len(garment_images))),
                format_func=lambda g: f"Fabric {g + 1}",
                index=st.session_state.photo_fabric_pairs[i],
                key=f"pair_select_{i}",
            )
            st.image(garment_images[chosen], width=90)
            st.session_state.photo_fabric_pairs[i] = chosen
    jobs = [("single", i, st.session_state.photo_fabric_pairs[i]) for i in range(len(person_images))]
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
generate_label = "✨ Generate Try-On" if len(jobs) <= 1 else f"✨ Generate {len(jobs)} Try-Ons"
# Total generations this shop can actually do right now: free daily quota
# (0 if none/exhausted) plus purchased credits — unless the shop has an
# unlimited daily_limit, in which case quota/credits don't matter at all.
_remaining_free_now = auth.remaining_quota(SHOP_ID, DAILY_LIMIT)
_credit_balance_now = credits.get_credit_balance(SHOP_ID)
_total_available_now = _remaining_free_now + _credit_balance_now
generate_clicked = st.button(
    generate_label,
    use_container_width=True,
    type="primary",
    disabled=not SERVICE_CONFIGURED or (not UNLIMITED_DAILY and _total_available_now <= 0),
)

st.markdown('<div class="card" style="margin-top: 1rem;"><h3>Result</h3>', unsafe_allow_html=True)
result_placeholder = st.empty()

if generate_clicked:
    st.session_state.last_error = None
    remaining_now = auth.remaining_quota(SHOP_ID, DAILY_LIMIT)
    global_remaining = rate_limiter.global_remaining_quota()
    credit_balance = credits.get_credit_balance(SHOP_ID)
    # Free daily quota is used first; once it runs out (or if the shop
    # has zero free quota to begin with), purchased credits cover the
    # rest. Unlimited shops (negative daily_limit) skip this check
    # entirely.
    total_available = remaining_now + credit_balance
    if not SERVICE_CONFIGURED:
        st.session_state.last_error = "Service isn't configured (missing API key on server)."
    elif not UNLIMITED_DAILY and len(jobs) > total_available:
        st.session_state.last_error = (
            f"This would use {len(jobs)} generations, but you only have {remaining_now} free "
            f"generation(s) left today (limit {DAILY_LIMIT}/day) plus {credit_balance} purchased "
            f"credit(s) — {total_available} total. Remove some photos/fabrics, or buy more "
            "credits from the sidebar."
        )
    elif rate_limiter.MAX_DAILY_GENERATIONS > 0 and len(jobs) > global_remaining:
        st.session_state.last_error = (
            "This would exceed the organization-wide daily generation cap "
            "(protecting the shared API budget). Please try again tomorrow, "
            "or contact the app owner."
        )
    elif st.session_state.photo_mode == "person" and not person_images:
        st.session_state.last_error = "Please upload your photo first (or switch to 'Generate AI Model')."
    elif st.session_state.outfit_mode and not (shirt_fabric_images and trouser_fabric_images):
        st.session_state.last_error = "Please upload both a shirt fabric and a trouser fabric photo first."
    elif not st.session_state.outfit_mode and not garment_images:
        st.session_state.last_error = "Please upload a garment / fabric photo first."
    elif not jobs:
        st.session_state.last_error = "Nothing to generate — check your photos/fabrics above."

    if st.session_state.last_error:
        with result_placeholder.container():
            st.error(st.session_state.last_error)
    else:
        with result_placeholder.container():

            # Snapshot everything the worker threads need from
            # st.session_state into plain local variables BEFORE
            # spawning any threads. st.session_state is only safely
            # readable on Streamlit's own script-run thread — reading it
            # from a ThreadPoolExecutor worker raises
            # `st.session_state has no attribute "..."`, which is what
            # broke the previous version of this parallel-generation code.
            photo_mode_snap = st.session_state.photo_mode
            garment_type_snap = st.session_state.garment_type
            outfit_shirt_type_snap = st.session_state.outfit_shirt_type
            color_choice_snap = st.session_state.color_choice
            shirt_color_choice_snap = st.session_state.shirt_color_choice
            trouser_color_choice_snap = st.session_state.trouser_color_choice

            def _run_one_job(job_kind, p_idx, g_idx):
                """Runs a single generation. Pure Python + network I/O
                only (no Streamlit calls — everything needed is passed
                in or read from the plain local snapshots above, never
                from st.session_state) so it's safe to run this on a
                background thread — lets several photos/fabrics in one
                click generate in parallel instead of one-by-one, which
                is the main speed win for multi-job batches. The shared
                rate_limiter.generation_slot() still caps how many of
                these run against the Gemini API at the same instant."""
                with rate_limiter.generation_slot():
                    if job_kind == "outfit":
                        shirt_selected_color = dict(COLOR_PALETTE).get(shirt_color_choice_snap)
                        trouser_selected_color = dict(COLOR_PALETTE).get(trouser_color_choice_snap)
                        result_img, result_text = _generate_with_key_rotation(
                            generate_full_outfit_tryon,
                            shirt_image=shirt_fabric_images,
                            trouser_image=trouser_fabric_images,
                            person_image=person_images[p_idx] if p_idx is not None else None,
                            ai_model_gender=ai_model_gender if photo_mode_snap == "ai_model" else None,
                            style=style_choice,
                            extra_notes=extra_notes,
                            model_name=MODEL_NAME,
                            shirt_color_override_name=(
                                shirt_color_choice_snap if shirt_selected_color else None
                            ),
                            shirt_color_override_hex=shirt_selected_color,
                            trouser_color_override_name=(
                                trouser_color_choice_snap if trouser_selected_color else None
                            ),
                            trouser_color_override_hex=trouser_selected_color,
                            shirt_garment_type=outfit_shirt_type_snap,
                        )
                        result_label = "Full Outfit"
                        result_color = f"{shirt_color_choice_snap} shirt / {trouser_color_choice_snap} trouser"
                    else:
                        selected_color = dict(COLOR_PALETTE).get(color_choice_snap)
                        result_img, result_text = _generate_with_key_rotation(
                            generate_tryon,
                            garment_image=garment_images[g_idx],
                            garment_type=garment_type_snap,
                            person_image=person_images[p_idx] if p_idx is not None else None,
                            ai_model_gender=ai_model_gender if photo_mode_snap == "ai_model" else None,
                            style=style_choice,
                            extra_notes=extra_notes,
                            model_name=MODEL_NAME,
                            color_override_name=(
                                color_choice_snap if selected_color else None
                            ),
                            color_override_hex=selected_color,
                        )
                        result_label = garment_type_snap
                        result_color = color_choice_snap
                return result_img, result_text, result_label, result_color

            results = []
            errors = []
            progress = st.progress(0.0) if len(jobs) > 1 else None
            queue_len = rate_limiter.current_queue_length()
            base_msg = (
                f"Generating {len(jobs)} try-ons in parallel... " if len(jobs) > 1 else
                "Generating your virtual try-on... "
            ) + (
                "this can take 20–70 seconds ⏳ (shirt + trousers are applied in two steps "
                "for accuracy, auto-retries if the API is briefly busy)"
                if st.session_state.outfit_mode else
                "this can take 10–40 seconds ⏳ (auto-retries if the API is briefly busy)"
            )
            spinner_msg = (
                base_msg + f" (queued behind {queue_len} other request(s) right now)"
                if queue_len > 0 else base_msg
            )
            max_workers = max(1, min(len(jobs), rate_limiter.MAX_CONCURRENT_GENERATIONS))
            outcomes = [None] * len(jobs)  # (result_img, result_text, result_label, result_color) or None
            job_errors = [None] * len(jobs)
            with st.spinner(spinner_msg):
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_idx = {
                        executor.submit(_run_one_job, job_kind, p_idx, g_idx): idx
                        for idx, (job_kind, p_idx, g_idx) in enumerate(jobs)
                    }
                    done_count = 0
                    for future in concurrent.futures.as_completed(future_to_idx):
                        idx = future_to_idx[future]
                        try:
                            outcomes[idx] = future.result()
                        except (GeminiTryOnError, Exception) as e:  # noqa: BLE001
                            job_errors[idx] = str(e)
                            if DEBUG_MODE:
                                st.exception(e)
                        done_count += 1
                        if progress is not None:
                            progress.progress(done_count / len(jobs))

            # Bookkeeping (credits/usage/history) happens back on the
            # main thread, in original job order, once every generation
            # has finished.
            for idx in range(len(jobs)):
                if job_errors[idx] is not None:
                    errors.append(job_errors[idx])
                    continue
                result_img, result_text, result_label, result_color = outcomes[idx]
                results.append((result_img, result_text))
                if not UNLIMITED_DAILY and auth.remaining_quota(SHOP_ID, DAILY_LIMIT) <= 0:
                    # Free daily quota already used up (or this shop has
                    # zero free quota) — spend from purchased credits instead.
                    credits.spend_credit(SHOP_ID, 1)
                else:
                    auth.increment_usage(SHOP_ID)
                auth.increment_usage(auth.GLOBAL_KEY)  # org-wide cap tracking

                buf = io.BytesIO()
                result_img.save(buf, format="PNG")
                st.session_state.history.append({
                    "image_bytes": buf.getvalue(),
                    "garment_type": result_label,
                    "color": result_color,
                    "time": datetime.now().strftime("%H:%M:%S"),
                })

            st.session_state.result_images = results
            st.session_state.result_errors = errors
            st.session_state.last_error = None

result_images = st.session_state.get("result_images") or []
result_errors = st.session_state.get("result_errors") or []

if result_images or result_errors:
    with result_placeholder.container():
        if result_errors:
            for err in result_errors:
                st.error(f"One generation failed: {err}")
        if result_images:
            res_cols = st.columns(min(2, len(result_images))) if len(result_images) > 1 else [st.container()]
            for i, (img, text) in enumerate(result_images):
                with res_cols[i % len(res_cols)]:
                    st.markdown('<div class="result-frame">', unsafe_allow_html=True)
                    st.image(img, use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                    buf = io.BytesIO()
                    img.save(buf, format="PNG")
                    st.download_button(
                        f"⬇️ Download {'Result ' + str(i + 1) if len(result_images) > 1 else 'Result'}",
                        data=buf.getvalue(),
                        file_name=f"virtual_tryon_{int(time.time())}_{i + 1}.png",
                        mime="image/png",
                        use_container_width=True,
                        key=f"download_{i}",
                    )
                    if text:
                        st.caption(text)
elif not generate_clicked:
    with result_placeholder.container():
        if st.session_state.last_error:
            st.error(st.session_state.last_error)
        else:
            st.info("Your AI-generated try-on will appear here after you click **Generate Try-On**.")

st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    """
    <div style="text-align:center; color:#8b7fae; margin-top:2rem; font-size:0.85rem;">
        Built with Streamlit + Google Gemini (Nano Banana) · For best results use a clear, well-lit photo.
    </div>
    """,
    unsafe_allow_html=True,
)
