# 🪄 AI Virtual Try-On Studio

Streamlit + Google Gemini (Nano Banana / `gemini-2.5-flash-image`) based
Virtual Try-On system, built to **sell access to multiple shops** — each
shop gets its own login and a daily generation limit, so your one shared
Gemini API key doesn't get drained by a single shop.

## 📁 Project Structure

```
ai-virtual-tryon/
├── app.py                    # Streamlit UI + shop login gate
├── auth.py                   # Shop login + daily usage-limit tracking
├── manage_shops.py           # CLI to add/remove/list shops
├── tryon_engine.py           # Gemini API wrapper (prompt + image generation)
├── requirements.txt
├── shops.json.example        # Format reference (real file: shops.json, gitignored)
├── .env.example               # Copy to .env for local dev
└── .streamlit/
    ├── config.toml            # Theme
    └── secrets.toml.example   # Copy for Streamlit Cloud deployment
```

---

## Part 1 — Local setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env`, add your Gemini key (get one free at https://aistudio.google.com/apikey):
```
GEMINI_API_KEY=your_key_here
```

## Part 2 — Add your shops (dukandars)

Every shop needs its own Shop ID + password so:
- they can't see each other's data
- you can set (and enforce) a **daily generation limit per shop**, so your
  shared API key's cost stays predictable

```bash
python manage_shops.py add sharma_cloth "Sharma Cloth House" MyPass123 20
python manage_shops.py add gupta_textiles "Gupta Textiles" Secret456 15
python manage_shops.py list
```

- 3rd argument = password you give that shop to log in
- 4th argument = max try-on generations **per day** for that shop (use `-1`
  for unlimited — not recommended if you're paying for the API; `0` gives a
  real zero free quota so that shop must rely entirely on purchased credits)

This writes to `shops.json` (git-ignored — never commit real passwords).
To remove a shop: `python manage_shops.py remove gupta_textiles`

Run locally to test the login screen:
```bash
streamlit run app.py
```

---

## Part 3 — Get it onto shopkeepers' mobile phones

**Important: this is a website, not a native app.** Nobody needs to install
anything from the Play Store. Once it's hosted online, any dukandar opens
the link in Chrome on their phone and it just works — exactly like the
demo you saw. The steps below are about **putting it online** so the link
works from anywhere, not just your own laptop.

### Option A — Streamlit Community Cloud (free, easiest, good for starting out)

1. Push this folder to a **private** GitHub repo (don't make it public —
   `shops.json`/`.env` are git-ignored, but keep the repo private anyway).
2. Go to https://share.streamlit.io → "New app" → pick your repo → main
   file `app.py`.
3. In **App settings → Secrets**, paste:
   ```toml
   GEMINI_API_KEY = "your_key_here"
   ```
4. Deploy. You'll get a URL like `https://your-app.streamlit.app` —
   send this link to every shop (or generate a QR code for it so they can
   scan it in-store).

   ⚠️ **Limitation:** on the free tier, the server's local disk (where
   `shops.json` / `usage.json` live) can reset when the app redeploys or
   sleeps from inactivity. For a handful of shops this is usually fine day
   to day, but for a paid product read Option B.

   **Getting shops added reliably on this tier:** running
   `python manage_shops.py add ...` on your own laptop only edits the
   `shops.json` file on *your* machine — it has zero effect on the deployed
   app, since Streamlit Cloud runs on a completely separate server with its
   own disk. For shops you (the owner) create, define them directly in the
   deployed app's **Secrets** instead, e.g.:
   ```toml
   GEMINI_API_KEY = "your_key_here"

   [[shops]]
   id = "sharma_cloth"
   name = "Sharma Cloth House"
   password_hash = "PASTE_THE_HASH_FROM_YOUR_LOCAL_shops.json_HERE"
   daily_limit = 20
   ```
   Run `python manage_shops.py add sharma_cloth "Sharma Cloth House" MyPass123 20`
   locally first, then open the local `shops.json` it created and copy the
   `password_hash` value into the block above (never put the plain
   password itself in secrets). Save the secrets — the app restarts
   automatically and that shop can log in. Shops that use the in-app
   **Sign Up** tab don't need this — they're written straight to the
   deployed server's own `shops.json` and can log in immediately (subject
   to the disk-reset limitation above).

### Option B — Render / Railway (~$7/month, persistent, recommended once you're actually charging shops)

1. Push the code to GitHub (private repo).
2. Create a new **Web Service** on https://render.com (or Railway).
   - Build command: `pip install -r requirements.txt`
   - Start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
3. Add a **persistent disk** mounted at the project folder (so `shops.json`
   and `usage.json` survive restarts/redeploys).
4. Add environment variable `GEMINI_API_KEY` in the dashboard (never in code).
5. You get a permanent URL — optionally attach your own domain (e.g.
   `tryon.yourbrand.com`) in the platform's domain settings.

### Making it feel like a real "app" on their phone (optional, free)

Once hosted, tell each shop:
1. Open the link in Chrome (Android) or Safari (iPhone).
2. Tap the browser menu → **"Add to Home Screen"**.
3. It now sits on their home screen with an icon, opens full-screen like a
   normal app — no app store needed.

---

## Part 4 — Running costs (what you're actually paying for)

Gemini image generation is billed **per image** by Google
(see current pricing: https://ai.google.dev/pricing). With 10+ shops and a
shared key, your monthly cost ≈ `(sum of each shop's daily_limit) × price
per image × days`. The daily limits in `manage_shops.py` are your main
lever to keep this predictable — set them conservatively at first and
raise them for shops that are paying more.

---

## 🩹 What was fixed along the way

- **Cropping** — the model defaulted to a square (1:1) output regardless of
  the uploaded photo's shape. The app now detects the person photo's real
  aspect ratio (portrait/landscape) and requests a matching output ratio,
  plus explicitly instructs the model not to crop the pose.
- **Errors after 2-3 generations in a row** — usually a free-tier
  rate-limit (429). The app now automatically retries (with a short wait)
  before showing an error, and the error message explains it's a rate
  limit rather than a generic failure.
- **Inconsistent fabric color across regenerations** — the model was
  "reinterpreting" the fabric color slightly differently each time. The
  app now extracts the fabric's dominant color as a hex code and tells the
  model to match it exactly, with lower generation randomness.
- **API key & model name** are never shown or editable in the UI — backend
  only, via `.env` / `st.secrets`.

## 📸 Feature: upload OR take a live photo

Both "Your Photo" and "Garment / Fabric Photo" have two tabs:
- **📁 Upload** — pick an existing file
- **📸 Take Photo** — opens the device's camera (phone or laptop webcam)
  directly in the browser using Streamlit's built-in `st.camera_input`, so a
  shopkeeper can snap the fabric or the customer right there in-store
  without needing to save a photo first.

## 🛡️ Feature: protection against traffic spikes / server crashes

With 10+ shops sharing ONE Gemini API key, heavy simultaneous usage could
crash requests or blow your budget. Two protections run automatically
(`rate_limiter.py`):

1. **Concurrency queue** — only a few generations (default 3, set via
   `MAX_CONCURRENT_GENERATIONS` in `.env`) run at once. If more shops click
   Generate at the same moment, the rest wait in a queue (shown in the
   spinner text: "queued behind N other requests") instead of all hammering
   Gemini simultaneously and triggering a wave of rate-limit errors.
2. **Jittered retry backoff** — when a rate limit does happen, each request
   waits a slightly randomized amount before retrying (`tryon_engine.py`),
   so many shops don't all retry at the exact same instant and re-trigger
   the same limit again.
3. **Optional organization-wide daily cap** — `MAX_DAILY_GENERATIONS` in
   `.env` caps total paid-engine generations across ALL shops combined,
   on top of each shop's own `daily_limit`. This catches the case where 20
   shops each stay under their own limit but the sum is still too much.

These are best-effort protections for a small/medium deployment — for real
scale, also set up billing alerts in Google AI Studio / Cloud Console.

## 👥 Feature: shop self-signup + admin usage dashboard

The login screen now has two tabs:
- **🔑 Login** — existing shops sign in as before.
- **🆕 Sign Up** — new shops can create their own account directly (Shop
  ID, business name, password). Self-signed-up shops start on a **zero**
  free daily quota (`auth.SIGNUP_DEFAULT_LIMIT = 0`) — they must buy credits
  (see below) before they can generate anything. This is intentional: it
  stops people from dodging payment by creating unlimited throwaway shop
  accounts. Raise a specific shop's quota later with
  `manage_shops.py add <id> <name> <same_password> <limit>` if you want to
  give a particular shop free generations.

To see usage across every shop in one place, mark yourself as an admin:
```bash
python manage_shops.py admin sharma_cloth
```
Logging in as that shop now shows an **Admin Dashboard** instead of the
try-on tool — every shop's name, ID, today's generation count, and their
limit, sorted by most active first, plus the organization-wide total.
Remove admin rights with `python manage_shops.py unadmin <id>`.

## 📸 Feature: multiple photos + fabrics = batch generation

Both "Your Photo" and "Garment / Fabric Photo" accept **multiple photos** —
upload several files at once, or take repeated camera shots (tap "➕ Add
this photo" after each capture) to build a small gallery, with a
"✕ Remove" button under each thumbnail.

How multiple photos/fabrics are handled — each combination generates a
**separate result image**, shown side by side with its own download button:

| You uploaded | What happens |
|---|---|
| 1 photo + 1 fabric | 1 result (normal) |
| N photos + 1 fabric | N results — that fabric tried on every photo |
| 1 photo + M fabrics | M results — every fabric tried on that photo |
| N photos + M fabrics (both > 1) | The app **asks you** which fabric goes with which photo (a picker appears — "Match Fabric to Photo"), then generates one result per pairing you confirm |
| "Generate AI Model" + M fabrics | M results — a fresh AI model generated per fabric |

The Generate button shows how many results a click will produce (e.g.
"✨ Generate 3 Try-Ons"), and each one counts against the shop's daily
limit — the app checks upfront that enough of the limit remains before
starting the batch.

## 🧑‍🎨 Feature: no photo? Generate an AI model instead

At the top of the form, choose **"Generate AI Model"** instead of **"Use My
Photo"**. In this mode, no user photo is needed at all — pick Male / Female
/ Any, upload just the fabric, and Gemini generates a brand-new fashion
model wearing that fabric, exactly like a catalog shoot. Great for shops
whose customers don't want to upload their own photo, or for showing a
fabric on a model before a customer even walks in.

## 🌐 Local port

The app now runs on **port 8505** by default (set in `.streamlit/config.toml`).
Run it the same way as before — `streamlit run app.py` — and open
`http://localhost:8505`.

## ⚡ Speed improvements

- **Parallel batch generation** — when a click produces more than one
  result (multiple photos and/or fabrics), all of them now generate
  **at the same time** instead of one after another, bounded by the same
  `MAX_CONCURRENT_GENERATIONS` slot used for cross-shop protection. A batch
  of 3 results that used to take ~3x as long now finishes in roughly the
  time of the slowest single generation.
- **Smaller upload payloads** — photos are resized to a slightly smaller
  max dimension before being sent to Gemini (`TRYON_MAX_SIDE` in `.env`,
  default `1024`), which uploads faster with no visible quality loss.
- **Reused API client + shorter retry backoff** — the Gemini client is now
  created once and reused across generations instead of being rebuilt every
  time, and the retry wait after a transient error is shorter
  (`TRYON_RETRY_WAIT_SECONDS`, default `5`).

## 🎨 Feature: color override

Under "Choose Color", **"Original"** uses the exact color/pattern of the
uploaded fabric photo. Any other swatch (White, Black, Olive, Navy, etc.)
overrides just the color while keeping the fabric's texture — useful when a
shop wants to show the same fabric print in different shades.

## 🔧 Tips for best results

- Well-lit, front-facing photo, plain-ish background.
- Face reasonably visible (not fully hidden behind a phone).
- One person only, one garment/fabric only, per photo.

## 🛠️ Troubleshooting

| Problem | Fix |
|---|---|
| Login screen won't accept a shop's password | Re-run `manage_shops.py add` for that shop with a new password |
| Sidebar shows "Service unavailable" | Set `GEMINI_API_KEY` in `.env`/secrets, restart |
| "Daily generation limit reached" | Expected — raise it: `manage_shops.py add <id> <name> <same_password> <new_limit>` |
| "blocked by Gemini's safety filters" | Try a different, more neutral photo |
| Image still looks cropped | Make sure the uploaded photo isn't already cropped square before upload |
| Rate limit errors even after retries | Your Gemini key's tier is too low for concurrent shops — upgrade billing tier in Google AI Studio |
| Want full tracebacks while developing | Set `TRYON_DEBUG=true` in `.env` |
| A shop that signed up through the app disappeared after a redeploy | You haven't set up Google Sheets storage yet — see the next section. Re-add the shop manually in the meantime (`manage_shops.py add ...`, or from the Admin Dashboard if it offers that). |

## 💾 Persistent storage (Google Sheets) — do this before going live

**Why you need this:** Streamlit Community Cloud rebuilds the live server
from scratch, straight from your GitHub repo, every time you push new
code. Shop signups, credit purchases, and daily usage counters only ever
get written to a local JSON file on the server while it's running — they
are **never** part of your git repo, so the next redeploy wipes them out.
A Google Sheet lives completely outside GitHub/Streamlit Cloud, so once
it's connected, that data survives redeploys, restarts, and sleep/wake
cycles untouched.

Until you do this setup, the app keeps working exactly as before (local
JSON files) — nothing breaks, you just stay exposed to the data-loss issue
above. It's a one-time, ~15 minute setup.

### Step 1 — create a Google Cloud service account

1. Go to [console.cloud.google.com](https://console.cloud.google.com/) and
   create a new project (or pick an existing one).
2. In the search bar, find **"Google Sheets API"** → click **Enable**.
   Do the same for **"Google Drive API"**.
3. Go to **APIs & Services → Credentials → Create Credentials → Service
   account**. Give it any name (e.g. "tryon-app"). Skip the optional role
   steps, click **Done**.
4. Click into the service account you just created → **Keys** tab →
   **Add Key → Create new key → JSON**. This downloads a `.json` file —
   keep it safe, it's effectively a password.

### Step 2 — create the Google Sheet and share it

1. Go to [sheets.google.com](https://sheets.google.com) and create a new,
   blank spreadsheet. Name it whatever you like (e.g. "TryOn App Data").
   You don't need to create any tabs/columns yourself — the app creates
   them automatically the first time it needs them.
2. Copy the Sheet's ID from its URL:
   `https://docs.google.com/spreadsheets/d/`**`THIS_LONG_ID_PART`**`/edit`
3. Open the downloaded JSON key file, find the `"client_email"` field
   (looks like `xxx@yyy.iam.gserviceaccount.com`). Click **Share** on your
   Google Sheet and share it with that exact email address, with
   **Editor** access.

### Step 3 — add the credentials to the app

**On Streamlit Community Cloud:** open your app → **Settings → Secrets**,
and add (this is TOML format — see `.streamlit/secrets.toml.example` in
this repo for the exact field list to copy from the downloaded JSON key):

```toml
GOOGLE_SHEET_ID = "the_id_you_copied_above"

[gcp_service_account]
type = "service_account"
project_id = "..."
# ...every other field from the downloaded JSON key, same names
```

**Running locally:** add to your `.env` file instead:
```
GOOGLE_SHEET_ID=the_id_you_copied_above
GOOGLE_SERVICE_ACCOUNT_JSON={"type": "service_account", "project_id": "...", ...}
```
(paste the entire downloaded JSON file's content as one line for
`GOOGLE_SERVICE_ACCOUNT_JSON`)

### Step 4 — migrate your existing data (optional, one-time)

If you already have shops/credits/usage in your local JSON files that you
want to keep, run this once from your own machine (after completing Steps
1-3 in your local `.env`):
```
python migrate_to_sheets.py
```

### Step 5 — restart the app

Redeploy / restart the app once after adding the secrets. From then on,
shop signups, credit balances, and usage all read/write the Google Sheet
automatically — you can open the Sheet directly any time to see the live
data, or even hand-edit it if you ever need to.

---
Made with ❤️ using Streamlit + Google Gemini.
