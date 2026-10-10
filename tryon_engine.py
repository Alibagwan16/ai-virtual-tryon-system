"""
tryon_engine.py
----------------
Wraps the Google Gemini image model ("Nano Banana" — gemini-2.5-flash-image)
to blend a person's photo with a garment/fabric photo into a realistic
virtual try-on image.
"""

from __future__ import annotations

import io
import os
import logging
import threading
import time
import random
import colorsys
from typing import Optional, Tuple

from PIL import Image

logger = logging.getLogger("tryon_engine")

# Resize large phone photos before sending — still keeps payloads reasonable,
# but accuracy matters more than shaving a second off upload time, so this
# stays high enough that fine fabric texture/pattern survives the resize.
# Configurable via env (TRYON_MAX_SIDE) if a deployment wants to trade detail
# for speed.
MAX_SIDE = int(os.getenv("TRYON_MAX_SIDE", "1280"))

# Low temperature = less creative drift = the model sticks much closer to the
# actual color/pattern of the uploaded fabric photo instead of reinterpreting
# it differently on every regeneration. Kept low so the FIRST generation is
# already as faithful as possible, not just later retries.
GENERATION_TEMPERATURE = 0.1

# Gemini 2.5 Flash Image only supports these output aspect ratios.
SUPPORTED_ASPECT_RATIOS = ["21:9", "16:9", "4:3", "3:2", "1:1", "9:16", "3:4", "2:3", "5:4", "4:5"]

# Free-tier API keys hit per-minute rate limits quickly. Retry a couple of
# times with a short wait instead of failing immediately on the 2nd/3rd click.
MAX_RETRIES = 2
RETRY_WAIT_SECONDS = int(os.getenv("TRYON_RETRY_WAIT_SECONDS", "5"))

# Reusing one genai.Client per API key avoids re-doing client setup/handshake
# work on every single generation (small but adds up when a shop batches
# several photos/fabrics in one click).
_CLIENT_CACHE: dict = {}
_CLIENT_LOCK = threading.Lock()


def _get_client(api_key: str):
    from google import genai
    with _CLIENT_LOCK:
        client = _CLIENT_CACHE.get(api_key)
        if client is None:
            client = genai.Client(api_key=api_key)
            _CLIENT_CACHE[api_key] = client
        return client


class GeminiTryOnError(Exception):
    """Raised when the Gemini API fails to return a usable image."""


def _resize_for_api(img: Image.Image, max_side: int = MAX_SIDE) -> Image.Image:
    """Downscale large images so requests are smaller/faster and less likely
    to hit payload limits, while keeping aspect ratio."""
    w, h = img.size
    if max(w, h) <= max_side:
        return img
    scale = max_side / float(max(w, h))
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return img.resize(new_size, Image.LANCZOS)


def _closest_aspect_ratio(img: Image.Image) -> str:
    """Picks the supported Gemini output aspect ratio closest to the
    person photo's real aspect ratio, so the result keeps the same framing
    (full body/half body) instead of being force-cropped into a square."""
    w, h = img.size
    target = w / h

    def ratio_value(r: str) -> float:
        a, b = r.split(":")
        return int(a) / int(b)

    return min(SUPPORTED_ASPECT_RATIOS, key=lambda r: abs(ratio_value(r) - target))


def _is_background_like(r: int, g: int, b: int) -> bool:
    """True for a pixel that looks like a plain studio backdrop (near-white,
    near-black, or a desaturated grey) rather than actual garment material.
    Readymade garment/product photos very often have exactly this kind of
    background filling a large part of the frame."""
    h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
    return (v > 0.90 and s < 0.12) or v < 0.08 or (s < 0.08 and 0.15 < v < 0.95)


def _foreground_pixels(pixels):
    """Filters out background-like pixels from a pixel list, UNLESS doing so
    would strip out most of the image — in which case the garment itself is
    probably genuinely white/black/grey, so we keep everything rather than
    bucket an almost-empty set."""
    foreground = [p for p in pixels if not _is_background_like(*p)]
    return foreground if len(foreground) >= len(pixels) * 0.15 else pixels


def dominant_color_hex(img: Image.Image) -> str:
    """Estimates the single dominant color of a fabric/garment photo and
    returns it as a #RRGGBB hex string. Used to give the model an exact,
    unambiguous color target instead of relying on it to "guess" the shade
    from the raw pixels each time, which is what caused different colors
    across regenerations of the same fabric photo.

    IMPORTANT: readymade garment / product photos often sit on a plain
    white/grey studio backdrop that can fill more of the frame than the
    garment itself — without filtering those pixels out first, the
    "dominant color" ends up being the background, not the garment, which
    then feeds a wrong color straight into the prompt and actively confuses
    the model (told to match white when the reference is clearly red, say).
    _foreground_pixels() strips that backdrop out before bucketing."""
    small = img.convert("RGB").resize((64, 64))
    pixels = _foreground_pixels(list(small.getdata()))

    # Bucket similar hues together and take the most common bucket's average
    # color — more robust than a plain average, which gets muddied by
    # shadows, folds and any remaining background peeking around the edges.
    buckets = {}
    for r, g, b in pixels:
        h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
        key = (round(h * 12), round(s * 4), round(v * 4))
        acc = buckets.setdefault(key, [0, 0, 0, 0])
        acc[0] += r
        acc[1] += g
        acc[2] += b
        acc[3] += 1

    best_key = max(buckets, key=lambda k: buckets[k][3])
    r_sum, g_sum, b_sum, count = buckets[best_key]
    r_avg, g_avg, b_avg = r_sum // count, g_sum // count, b_sum // count
    return f"#{r_avg:02X}{g_avg:02X}{b_avg:02X}"


def dominant_color_hex_multi(imgs) -> str:
    """Like dominant_color_hex, but combines pixels from SEVERAL reference
    photos of the SAME fabric (e.g. different angles, a close-up of the
    weave, a wider shot of the whole piece) into one bucket count. This is
    more robust than reading a single photo, which can be skewed by one
    shot being under/over-exposed or shot at an angle that catches glare."""
    imgs = list(imgs)
    if len(imgs) == 1:
        return dominant_color_hex(imgs[0])

    # Filter each photo's background separately before pooling — pooling
    # first could let one photo's large white backdrop swamp the real
    # garment-color signal from the others.
    all_pixels = []
    for img in imgs:
        small = img.convert("RGB").resize((64, 64))
        all_pixels.extend(_foreground_pixels(list(small.getdata())))

    buckets = {}
    for r, g, b in all_pixels:
        h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
        key = (round(h * 12), round(s * 4), round(v * 4))
        acc = buckets.setdefault(key, [0, 0, 0, 0])
        acc[0] += r
        acc[1] += g
        acc[2] += b
        acc[3] += 1
    best_key = max(buckets, key=lambda k: buckets[k][3])
    r_sum, g_sum, b_sum, count = buckets[best_key]
    r_avg, g_avg, b_avg = r_sum // count, g_sum // count, b_sum // count
    return f"#{r_avg:02X}{g_avg:02X}{b_avg:02X}"


def _user_instructions_block(extra_notes: Optional[str]) -> str:
    """Formats the user's free-text instructions as a clearly-labeled,
    HIGH-PRIORITY block placed right after the task description (not buried
    at the very end of a long prompt, which is where models pay the least
    attention). Returns "" when there are no notes, so callers can safely
    interpolate it without an extra blank section."""
    notes = (extra_notes or "").strip()
    if not notes:
        return ""
    return (
        "\nUSER INSTRUCTIONS — HIGH PRIORITY, follow these exactly:\n"
        f"\"{notes}\"\n"
        "These instructions must be honored alongside every rule below. If "
        "they conflict with a styling/background/pose choice below, the "
        "user's instructions win. They do NOT override the identity-"
        "preservation or no-hallucination rules — never change the person's "
        "face/identity or invent objects just because a note asks for "
        "something unrelated to that.\n"
    )


_NO_HALLUCINATION_PERSON_RULES = """DO NOT HALLUCINATE — HARD RULES (these override "creativity"). Follow every
rule below exactly; these are the most important instructions in this
prompt:
- #1 MOST CRITICAL RULE — FACE/IDENTITY: the person in the output image
  MUST be the exact same person as in image 1 — same face, same identity,
  recognizable as the same individual by anyone who knows them. Copy the
  face pixel-for-pixel from image 1 wherever it is visible and not covered
  by the garment; do not redraw, beautify, restyle, or regenerate it, even
  partially. This rule overrides every other instruction in this prompt,
  including matching a reference garment's collar/neckline exactly — if
  achieving a perfect collar fit would require touching the face/head at
  all, prioritize keeping the face untouched over the collar's precision.
  Also do not change the person's gender, age, ethnicity, skin tone,
  hairstyle/hair color, body shape/weight/height, or pose — copy all of
  these exactly from the person's photo too. If part of the face is blocked
  (e.g. by a phone during a mirror selfie, or by hair), keep that pose and
  that exact occlusion as-is — do not invent, reveal, or guess a different
  face or pose underneath/behind it.
- Do not invent, add, or remove any object, accessory (glasses, hat, watch,
  jewelry, bag, etc.), logo, tattoo, background element, or extra person
  that is not present in the reference images — unless the user
  instructions above explicitly ask for it.
- Do not change or "fix" anatomy: keep the exact same number of fingers,
  hands, arms, legs, and the same body proportions as the original photo.
  Do not distort, warp, elongate, or duplicate any body part.
- Do not duplicate the person or add a second/third person into the image —
  there must be exactly the same number of people as in the original photo.
- Only change the specific clothing area(s) this task is about (the
  garment/fabric being tried on). Leave every other clothing item, shoes,
  accessory, and body part exactly as they were in the original photo,
  unless the user instructions above explicitly say to change them.
- ANY CAMERA ANGLE IS VALID — the person's photo may be taken from the
  front, side (profile), back, three-quarter, or any other angle, and may
  be a full-body, half-body, or close-up crop. Always apply the garment to
  the person exactly as they are shown. Do NOT rotate, turn, or reframe the
  person into a different angle/viewpoint than the original photo, and do
  NOT refuse or skip clothing them just because the face, front, or another
  body part isn't visible in that angle. If the face or other body parts
  are not visible (e.g. a back-view or side-view photo), do not invent or
  guess what they look like — simply leave them out of frame exactly as in
  the original, and still dress the visible body naturally in the garment.
- Do not change the background/setting, lighting direction, or camera
  framing/zoom unless the style instruction below or the user instructions
  above ask you to.
- If you are ever unsure about any detail, default to matching the
  reference images exactly rather than guessing or embellishing — when in
  doubt, change less, not more.
"""

_NO_HALLUCINATION_AI_MODEL_RULES = """DO NOT HALLUCINATE — HARD RULES (these override "creativity"). Follow every
rule below exactly; these are the most important instructions in this
prompt:
- Do not invent, add, or remove any accessory, logo, pattern, or extra
  garment that is not present in (or a reasonable tailoring of) the fabric
  reference image(s) — unless the user instructions above explicitly ask
  for it.
- Do not add extra people, text, watermarks, or props to the scene — there
  must be exactly one person in the final image.
- Keep anatomy fully realistic and plausible: correct number of fingers,
  hands, arms and legs, natural body proportions, no distorted or duplicated
  body parts.
- If you are ever unsure about a detail (color, pattern, texture), default
  to matching the reference fabric image(s) exactly rather than guessing or
  inventing new details.
"""


_ACCURACY_PREAMBLE = (
    "This is a single, final generation — there is no follow-up correction "
    "round, so every detail must be right on this first attempt. Prioritize "
    "photographic realism and exact fidelity to the reference image(s) over "
    "creative interpretation; this is a precise compositing task, not free "
    "artistic generation."
)


_MULTI_ZONE_GARMENT_TYPES = {
    "sherwani", "kurta pajama", "nehru jacket", "lehenga", "saree fabric", "saree", "kurta",
    "indo western", "bandhgala", "tuxedo",
}


def _is_multi_zone_garment(garment_type: str) -> bool:
    """True for garment types that commonly carry TWO (or more) distinct
    design zones on the same piece — e.g. a sherwani's plain body vs its
    heavily embroidered collar/placket/cuffs/hem border, a tuxedo's satin
    lapel vs its main fabric, or a lehenga's skirt panel vs its contrasting
    border. Matched loosely (substring) so a free-typed type like "Wedding
    Sherwani" still triggers it."""
    gt = garment_type.lower()
    return any(key in gt for key in _MULTI_ZONE_GARMENT_TYPES)


_TOP_ONLY_GARMENT_TYPES = {
    "blazer", "jacket", "sherwani", "indo western", "bandhgala",
    "nehru jacket", "tuxedo",
}


def _is_top_only_garment(garment_type: str) -> bool:
    """True for upper-body-only formal/ethnic garment types (blazer,
    sherwani, bandhgala, tuxedo, etc.) — as opposed to types that already
    describe a full outfit (Kurta Pajama, 3 Piece Suit, Pant Shirt, Lehenga,
    Trousers, ...). Used only in AI-model mode (no uploaded person photo) to
    decide whether the model needs explicit guidance on what bottom to
    generate, since there's no original photo supplying one."""
    gt = garment_type.lower()
    if "pajama" in gt or "pyjama" in gt or "suit" in gt or "pant" in gt:
        return False  # these already name/imply a complete bottom
    return any(key in gt for key in _TOP_ONLY_GARMENT_TYPES)


_TOP_ONLY_BOTTOM_PAIRING_RULE = """- BOTTOM/TROUSERS: the reference image only shows the upper-body garment
  ({garment_type}), so you must also dress the model in well-fitted, simply
  tailored trousers to complete the look — choose a clean, common/neutral
  color that sensibly complements the {garment_type} (e.g. black, charcoal,
  navy, or a matching/complementary tone drawn from the garment's own
  color), with no loud pattern or clashing color. Keep the trousers
  understated so all the attention stays on the {garment_type} itself.
"""


_FABRIC_MATERIAL_RULES = """- FABRIC TYPE/MATERIAL: not just the color and pattern, but the fabric's
  actual MATERIAL must look right in the output — identify what it visually
  is from the reference image and render that same material's authentic
  look, texture and drape:
  * Linen: matte (non-shiny), a visibly slubbed/irregular woven texture,
    natural soft creases/wrinkles in the draped fabric. Never render linen
    as perfectly smooth or glossy.
  * Cotton/poplin/shirting: crisp but matte, an even fine weave, clean
    structured folds.
  * Satin/silk: smooth with a visible glossy sheen and light highlights
    along the folds, a fluid/liquid-like drape.
  * Denim/twill: a visibly diagonal weave texture, sturdier/heavier drape.
  * Wool/tweed: textured, matte, often subtly flecked, a structured
    slightly heavier drape.
  * Velvet: a plush soft-pile texture with direction-dependent sheen
    (lighter/darker patches depending on nap direction).
  * Chiffon/georgette/other sheers: lightweight, semi-translucent, flowing
    drape with fine soft folds.
  * Jacquard/brocade: a raised, woven-in texture, not a flat printed pattern.
  Let the reference image's own texture/sheen/weave tell you the material —
  do not default every fabric to a generic smooth, texture-less look
  regardless of what it actually is.
"""


_MULTI_ZONE_PATTERN_RULES = """- MULTI-ZONE DESIGN — READ CAREFULLY: ethnic wear like this very often has
  TWO OR MORE genuinely different patterns/colors on the SAME piece — for
  example a plain or subtly-textured body with a heavily embroidered/
  contrast-colored collar, placket (front strip), cuffs, and hem border; or
  a solid panel with a separately patterned border (as on a lehenga or saree
  fabric). Look closely at the whole reference image, not just its most
  common color. Reproduce EVERY visually distinct zone exactly as shown —
  its own color, embroidery/print density, and where it sits on the
  garment — rather than blending all zones into one averaged color/pattern
  or only rendering the main body's look. If a border, placket, or cuff
  design is visible anywhere in the reference, it MUST also be visible in
  the same position on the output garment.
"""


def _build_prompt(
    garment_type: str,
    style: str,
    extra_notes: str,
    fabric_hex: Optional[str] = None,
    color_override_name: Optional[str] = None,
    color_override_hex: Optional[str] = None,
    ai_model_gender: Optional[str] = None,
    simple: bool = False,
    garment_image_count: int = 1,
) -> str:
    """Builds a detailed instruction prompt for the image model.

    If ai_model_gender is set, this builds a prompt for generating a fresh
    AI fashion model wearing the fabric (no user photo involved). Otherwise
    it builds the normal "dress this specific person" try-on prompt.

    simple=True builds a much shorter fallback version (no hard-rule block,
    no user extra_notes) used as a one-time retry when the full prompt made
    the model return no image at all (finish_reason other than STOP/OK) —
    a smaller, plainer request is less likely to trigger that.
    """

    style_instructions = {
        "Realistic studio photo": "Render the final image as a clean, realistic studio "
                                   "photograph with soft, even lighting and a neutral backdrop.",
        "Outdoor natural light": "Render the final image outdoors in soft natural daylight.",
        "Fashion catalog shot": "Render the final image like a professional fashion "
                                 "e-commerce catalog photo, sharp focus, flattering pose.",
        "Same background as user photo": "Keep the exact same background, lighting and "
                                          "camera angle as the person's original photo.",
    }
    style_line = style_instructions.get(style, style_instructions["Realistic studio photo"])

    if color_override_hex:
        color_line = (
            f"- Render the {garment_type.lower()} in the color \"{color_override_name}\" "
            f"(approximate hex {color_override_hex}), keeping the fabric's texture/pattern."
        ) if simple else (
            f"- COLOR OVERRIDE: ignore the exact color of the fabric/garment reference image. "
            f"Instead, render the {garment_type.lower()} in the color \"{color_override_name}\" "
            f"(approximate hex {color_override_hex}). Keep the fabric's material, weave/knit "
            f"texture and any pattern style (e.g. stripes/checks) from the reference image, but "
            f"shift its overall hue to this requested color."
        )
    elif _is_multi_zone_garment(garment_type):
        # For garments that often have 2+ distinct design zones (sherwani,
        # lehenga, etc.), forcing the WHOLE garment to one single dominant
        # hex is actively wrong — it would flatten away the contrast
        # border/embroidery. Frame the extracted hex as just the primary/
        # body tone, and defer to the actual reference image for every zone.
        color_line = (
            f"- The reference's overall/primary tone is roughly hex {fabric_hex}, but this "
            f"garment likely has multiple color zones — follow the reference image itself for "
            f"each zone's actual color, not this one hex."
        ) if simple else (
            f"- COLOR GUIDANCE (multi-zone garment): the reference image's overall/primary "
            f"body tone is approximately hex {fabric_hex} — use that as a rough guide for the "
            f"main body only. Do NOT force this single hex onto the entire garment. Instead, "
            f"read each visually distinct zone (body, collar, placket, cuffs, border, etc.) "
            f"directly from the reference image and reproduce ITS OWN actual color and "
            f"embroidery/print there, exactly as shown."
        )
    else:
        color_line = (
            f"- Match the fabric's color (approx hex {fabric_hex}) and pattern as closely as possible."
        ) if simple else (
            f"- COLOR ACCURACY IS CRITICAL: the garment/fabric reference image has an "
            f"approximate dominant color of hex {fabric_hex}. Reproduce this exact color (and "
            f"any pattern — stripes, checks, print — visible in that image) as closely as "
            f"possible. Do NOT invent a different color or pattern; match the fabric exactly, "
            f"the same way every time this fabric is used."
        )

    multi_zone_rules = _MULTI_ZONE_PATTERN_RULES if (_is_multi_zone_garment(garment_type) and not simple) else ""
    # Only relevant in AI-model mode — in person-photo mode the person's own
    # (now background-preserved-by-default) trousers are already in frame.
    bottom_pairing_rule = (
        _TOP_ONLY_BOTTOM_PAIRING_RULE.format(garment_type=garment_type)
        if (ai_model_gender and _is_top_only_garment(garment_type) and not simple)
        else ""
    )

    if ai_model_gender:
        # Text-to-image mode: no person photo, generate a fresh fashion model.
        gender_line = {
            "Male": "a good-looking adult male fashion model",
            "Female": "a good-looking adult female fashion model",
            "Any": "a good-looking adult fashion model",
        }.get(ai_model_gender, "a good-looking adult fashion model")

        if simple:
            return f"""Generate one photorealistic photo of {gender_line} wearing a
well-fitted {garment_type.lower()} made from the fabric shown in the reference
image. Full or three-quarter body shot, well-lit, sharp focus, realistic
anatomy, no text/watermark/extra people.
{color_line}
{style_line}
Return only the final image."""

        garment_block = _describe_image_block(1, garment_image_count, garment_type.lower())
        prompt = f"""You are a professional AI fashion photographer and image generator.
{_ACCURACY_PREAMBLE}

You are given reference image(s) for the garment/fabric: {garment_block}

TASK: Generate ONE brand-new, photorealistic image of {gender_line} wearing a
well-fitted {garment_type.lower()} made from/matching the fabric shown in the
reference image(s). Invent a natural, confident standing pose and a realistic,
attractive face/hairstyle/skin tone appropriate for a professional fashion
catalog — this is a fictional AI-generated model, not a real person.
{_user_instructions_block(extra_notes)}
{_NO_HALLUCINATION_AI_MODEL_RULES}
Strict requirements:
- The garment must be tailored from the exact fabric/material/pattern shown
  in the reference image(s).
- If the reference image is a product photo of an ALREADY-STITCHED,
  readymade garment (e.g. photographed flat, on a hanger, on a mannequin,
  or worn by a different model) rather than a loose fabric swatch, copy
  that exact garment's design — color, print, logo/graphic placement,
  collar/sleeve style, silhouette — onto the fashion model you are
  generating. Completely ignore any mannequin or other model shown in that
  reference photo; only the garment design transfers.
- STRUCTURE/LENGTH/LAYERING — copy these from the reference photo exactly,
  not from a generic mental image of "{garment_type}": its length (cropped/
  waist-length vs knee-length vs ankle-length), whether it is a single piece
  or an open jacket/coat layered OVER a separate visible inner layer (e.g. a
  long embroidered jacket worn open over a plain kurta/shirt underneath —
  if the reference shows two distinct layers like this, the output must
  show both layers too, not merge them into one piece or drop the inner
  layer), and the collar/closure style. "{garment_type}" is only a loose
  category hint; if it ever conflicts with what the reference image
  actually shows, the reference image wins — always copy the real garment
  in front of you, not a generic/default idea of what that category
  usually looks like.
{color_line}
{multi_zone_rules}{bottom_pairing_rule}{_FABRIC_MATERIAL_RULES}- If the fabric has any woven self-pattern, jacquard, herringbone, subtle
  texture, or weave visible in the reference (even on a mostly one-color
  fabric), reproduce that texture in the rendered garment — do not flatten
  it into a plain, texture-less solid color.
- Full body or three-quarter body shot, well-lit, sharp focus, realistic
  proportions and anatomy.
- Do not add any text, watermark, logo, or extra people to the image.
- {style_line}
- This is an ordinary fashion/e-commerce catalog visualization — the model
  is fully and modestly clothed.

Output: return only the final generated photo image (no explanation needed).
"""
        return prompt

    if simple:
        return f"""Image 1 is a person. Image 2 is a {garment_type.lower()} fabric/garment.
Generate one photorealistic photo of the SAME person from image 1, now
wearing the {garment_type.lower()} from image 2, keeping their face, identity,
pose, body and background unchanged. Adapt the garment naturally to their
body with realistic folds and shadows.
{color_line}
{style_line}
Return only the final image, no text."""

    garment_block = _describe_image_block(2, garment_image_count, garment_type.lower())
    prompt = f"""You are a professional virtual try-on / fashion image compositing engine.
{_ACCURACY_PREAMBLE}

You are given reference images in this order:
1. Image 1 is a photo of a real person.
2. {garment_block}

TASK: Generate ONE new photorealistic image of the SAME person from image 1,
now wearing the {garment_type.lower()} shown in the reference image(s) above.
{_user_instructions_block(extra_notes)}
{_NO_HALLUCINATION_PERSON_RULES}
Strict requirements:
- Preserve the person's face, identity, skin tone, hair, body shape and pose exactly.
  If part of the face is blocked (e.g. by a phone during a mirror selfie), keep
  that pose as-is — do not invent a different face or pose.
- The photo may be from ANY camera angle (front, side/profile, back,
  three-quarter, close-up, etc.) — keep that exact same angle/viewpoint in
  the output. Do not rotate the person to face the camera if they weren't,
  and dress them fully in the garment no matter which angle is shown.
- Replace only the relevant clothing area with the garment/fabric shown
  above, adapting it naturally to the person's body with realistic folds,
  shadows, and fit (as if they are actually wearing it). This must be a
  REAL, VISIBLE structural change to that clothing area — not a subtle tint
  or a near-copy of the original clothing.
- If the reference is a fabric/textile swatch rather than a stitched
  garment, tailor it into a well-fitted {garment_type.lower()} using that
  fabric's pattern and texture.
- If the reference image is instead a product photo of an ALREADY-STITCHED,
  readymade garment (e.g. a shirt/t-shirt photographed flat, on a hanger,
  on a mannequin, or being worn by a different model), extract ONLY that
  garment's exact design — its color, print, logo/graphic placement, collar/
  sleeve style, and silhouette — and put THAT SAME garment on the person
  from image 1. Completely ignore any mannequin, hanger, or other model
  shown in that reference photo — none of their body, face, pose, or skin
  tone should influence the output in any way; only the garment itself
  transfers. This is a REQUIRED, highly visible change — the person's
  current top must clearly become this exact garment's design, not stay
  close to what they were already wearing.
- STRUCTURE/LENGTH/LAYERING — copy these from the reference photo exactly,
  not from a generic mental image of "{garment_type}": its length (cropped/
  waist-length vs knee-length vs ankle-length), whether it is a single piece
  or an open jacket/coat layered OVER a separate visible inner layer (e.g. a
  long embroidered jacket worn open over a plain kurta/shirt underneath —
  if the reference shows two distinct layers like this, the output must
  show both layers too, not merge them into one piece or drop the inner
  layer), and the collar/closure style. "{garment_type}" is only a loose
  category hint; if it ever conflicts with what the reference image
  actually shows, the reference image wins — always copy the real garment
  in front of you, not a generic/default idea of what that category
  usually looks like.
{color_line}
{multi_zone_rules}{_FABRIC_MATERIAL_RULES}- If the fabric has any woven self-pattern, jacquard, herringbone, subtle
  texture, or weave visible in the reference (even on a mostly one-color
  fabric), reproduce that texture in the rendered garment — do not flatten
  it into a plain, texture-less solid color.
- Match lighting and shadows between the garment and the rest of the body so
  the result looks like a single real photograph, not a collage.
- FRAMING: keep the exact same framing as the person's original photo — do
  not zoom in, crop, or cut off the head, feet, hands or edges of the body.
  The full pose/body extent visible in image 1 must remain fully visible
  in the output.
- Do not add any text, watermark, logo, or extra people to the image.
- {style_line}
- This is an ordinary fashion/e-commerce try-on visualization request — the
  person is fully clothed in both the input and output images.
- One final reminder since it matters more than anything else here: the
  face must be copied exactly from image 1, the same real individual, not
  a similar-looking generated face. Always produce the image with image 1's
  real face on it.

Output: return only the final composited photo image (no explanation needed).
"""
    return prompt


def _describe_image_block(start_idx: int, count: int, part_name: str) -> str:
    """Builds the sentence describing which numbered image(s) belong to one
    garment (shirt or trouser). When more than one photo was provided for
    that garment, makes it explicit that ALL of them show the SAME fabric
    from different angles/close-ups — not different garments — so the model
    combines them into one accurate read of the material instead of getting
    confused about how many garments are involved."""
    if count <= 1:
        return f"Image {start_idx} is the {part_name} fabric or garment."
    end_idx = start_idx + count - 1
    return (
        f"Images {start_idx}-{end_idx} are MULTIPLE PHOTOS OF THE SAME "
        f"{part_name.upper()} fabric/garment — different angles, distances, "
        f"or lighting of ONE single item (e.g. a wide shot plus a close-up of "
        f"the weave/print, or front and back of the same piece). They are NOT "
        f"separate garments. Cross-reference all of them together to pin down "
        f"this fabric's true color, weave/knit texture, and pattern as "
        f"accurately as possible before rendering it."
    )


def _finish_reason_str(candidate) -> str:
    reason = getattr(candidate, "finish_reason", None)
    return str(reason) if reason is not None else "UNKNOWN"


# finish_reason values that mean "a real image came back and everything is
# fine" (or we simply couldn't tell, e.g. an older SDK not exposing it) — any
# OTHER value (IMAGE_OTHER, SAFETY, RECITATION, etc.) means the model chose
# not to produce an image, which is retriable with a simplified request.
_OK_FINISH_REASONS = ("STOP", "1", "FINISH_REASON_STOP", "UNKNOWN")


def _extract_image_and_text(response):
    """Pulls the generated image (if any) + any text out of a Gemini
    response's first candidate. Shared by generate_tryon and
    generate_full_outfit_tryon so both use identical extraction logic."""
    candidate = response.candidates[0]
    finish_reason = _finish_reason_str(candidate)

    result_image = None
    result_text = None
    content = getattr(candidate, "content", None)
    parts = getattr(content, "parts", None) or []
    for part in parts:
        inline_data = getattr(part, "inline_data", None)
        if inline_data is not None and getattr(inline_data, "data", None):
            try:
                result_image = Image.open(io.BytesIO(inline_data.data)).convert("RGB")
            except Exception as e:  # noqa: BLE001
                raise GeminiTryOnError(f"Received an unreadable image from Gemini: {e}") from e
        text = getattr(part, "text", None)
        if text:
            result_text = (result_text + "\n" + text) if result_text else text

    return result_image, result_text, finish_reason


def generate_tryon(
    api_key: str,
    garment_image,
    garment_type: str,
    person_image: Optional[Image.Image] = None,
    ai_model_gender: Optional[str] = None,
    style: str = "Realistic studio photo",
    extra_notes: str = "",
    model_name: str = "gemini-2.5-flash-image",
    color_override_name: Optional[str] = None,
    color_override_hex: Optional[str] = None,
    temperature: Optional[float] = None,
) -> Tuple[Image.Image, Optional[str]]:
    """
    Calls the Gemini image model to produce ONE try-on image from ONE person
    photo + one garment/fabric (possibly shown via several reference photos).

    garment_image accepts EITHER a single PIL Image OR a list of PIL Images —
    passing several photos of the SAME fabric (different angles, a close-up
    of the weave, front/back) lets the model cross-reference them for a more
    accurate color/pattern/texture read instead of guessing from one shot.

    Two modes:
    - person_image is provided -> dress THAT specific person in the garment.
    - person_image is None (ai_model_gender set instead) -> generate a fresh
      AI fashion model wearing the garment, no user photo needed.

    When the app has multiple person photos and/or multiple garment photos,
    it calls this function once per (person, garment) pair it needs to
    generate — batching/pairing across multiple photos is handled in app.py,
    not here, so each call always deals with exactly one image on each side
    (each "side" possibly being several angle-photos of one fabric).

    Raises GeminiTryOnError on failure (missing key, blocked content, no image
    returned, network/API errors, etc). The error message is written to be
    shown directly to the end user.
    """
    if person_image is None and not ai_model_gender:
        raise GeminiTryOnError(
            "Internal error: either a person photo or an AI-model gender must be provided."
        )

    # Normalize garment_image to a non-empty list of PIL Images so callers
    # can pass either a single image (most call sites) or a list of
    # multi-angle photos of the same garment (used by the outfit chaining
    # in generate_full_outfit_tryon).
    garment_images = (
        [garment_image] if isinstance(garment_image, Image.Image) else list(garment_image or [])
    )
    if not garment_images:
        raise GeminiTryOnError("Please provide at least one garment/fabric photo.")

    try:
        from google import genai
        from google.genai import types
    except ImportError as e:  # pragma: no cover
        raise GeminiTryOnError(
            "The 'google-genai' package is not installed on the server. "
            "Run: pip install google-genai"
        ) from e

    if not api_key:
        raise GeminiTryOnError("Missing Gemini API key on the server.")

    try:
        client = _get_client(api_key)
    except Exception as e:  # noqa: BLE001
        raise GeminiTryOnError(f"Could not initialize Gemini client: {e}") from e

    # Keep payloads small & fast, and avoid some upstream size-limit failures.
    if person_image is not None:
        person_image = _resize_for_api(person_image)
    garment_images = [_resize_for_api(img) for img in garment_images]

    fabric_hex = dominant_color_hex_multi(garment_images)
    prompt = _build_prompt(
        garment_type,
        style,
        extra_notes,
        fabric_hex=fabric_hex,
        color_override_name=color_override_name,
        color_override_hex=color_override_hex,
        ai_model_gender=None if person_image is not None else (ai_model_gender or "Any"),
        garment_image_count=len(garment_images),
    )

    if person_image is not None:
        contents = [prompt, person_image, *garment_images]
    else:
        contents = [prompt, *garment_images]

    # Use permissive-but-safe thresholds so ordinary clothed photos aren't
    # falsely flagged by the safety filters (a common cause of "no image
    # returned" for mirror selfies / bare arms / swimwear-adjacent garments).
    try:
        safety_settings = [
            types.SafetySetting(category=cat, threshold="BLOCK_ONLY_HIGH")
            for cat in (
                "HARM_CATEGORY_HARASSMENT",
                "HARM_CATEGORY_HATE_SPEECH",
                "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "HARM_CATEGORY_DANGEROUS_CONTENT",
            )
        ]
    except Exception:  # noqa: BLE001
        safety_settings = None

    config_kwargs = {
        "response_modalities": ["Text", "Image"],
        "temperature": temperature if temperature is not None else GENERATION_TEMPERATURE,
    }
    if safety_settings:
        config_kwargs["safety_settings"] = safety_settings

    try:
        aspect_ratio = _closest_aspect_ratio(person_image) if person_image is not None else "3:4"
        config_kwargs["image_config"] = types.ImageConfig(aspect_ratio=aspect_ratio)
    except Exception:  # noqa: BLE001 — older SDK without ImageConfig support
        pass

    def _build_config():
        try:
            return types.GenerateContentConfig(**config_kwargs)
        except Exception:  # noqa: BLE001 — some SDK/model combos reject certain fields
            fallback = dict(config_kwargs)
            fallback.pop("image_config", None)
            try:
                return types.GenerateContentConfig(**fallback)
            except Exception:  # noqa: BLE001
                return types.GenerateContentConfig(response_modalities=["Text", "Image"])

    response = None
    last_error: Optional[Exception] = None

    for attempt in range(MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=_build_config(),
            )
            last_error = None
            break
        except Exception as e:  # noqa: BLE001
            last_error = e
            msg = str(e)
            is_rate_limit = "RESOURCE_EXHAUSTED" in msg or "429" in msg or "rate limit" in msg.lower()
            is_transient = "500" in msg or "503" in msg or "UNAVAILABLE" in msg or "DEADLINE_EXCEEDED" in msg
            if (is_rate_limit or is_transient) and attempt < MAX_RETRIES:
                # Jittered backoff: each caller waits a slightly different
                # amount of time, so if several shops get rate-limited at
                # the same moment they don't all retry in the same instant
                # and re-trigger the same rate limit again (thundering herd).
                wait = RETRY_WAIT_SECONDS * (attempt + 1) + random.uniform(0, 4)
                logger.warning(
                    "Gemini call failed (attempt %d/%d): %s — retrying in %.1fs",
                    attempt + 1, MAX_RETRIES + 1, msg, wait,
                )
                time.sleep(wait)
                continue
            break

    if response is None:
        msg = str(last_error) if last_error else "Unknown error"
        logger.error("Gemini API call failed: %s", msg)
        if "API_KEY_INVALID" in msg or "API key not valid" in msg or "PERMISSION_DENIED" in msg or "403" in msg:
            raise GeminiTryOnError(
                "The Gemini API key was rejected by Google. Double-check it's correct, "
                "active, and has the Generative Language API enabled."
            ) from last_error
        if "RESOURCE_EXHAUSTED" in msg or "429" in msg:
            raise GeminiTryOnError(
                "You've hit the Gemini free-tier rate limit (it allows only a few requests "
                "per minute). Please wait about 30-60 seconds before generating again."
            ) from last_error
        if "not found" in msg.lower() or "404" in msg:
            raise GeminiTryOnError(
                f"Model '{model_name}' isn't available for this API key/region. "
                "Try setting GEMINI_MODEL=gemini-2.5-flash-image-preview in your .env."
            ) from last_error
        raise GeminiTryOnError(f"Gemini API request failed: {msg}") from last_error

    if not getattr(response, "candidates", None):
        feedback = getattr(response, "prompt_feedback", None)
        block_reason = getattr(feedback, "block_reason", None) if feedback else None
        if block_reason:
            raise GeminiTryOnError(
                f"Your request was blocked by Gemini's safety filters (reason: {block_reason}). "
                "Try a different, clearly non-sensitive photo."
            )
        raise GeminiTryOnError(
            "The model returned no result. Please try again with clearer photos."
        )

    result_image, result_text, finish_reason = _extract_image_and_text(response)

    if result_image is None and finish_reason.upper() not in _OK_FINISH_REASONS:
        # The model chose not to produce an image (e.g. IMAGE_OTHER) rather
        # than a hard safety block. This is often caused by an overloaded
        # prompt/instruction set rather than the photo itself, so retry ONCE
        # with a much shorter, simpler prompt before giving up.
        logger.warning(
            "generate_tryon: no image (reason=%s) — retrying with a simplified prompt",
            finish_reason,
        )
        try:
            simple_prompt = _build_prompt(
                garment_type, style, "", fabric_hex=fabric_hex,
                color_override_name=color_override_name,
                color_override_hex=color_override_hex,
                ai_model_gender=None if person_image is not None else (ai_model_gender or "Any"),
                simple=True,
            )
            simple_contents = (
                [simple_prompt, person_image, garment_images[0]]
                if person_image is not None
                else [simple_prompt, garment_images[0]]
            )
            fallback_response = client.models.generate_content(
                model=model_name, contents=simple_contents, config=_build_config(),
            )
            fb_image, fb_text, fb_reason = _extract_image_and_text(fallback_response)
            if fb_image is not None:
                return fb_image, fb_text
            finish_reason = fb_reason
        except Exception as e:  # noqa: BLE001
            logger.warning("generate_tryon: fallback retry also failed: %s", e)

    if result_image is None:
        # No image came back — this is almost always either a safety refusal
        # or the model choosing to only reply with text. Surface *why*.
        if finish_reason.upper() not in _OK_FINISH_REASONS:
            raise GeminiTryOnError(
                f"Generation was stopped before producing an image (reason: {finish_reason}). "
                "Try a different photo, a different garment image, or simplify your "
                "extra instructions."
            )
        if result_text:
            raise GeminiTryOnError(
                f"The model responded with text instead of an image: \u201c{result_text.strip()[:300]}\u201d "
                "— try a clearer/simpler photo pair."
            )
        raise GeminiTryOnError(
            "No image was returned by the model. Try again, or use clearer photos "
            "(good lighting, single visible person, single visible garment)."
        )

    return result_image, result_text


def generate_full_outfit_tryon(
    api_key: str,
    shirt_image,
    trouser_image,
    person_image: Optional[Image.Image] = None,
    ai_model_gender: Optional[str] = None,
    style: str = "Realistic studio photo",
    extra_notes: str = "",
    model_name: str = "gemini-2.5-flash-image",
    shirt_color_override_name: Optional[str] = None,
    shirt_color_override_hex: Optional[str] = None,
    trouser_color_override_name: Optional[str] = None,
    trouser_color_override_hex: Optional[str] = None,
    shirt_garment_type: str = "Shirt/top",
) -> Tuple[Image.Image, Optional[str]]:
    """
    "Full Outfit" mode: dresses the person/AI model in a shirt fabric AND a
    trouser fabric.

    IMPLEMENTATION NOTE (important): this used to be ONE Gemini call editing
    both the shirt and the trousers at the same time. In testing that was
    unreliable — asking an image-editing model to make two independent,
    large structural edits (swap the top AND swap the bottom) in a single
    pass meant it would often render one garment (usually the shirt) while
    barely touching the other (almost always the original trousers were
    left completely unchanged), no matter how strongly the prompt insisted
    otherwise.

    This is now implemented as TWO chained single-garment calls instead,
    reusing generate_tryon (the same well-tested single-garment code path):
        1. Apply the shirt fabric to the person / generate the AI model.
        2. Apply the trouser fabric to the RESULT of step 1, treating it as
           an ordinary photo — exactly the same code path used for any real
           uploaded photo.
    Each step only has to make ONE structural change, which the model
    handles far more reliably — at the cost of roughly double the
    generation time, since step 2 depends on step 1's output and the two
    calls cannot be run in parallel.

    shirt_image / trouser_image each accept EITHER a single PIL Image OR a
    list of PIL Images — passing several photos of the SAME fabric (e.g.
    different angles, a close-up of the weave, front and back of the piece)
    lets the model cross-reference them for a more accurate color/pattern/
    texture read instead of guessing from one shot.

    Same two modes as generate_tryon:
    - person_image is provided -> dress THAT specific person in the outfit.
    - person_image is None (ai_model_gender set instead) -> generate a fresh
      AI fashion model wearing the outfit, no user photo needed.

    Raises GeminiTryOnError on failure, with the same user-facing error
    messages as generate_tryon (raised from whichever of the two steps hit
    the problem).
    """
    if person_image is None and not ai_model_gender:
        raise GeminiTryOnError(
            "Internal error: either a person photo or an AI-model gender must be provided."
        )

    shirt_images = [shirt_image] if isinstance(shirt_image, Image.Image) else list(shirt_image or [])
    trouser_images = [trouser_image] if isinstance(trouser_image, Image.Image) else list(trouser_image or [])
    if not shirt_images or not trouser_images:
        raise GeminiTryOnError(
            "Please provide at least one shirt fabric photo and one trouser fabric photo."
        )

    # --- Step 1: shirt/top (or sherwani/bandhgala/indo western/etc. — the
    # top half can be any garment type, not just a plain shirt; this is what
    # shirt_garment_type picks, same options as Single Garment mode) --------
    # Either dress the real person, or (no person photo) generate a fresh AI
    # model wearing it — exactly generate_tryon's two normal modes.
    step1_image, step1_text = generate_tryon(
        api_key=api_key,
        garment_image=shirt_images,
        garment_type=shirt_garment_type,
        person_image=person_image,
        ai_model_gender=ai_model_gender,
        style=style,
        extra_notes=extra_notes,
        model_name=model_name,
        color_override_name=shirt_color_override_name,
        color_override_hex=shirt_color_override_hex,
    )

    # --- Step 2: trousers, applied on top of step 1's output ---------------
    # From here on we always have a concrete photo to edit (either the
    # user's own photo now wearing the new shirt, or the freshly generated
    # AI model) — so this step always runs in "dress this specific photo"
    # mode. We force the style that preserves step 1's framing/background/
    # lighting exactly, since that IS the final look we want; step 2 should
    # only change the trousers, nothing else about the image.
    #
    # A slightly higher temperature is used here specifically because the
    # single most common failure was the model being too conservative to
    # fully replace a highly recognizable original garment (e.g. blue denim
    # jeans) — a little more sampling freedom makes it commit to the
    # structural change instead of leaving the original mostly intact.
    step2_image, step2_text = generate_tryon(
        api_key=api_key,
        garment_image=trouser_images,
        garment_type="Trousers",
        person_image=step1_image,
        ai_model_gender=None,
        style="Same background as user photo",
        extra_notes=extra_notes,
        model_name=model_name,
        color_override_name=trouser_color_override_name,
        color_override_hex=trouser_color_override_hex,
        temperature=min(GENERATION_TEMPERATURE + 0.15, 0.4),
    )

    combined_text = None
    for t in (step1_text, step2_text):
        if t:
            combined_text = (combined_text + "\n" + t) if combined_text else t

    return step2_image, combined_text



