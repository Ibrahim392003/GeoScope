"""GeoScope — Explainable Sentinel-2 Land-Use Classification."""
import io
import time
from pathlib import Path

import matplotlib
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

matplotlib.use("Agg")
from matplotlib import cm  # noqa: E402

CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
]
DISPLAY_NAMES = {
    "AnnualCrop": "Annual Crop",
    "Forest": "Forest",
    "HerbaceousVegetation": "Herbaceous Vegetation",
    "Highway": "Highway",
    "Industrial": "Industrial",
    "Pasture": "Pasture",
    "PermanentCrop": "Permanent Crop",
    "Residential": "Residential",
    "River": "River",
    "SeaLake": "Sea & Lake",
}
ICONS = {
    "AnnualCrop": "🌾", "Forest": "🌲", "HerbaceousVegetation": "🌿",
    "Highway": "🛣️", "Industrial": "🏭", "Pasture": "🐄",
    "PermanentCrop": "🍇", "Residential": "🏘️", "River": "🏞️", "SeaLake": "🌊",
}
DESCRIPTIONS = {
    "AnnualCrop": "Cultivated agricultural fields planted and harvested annually.",
    "Forest": "Dense tree canopy, woodland, and natural timber cover.",
    "HerbaceousVegetation": "Natural grasslands, shrubs, and wild low vegetation.",
    "Highway": "Major multi-lane motorways, interchanges, and arterial corridors.",
    "Industrial": "Factories, warehouses, and large commercial facilities.",
    "Pasture": "Managed grazing meadows and open agricultural grassland.",
    "PermanentCrop": "Orchards, vineyards, and perennial tree-crop plantations.",
    "Residential": "Suburban housing grids, rooftops, and neighborhood streets.",
    "River": "Flowing rivers, streams, and engineered waterways.",
    "SeaLake": "Open coastal seas, reservoirs, and large freshwater lakes.",
}
VISUAL_CUES = {
    "AnnualCrop": "rectangular plot boundaries and uniform seasonal crop rows",
    "Forest": "dense, textured tree crowns and continuous canopy cover",
    "HerbaceousVegetation": "irregular patches of wild grass and low scrub",
    "Highway": "linear paved corridors and road interchange geometry",
    "Industrial": "large geometric warehouse rooftops and concrete yards",
    "Pasture": "smooth grazing fields with soft natural borders",
    "PermanentCrop": "regularly spaced orchard trees or vineyard rows",
    "Residential": "clustered house rooftops and neighborhood street grids",
    "River": "winding water channels and distinct shoreline banks",
    "SeaLake": "expansive, uniform open water surfaces",
}

CKPT_DIR = Path("models/checkpoints")
MODELS = {
    "EfficientNet-B0": (
        "efficientnet", "efficientnet_best.pth", "98.64%", "1.08 ms", "4.0M", "16.4 MB",
        "Recommended", "Best overall balance of high accuracy (98.64%) and lightweight speed.",
    ),
    "ResNet-18": (
        "resnet18", "resnet18_best.pth", "98.37%", "0.85 ms", "11.2M", "44.8 MB",
        "Fastest", "Lowest latency (0.85 ms/img) using a classic residual CNN.",
    ),
    "ViT-B/16": (
        "vit", "vit_best.pth", "98.84%", "10.74 ms", "85.8M", "343.3 MB",
        "Most Accurate", "Highest test accuracy (98.84%) using global self-attention.",
    ),
}
DRIVE_FOLDER = "https://drive.google.com/drive/folders/1t420bNansMSE7BrX4Qnnc8burWkLmg0m"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Every displayed image is rendered at this size; CSS scales it to the column.
DISPLAY = 448

TFM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ----------------------------------------------------------------- Models
def _extract_state(ckpt):
    if isinstance(ckpt, dict):
        for k in ("model_state_dict", "state_dict", "model"):
            if k in ckpt and isinstance(ckpt[k], dict):
                ckpt = ckpt[k]
                break
    out = {}
    for k, v in ckpt.items():
        for p in ("module.", "model.", "backbone."):
            if k.startswith(p):
                k = k[len(p):]
        out[k] = v
    return out


def build_model(key, state):
    if key == "resnet18":
        m = models.resnet18(weights=None)
        m.fc = (nn.Sequential(nn.Dropout(), nn.Linear(512, 10))
                if "fc.1.weight" in state else nn.Linear(512, 10))
    elif key == "efficientnet":
        m = models.efficientnet_b0(weights=None)
        m.classifier[1] = nn.Linear(1280, 10)
    else:
        m = models.vit_b_16(weights=None)
        m.heads.head = (nn.Sequential(nn.Dropout(), nn.Linear(768, 10))
                        if "heads.head.1.weight" in state else nn.Linear(768, 10))
    m.load_state_dict(state)
    return m.to(DEVICE).eval()


def ensure_checkpoints():
    try:
        import gdown
        CKPT_DIR.mkdir(parents=True, exist_ok=True)
        gdown.download_folder(DRIVE_FOLDER, output=str(CKPT_DIR), quiet=True)
    except Exception as e:  # noqa: BLE001
        st.warning(f"Could not auto-download checkpoints: {e}")


@st.cache_resource(show_spinner="Loading model...")
def load_model(name):
    key, fname = MODELS[name][:2]
    path = CKPT_DIR / fname
    if not path.exists():
        ensure_checkpoints()
    if not path.exists():
        return None
    ckpt = torch.load(path, map_location=DEVICE)
    return build_model(key, _extract_state(ckpt))


# --------------------------------------------------------------- Grad-CAM
def grad_cam(model, key, x):
    store = {}
    layer = (
        model.layer4 if key == "resnet18" else
        model.features[8] if key == "efficientnet" else
        model.encoder.layers[-1].ln_1
    )

    def fwd(_, __, out):
        store["a"] = out
        out.register_hook(lambda g: store.__setitem__("g", g))

    h = layer.register_forward_hook(fwd)
    try:
        model.zero_grad()
        logits = model(x)
        probs = F.softmax(logits, 1)[0].detach().cpu().numpy()
        idx = int(logits.argmax(1))
        logits[0, idx].backward()
    finally:
        h.remove()

    a, g = store["a"], store["g"]
    if key == "vit":
        a = a[:, 1:, :].reshape(1, 14, 14, -1).permute(0, 3, 1, 2)
        g = g[:, 1:, :].reshape(1, 14, 14, -1).permute(0, 3, 1, 2)
    w = g.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * a).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)[0, 0]
    cam = cam.detach().cpu().numpy()
    # Percentile clip: one hot patch (common in ViT) can't wash out the rest
    cam = np.clip(cam / (np.percentile(cam, 99) + 1e-8), 0, 1)
    return probs, idx, cam


def upscale(img, size=DISPLAY):
    return img.resize((size, size), Image.LANCZOS)


def render_heatmap(img, cam, alpha, pure=False, size=DISPLAY):
    base = np.asarray(upscale(img, size), dtype=np.float32) / 255.0
    # Smoothly upsample the CAM BEFORE colorizing so it isn't blocky
    cam_big = Image.fromarray((cam * 255).astype(np.uint8)).resize((size, size), Image.BICUBIC)
    cam_big = np.asarray(cam_big, dtype=np.float32) / 255.0
    # Thermal colormap (matches the UI palette); swap to cm.jet for the classic look
    heat = cm.inferno(cam_big)[..., :3]
    arr = heat if pure else np.clip((1 - alpha) * base + alpha * heat, 0, 1)
    return Image.fromarray((arr * 255).astype(np.uint8))


def show_img(image, caption=None):
    """Fill the column width; works on old and new Streamlit versions."""
    try:
        st.image(image, caption=caption, width="stretch")
    except Exception:  # noqa: BLE001
        st.image(image, caption=caption, use_container_width=True)


def image_to_bytes(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


def predict(name, img):
    model = load_model(name)
    if model is None:
        return None
    x = TFM(img).unsqueeze(0).to(DEVICE)
    t = time.perf_counter()
    probs, idx, cam = grad_cam(model, MODELS[name][0], x)
    return dict(probs=probs, idx=idx, cam=cam, ms=(time.perf_counter() - t) * 1000)


# -------------------------------------------------------------------- UI
st.set_page_config(
    page_title="GeoScope",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;700;800&family=Unbounded:wght@600;800&display=swap');

    /* ------------------------------------------------------------------
       GeoScope palette: indigo night + thermal orange
       Deep Space #0E1238  page        Panel   #161C4A  cards
       Edge       #2A3270  borders     Text    #F1F3FF  /  Muted #A3ABD8
       Periwinkle #8EA2FF  cool accent
       Thermal    #FF7A2F  the pop (same hue family as the heatmap's hot zones)
       Solar      #FFC14D  thermal highlight
       ------------------------------------------------------------------ */

    [data-testid="stSidebar"], [data-testid="collapsedControl"] { display: none !important; }

    .stApp, [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(900px 520px at 6% -8%, rgba(142,162,255,.20), transparent 60%),
            radial-gradient(800px 520px at 100% 0%, rgba(255,122,47,.10), transparent 60%),
            #0E1238 !important;
        color: #F1F3FF !important;
        font-family: 'Plus Jakarta Sans', system-ui, sans-serif;
    }
    [data-testid="stHeader"] { background: transparent !important; }
    .block-container { padding-top: 1.6rem !important; padding-bottom: 3rem !important; max-width: 1180px; }

    /* ---- Nav + hero ---- */
    .nav { display: flex; align-items: center; justify-content: space-between; margin-bottom: 1rem; }
    .brand { display: flex; align-items: center; gap: 10px;
             font-family: 'Unbounded', sans-serif; font-weight: 800; font-size: 1.05rem; color: #F1F3FF; }
    .nav-tags { display: flex; gap: 8px; }
    .tag { padding: 5px 13px; border-radius: 999px; font-size: .78rem; font-weight: 700;
           color: #A3ABD8; border: 1px solid #2A3270; }

    .hero {
        padding: 3.2rem 2.8rem; margin-bottom: 1.6rem;
        border: 1px solid #2A3270; border-radius: 28px; overflow: hidden;
        background:
            repeating-radial-gradient(circle at 88% 50%, transparent 0 20px, rgba(255,255,255,.065) 20px 21px),
            radial-gradient(circle at 88% 50%, rgba(255,193,77,.95) 0, rgba(255,122,47,.80) 46px,
                            rgba(181,55,155,.50) 130px, transparent 270px),
            #161C4A;
    }
    .hero-title {
        font-family: 'Unbounded', sans-serif; font-weight: 800;
        font-size: clamp(1.8rem, 3.6vw, 2.9rem); line-height: 1.12; letter-spacing: -0.03em;
        max-width: 22ch; color: #F1F3FF;
    }
    .hero-sub { max-width: 34rem; margin-top: 1rem; font-size: 1.02rem; line-height: 1.6; color: #C9CEEF; }

    /* ---- Segmented tabs ---- */
    [data-baseweb="tab-list"] {
        width: fit-content; gap: 4px; padding: 5px;
        background: #161C4A; border: 1px solid #2A3270; border-radius: 999px;
    }
    button[data-baseweb="tab"] { border-radius: 999px; padding: 8px 20px; height: auto; }
    button[data-baseweb="tab"] p { color: #A3ABD8; font-weight: 700; }
    button[data-baseweb="tab"][aria-selected="true"] { background: #FF7A2F; }
    button[data-baseweb="tab"][aria-selected="true"] p { color: #0E1238; }
    [data-baseweb="tab-highlight"], [data-baseweb="tab-border"] { display: none !important; }
    [data-baseweb="tab-panel"] { padding-top: 1.2rem; }

    /* ---- Cards ---- */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: #161C4A !important;
        border: 1px solid #2A3270 !important;
        border-radius: 22px !important;
        box-shadow: none !important;
    }
    .card-title { font-weight: 800; font-size: 1.02rem; margin-bottom: .3rem; color: #F1F3FF; }
    .hint { font-size: .88rem; color: #A3ABD8; margin-left: 6px; }
    .hint b { color: #F1F3FF; }

    /* ---- Pills ---- */
    .pill-pop, .pill-peri, .pill-light {
        padding: 4px 12px; border-radius: 999px; font-size: .78rem; font-weight: 700; display: inline-block;
    }
    .pill-pop   { background: rgba(255,122,47,.16); color: #FFA566; border: 1px solid rgba(255,122,47,.5); }
    .pill-peri  { background: rgba(142,162,255,.14); color: #B4C0FF; border: 1px solid rgba(142,162,255,.45); }
    .pill-light { background: #F1F3FF; color: #0E1238; }

    /* ---- Result head with confidence ring ---- */
    .result-head { display: flex; align-items: center; justify-content: space-between; gap: 1.2rem; margin-bottom: 1rem; }
    .result-class {
        font-family: 'Unbounded', sans-serif; font-weight: 800;
        font-size: clamp(1.3rem, 2.4vw, 1.9rem); line-height: 1.15; letter-spacing: -0.02em; color: #F1F3FF;
    }
    .result-desc { margin: .5rem 0 .8rem; color: #A3ABD8; font-size: .92rem; line-height: 1.5; }
    .ring {
        position: relative; flex: none; width: 104px; height: 104px; border-radius: 50%;
        display: grid; place-items: center;
        background: conic-gradient(#FF7A2F calc(var(--p) * 1%), #2A3270 0);
    }
    .ring::before { content: ""; position: absolute; inset: 10px; border-radius: 50%; background: #161C4A; }
    .ring span { position: relative; font-family: 'Unbounded', sans-serif; font-weight: 700; font-size: .98rem; }

    .analysis-callout {
        background: rgba(142,162,255,.08); border: 1px solid #2A3270; border-left: 4px solid #8EA2FF;
        border-radius: 14px; padding: .95rem 1.15rem; margin: .4rem 0 1rem;
        font-size: .92rem; line-height: 1.6; color: #DDE1FA;
    }
    .analysis-callout b { color: #FFFFFF; }

    /* ---- Dropzone ---- */
    [data-testid="stFileUploader"] section {
        background: #121846 !important; border: 1.5px dashed #3A4590 !important; border-radius: 16px !important;
    }
    [data-testid="stFileUploader"] section:hover { border-color: #FF7A2F !important; background: rgba(255,122,47,.07) !important; }

    /* ---- Images: fill the column, equal squares, no overflow ---- */
    [data-testid="stImage"] { width: 100% !important; }
    [data-testid="stImage"] img {
        border-radius: 14px; border: 1px solid #2A3270;
        width: 100% !important; height: auto !important;
        aspect-ratio: 1 / 1; object-fit: cover;
    }

    /* ---- Buttons ---- */
    [data-testid="stDownloadButton"] button {
        background: #FF7A2F; border: none; border-radius: 12px; font-weight: 800;
        transition: background .15s ease;
    }
    [data-testid="stDownloadButton"] button p { color: #0E1238; }
    [data-testid="stDownloadButton"] button:hover { background: #FFC14D; }
    button:focus-visible, [data-baseweb="tab"]:focus-visible { outline: 2px solid #FFC14D !important; outline-offset: 2px; }

    /* ---- Probability bars ---- */
    .bar-row { display: flex; align-items: center; margin: 10px 0; font-size: .88rem; color: #F1F3FF; }
    .bar-label { width: 185px; font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .bar-track { flex: 1; background: #232B63; border-radius: 999px; height: 10px; overflow: hidden; margin: 0 12px; }
    .bar-fill  { height: 10px; border-radius: 999px; }
    .bar-val   { width: 54px; text-align: right; font-variant-numeric: tabular-nums; font-weight: 800; }

    /* ---- Model tiles ---- */
    .model-card { background: #161C4A; border: 1px solid #2A3270; border-radius: 24px; padding: 1.6rem; }
    .model-card.rec { border-color: #FF7A2F; background: linear-gradient(160deg, rgba(255,122,47,.20), #161C4A 58%); }
    .model-name { font-weight: 800; font-size: 1.3rem; margin: .9rem 0 .35rem; }
    .model-desc { color: #A3ABD8; font-size: .9rem; line-height: 1.5; min-height: 3.1em; }
    .big-num { font-family: 'Unbounded', sans-serif; font-weight: 800; font-size: 2.7rem; letter-spacing: -0.03em; margin-top: 1.2rem; }
    .big-lbl { color: #A3ABD8; font-size: .82rem; }
    .mini-row { display: flex; gap: 1.6rem; margin-top: 1.2rem; padding-top: 1.1rem; border-top: 1px solid #2A3270; }
    .mini-row b { display: block; font-size: 1.02rem; }
    .mini-row span { color: #A3ABD8; font-size: .78rem; }

    /* ---- Class guide ---- */
    .cls-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(215px, 1fr)); gap: 14px; }
    .cls-card { background: #161C4A; border: 1px solid #2A3270; border-radius: 20px; padding: 1.2rem 1.3rem; }
    .cls-ico { font-size: 1.9rem; }
    .cls-name { font-weight: 800; margin: .6rem 0 .3rem; }
    .cls-desc { color: #A3ABD8; font-size: .86rem; line-height: 1.45; }
    .cls-cue { margin-top: .7rem; padding-top: .7rem; border-top: 1px solid #2A3270; color: #B4C0FF; font-size: .8rem; line-height: 1.45; }

    @media (max-width: 640px) {
        .hero { padding: 2rem 1.4rem; }
        .nav-tags { display: none; }
        .bar-label { width: 120px; }
        .ring { width: 84px; height: 84px; }
    }
</style>
""", unsafe_allow_html=True)

BAR_COLORS = ["linear-gradient(90deg,#FF7A2F,#FFC14D)", "#8EA2FF", "#6F83E6", "#5568C4", "#3F4E9E"]


def prob_bars(probs, order):
    return "".join(
        f'<div class="bar-row">'
        f'<span class="bar-label">{ICONS[CLASSES[i]]} {DISPLAY_NAMES[CLASSES[i]]}</span>'
        f'<div class="bar-track"><div class="bar-fill" style="width:{probs[i]*100:.1f}%;'
        f'background:{BAR_COLORS[j]};"></div></div>'
        f'<span class="bar-val">{probs[i]:.1%}</span>'
        f'</div>'
        for j, i in enumerate(order[:5])
    )


# ----------------------------------------------------------- Nav + hero
st.markdown(
    "<div class='nav'>"
    "<div class='brand'>"
    "<svg width='28' height='28' viewBox='0 0 28 28'><circle cx='14' cy='14' r='11' fill='none' "
    "stroke='#8EA2FF' stroke-width='2'/><circle cx='14' cy='14' r='5' fill='#FF7A2F'/></svg>GeoScope</div>"
    "<div class='nav-tags'><span class='tag'>EuroSAT</span><span class='tag'>PyTorch</span></div>"
    "</div>"
    "<div class='hero'>"
    "<div class='hero-title'>See what the satellite model sees.</div>"
    "<div class='hero-sub'>Drop in a Sentinel-2 patch. GeoScope names the land use and paints the "
    "pixels that drove the decision.</div>"
    "</div>",
    unsafe_allow_html=True,
)

tab_classify, tab_models, tab_guide = st.tabs(["Classify", "Models", "Land-use guide"])

# ---------------------------------------------------------------- Classify
with tab_classify:
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns([1.2, 1.4, 1.1, 1.1], gap="medium")
        with c1:
            mode = st.radio("Mode", ["Single model", "Compare all 3"], horizontal=True)
        with c2:
            name = st.selectbox(
                "Model",
                list(MODELS),
                index=0,
                format_func=lambda m: f"{m} ({MODELS[m][6]})",
                disabled=(mode == "Compare all 3"),
            )
        with c3:
            lens = st.radio("Heatmap view", ["Overlay", "Heatmap only"], horizontal=True)
        with c4:
            alpha = st.slider(
                "Overlay opacity",
                0.15, 0.90, 0.45, 0.05,
                disabled=(lens == "Heatmap only"),
            )

        if mode == "Single model":
            st.markdown(
                f"<span class='pill-pop'>{MODELS[name][6]}</span>"
                f"<span class='hint'>{MODELS[name][7]} Test accuracy: <b>{MODELS[name][2]}</b></span>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<span class='pill-peri'>Ensemble</span>"
                "<span class='hint'>Runs all three models side by side. "
                "Trust <b>ViT-B/16</b> as the tie-breaker on ambiguous patches.</span>",
                unsafe_allow_html=True,
            )

    pure_map = (lens == "Heatmap only")

    # ========================================================== SINGLE MODEL
    if mode == "Single model":
        col_l, col_r = st.columns([1, 1.25], gap="large")

        with col_l:
            with st.container(border=True):
                st.markdown("<div class='card-title'>Upload a satellite patch</div>", unsafe_allow_html=True)
                up = st.file_uploader(
                    "PNG, JPG, or TIF",
                    type=["png", "jpg", "jpeg", "tif", "tiff"],
                    label_visibility="collapsed",
                )

                if up is not None:
                    try:
                        img = Image.open(up).convert("RGB")
                    except Exception:
                        st.error("Could not read this file as an image.")
                        st.stop()

                    with st.spinner(f"Running {name}..."):
                        r = predict(name, img)

                    if r is None:
                        st.error(f"Model weights for {name} not found in `{CKPT_DIR}/`.")
                        st.stop()

                    st.markdown("<div class='card-title' style='margin-top:.8rem'>Original and heatmap</div>",
                                unsafe_allow_html=True)
                    cam_vis = render_heatmap(img, r["cam"], alpha, pure=pure_map)
                    v1, v2 = st.columns(2, gap="medium")
                    with v1:
                        show_img(upscale(img), caption=f"Original ({img.size[0]}×{img.size[1]})")
                    with v2:
                        show_img(cam_vis, caption=f"Grad-CAM, {r['ms']:.0f} ms")

                    st.caption("Bright yellow and orange: high influence. Dark purple: low influence.")

                    st.download_button(
                        label="Download heatmap (PNG)",
                        data=image_to_bytes(cam_vis),
                        file_name=f"geoscope_{CLASSES[r['idx']].lower()}_gradcam.png",
                        mime="image/png",
                        use_container_width=True,
                    )

        with col_r:
            with st.container(border=True):
                if up is None:
                    st.markdown(
                        "<div class='card-title'>Waiting for a patch</div>"
                        "<div class='analysis-callout'>Upload an image on the left. You'll get the predicted "
                        "land-use class, a confidence breakdown, and a Grad-CAM map of what drove the call.</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    order = np.argsort(r["probs"])[::-1]
                    top_i, runner_i = int(order[0]), int(order[1])
                    cls = CLASSES[top_i]
                    runner_cls = CLASSES[runner_i]
                    conf = float(r["probs"][top_i])
                    runner_conf = float(r["probs"][runner_i])
                    margin = (conf - runner_conf) * 100

                    st.markdown(
                        f"<div class='result-head'><div>"
                        f"<div class='result-class'>{ICONS[cls]} {DISPLAY_NAMES[cls]}</div>"
                        f"<div class='result-desc'>{DESCRIPTIONS[cls]}</div>"
                        f"<span class='pill-pop'>+{margin:.1f} pts over {DISPLAY_NAMES[runner_cls]}</span>"
                        f"</div>"
                        f"<div class='ring' style='--p:{conf*100:.1f}'><span>{conf:.1%}</span></div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                    certainty = (
                        "high-certainty" if conf >= 0.85 else
                        "moderate-confidence" if conf >= 0.60 else
                        "low-confidence"
                    )
                    st.markdown(
                        f"<div class='analysis-callout'>"
                        f"<b>{name}</b> made a <b>{certainty}</b> call: {DISPLAY_NAMES[cls]} at {conf:.1%}, "
                        f"with {DISPLAY_NAMES[runner_cls]} next at {runner_conf:.1%}. "
                        f"The bright regions in the heatmap mark where it found {VISUAL_CUES[cls]}."
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                    if conf < 0.60:
                        st.warning("Borderline confidence. Switch Mode to 'Compare all 3' to cross-check.")

                    st.markdown("<div class='card-title'>Top 5 probabilities</div>", unsafe_allow_html=True)
                    st.markdown(prob_bars(r["probs"], order), unsafe_allow_html=True)

    # ========================================================= COMPARE ALL 3
    else:
        with st.container(border=True):
            st.markdown("<div class='card-title'>Upload a patch to run all three models</div>", unsafe_allow_html=True)
            up = st.file_uploader(
                "PNG, JPG, or TIF",
                type=["png", "jpg", "jpeg", "tif", "tiff"],
                label_visibility="collapsed",
            )

        if up is not None:
            try:
                img = Image.open(up).convert("RGB")
            except Exception:
                st.error("Could not read this file as an image.")
                st.stop()

            cols = st.columns([1, 1, 1, 1], gap="medium")
            with cols[0]:
                with st.container(border=True):
                    st.markdown("<div class='card-title'>Original</div>", unsafe_allow_html=True)
                    show_img(upscale(img), caption=f"{img.size[0]}×{img.size[1]} px")

            results = {}
            for col, n in zip(cols[1:], MODELS):
                with col:
                    with st.container(border=True):
                        st.markdown(f"<div class='card-title'>{n}</div>"
                                    f"<span class='pill-peri'>{MODELS[n][6]}</span>", unsafe_allow_html=True)
                        with st.spinner("Running..."):
                            r = predict(n, img)
                        if r is None:
                            st.error("Missing weights")
                            continue
                        results[n] = r
                        cls = CLASSES[r["idx"]]
                        conf = float(r["probs"][r["idx"]])
                        cam_vis = render_heatmap(img, r["cam"], alpha, pure=pure_map)
                        show_img(cam_vis, caption=f"Grad-CAM, {r['ms']:.0f} ms")
                        st.markdown(f"**{ICONS[cls]} {DISPLAY_NAMES[cls]}**")
                        st.caption(f"Confidence: **{conf:.1%}**")

            if len(results) > 1:
                with st.container(border=True):
                    st.markdown("<div class='card-title'>Model comparison</div>", unsafe_allow_html=True)
                    preds = [DISPLAY_NAMES[CLASSES[r["idx"]]] for r in results.values()]
                    unique_preds = set(preds)
                    avg_probs = np.mean([r["probs"] for r in results.values()], axis=0)
                    ens_order = np.argsort(avg_probs)[::-1]
                    ens_top = CLASSES[int(ens_order[0])]
                    ens_runner = CLASSES[int(ens_order[1])]

                    best_model = max(results.items(), key=lambda kv: float(kv[1]["probs"][kv[1]["idx"]]))
                    fastest_model = min(results.items(), key=lambda kv: kv[1]["ms"])

                    if len(unique_preds) == 1:
                        st.markdown(
                            f"<div class='analysis-callout'>"
                            f"<b>All three agree: {preds[0]}.</b> Mean confidence is "
                            f"<b>{avg_probs[ens_order[0]]:.1%}</b>. <b>{best_model[0]}</b> was the most "
                            f"decisive ({float(best_model[1]['probs'][best_model[1]['idx']]):.1%}) and "
                            f"<b>{fastest_model[0]}</b> ran fastest ({fastest_model[1]['ms']:.0f} ms). "
                            f"In every heatmap the bright regions line up with {VISUAL_CUES[ens_top]}."
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                    else:
                        breakdown = ", ".join(
                            f"<b>{m}</b> says {DISPLAY_NAMES[CLASSES[res['idx']]]} "
                            f"({float(res['probs'][res['idx']]):.1%})"
                            for m, res in results.items()
                        )
                        st.markdown(
                            f"<div class='analysis-callout' style='border-left-color:#FF7A2F;'>"
                            f"<b>Split decision.</b> {breakdown}. Averaged across all three, "
                            f"<b>{DISPLAY_NAMES[ens_top]}</b> ({avg_probs[ens_order[0]]:.1%}) edges out "
                            f"<b>{DISPLAY_NAMES[ens_runner]}</b> ({avg_probs[ens_order[1]]:.1%}). "
                            f"On borderline patches, go with <b>ViT-B/16</b> (98.84% test accuracy)."
                            f"</div>",
                            unsafe_allow_html=True,
                        )

                    st.markdown("<div class='card-title'>Ensemble average, top 5</div>", unsafe_allow_html=True)
                    st.markdown(prob_bars(avg_probs, ens_order), unsafe_allow_html=True)

# ----------------------------------------------------------------- Models
with tab_models:
    cols = st.columns(3, gap="medium")
    pill_classes = ["pill-pop", "pill-peri", "pill-light"]
    for col, (n, v), p_cls in zip(cols, MODELS.items(), pill_classes):
        with col:
            rec = " rec" if v[6] == "Recommended" else ""
            st.markdown(
                f"<div class='model-card{rec}'>"
                f"<span class='{p_cls}'>{v[6]}</span>"
                f"<div class='model-name'>{n}</div>"
                f"<div class='model-desc'>{v[7]}</div>"
                f"<div class='big-num'>{v[2]}</div>"
                f"<div class='big-lbl'>test accuracy</div>"
                f"<div class='mini-row'>"
                f"<div><b>{v[3]}</b><span>per image</span></div>"
                f"<div><b>{v[4]}</b><span>parameters</span></div>"
                f"<div><b>{v[5]}</b><span>checkpoint</span></div>"
                f"</div></div>",
                unsafe_allow_html=True,
            )

# ------------------------------------------------------------------ Guide
with tab_guide:
    cards = "".join(
        f"<div class='cls-card'><div class='cls-ico'>{ICONS[c]}</div>"
        f"<div class='cls-name'>{DISPLAY_NAMES[c]}</div>"
        f"<div class='cls-desc'>{DESCRIPTIONS[c]}</div>"
        f"<div class='cls-cue'>Look for {VISUAL_CUES[c]}.</div></div>"
        for c in CLASSES
    )
    st.markdown(f"<div class='cls-grid'>{cards}</div>", unsafe_allow_html=True)