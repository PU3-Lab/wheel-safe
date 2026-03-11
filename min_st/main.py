import base64
import time
import random
from io import BytesIO
from typing import Dict, Optional

import requests
import streamlit as st
from PIL import Image, ImageOps

from streamlit_extras.stylable_container import stylable_container

# ---------------------------------------------------
# Page Config
# ---------------------------------------------------
st.set_page_config(
    page_title="Wheel-Safe",
    page_icon="♿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------
# Theme / CSS
# ---------------------------------------------------
PRIMARY = "#346E4C"
PRIMARY_DARK = "#234631"
BG = "#F4F7F4"
CARD = "#FFFFFF"
TEXT = "#1F2A24"
MUTED = "#5F6B64"
SAFE = "#2ECC71"
CAUTION = "#F1C40F"
DANGER = "#E74C3C"

st.markdown(
    f"""
    <style>
    :root {{
        --primary: {PRIMARY};
        --primary-dark: {PRIMARY_DARK};
        --bg: {BG};
        --card: {CARD};
        --text: {TEXT};
        --muted: {MUTED};
        --safe: {SAFE};
        --caution: {CAUTION};
        --danger: {DANGER};
        --radius-xl: 28px;
        --radius-lg: 20px;
        --shadow: 0 10px 30px rgba(35, 70, 49, 0.10);
    }}

    .stApp {{
        background:
            radial-gradient(circle at top left, rgba(52,110,76,0.10), transparent 30%),
            linear-gradient(180deg, #F7FAF8 0%, #EEF4EF 100%);
        color: var(--text);
    }}

    .block-container {{
        max-width: 1200px;
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }}

    .app-shell {{
        background: rgba(255,255,255,0.62);
        border: 1px solid rgba(52,110,76,0.08);
        border-radius: 32px;
        padding: 20px;
        box-shadow: var(--shadow);
        backdrop-filter: blur(8px);
    }}

    .topbar {{
        display:flex;
        justify-content:space-between;
        align-items:center;
        margin-bottom:18px;
        padding: 12px 14px;
        border-radius: 22px;
        background: rgba(255,255,255,0.7);
        border: 1px solid rgba(52,110,76,0.08);
    }}

    .brand {{
        display:flex;
        gap:14px;
        align-items:center;
    }}

    .brand-icon {{
        width:56px;
        height:56px;
        border-radius:18px;
        background: linear-gradient(135deg, var(--primary), var(--primary-dark));
        color:white;
        display:flex;
        align-items:center;
        justify-content:center;
        font-size:30px;
        font-weight:700;
        box-shadow: 0 8px 24px rgba(35,70,49,0.25);
    }}

    .brand-title {{
        font-size:28px;
        font-weight:800;
        line-height:1.05;
        color: var(--primary-dark);
    }}

    .brand-sub {{
        font-size:14px;
        color: var(--muted);
        margin-top:4px;
    }}

    .status-chip {{
        padding: 10px 16px;
        border-radius: 999px;
        background: rgba(52,110,76,0.10);
        color: var(--primary-dark);
        font-weight: 700;
        font-size: 14px;
    }}

    .hero-card, .content-card, .result-card, .metric-card, .soft-card {{
        background: var(--card);
        border-radius: var(--radius-xl);
        padding: 24px;
        border: 1px solid rgba(52,110,76,0.08);
        box-shadow: var(--shadow);
    }}

    .hero-card {{
        padding: 34px 28px;
        background:
            linear-gradient(135deg, rgba(52,110,76,0.96), rgba(35,70,49,0.98));
        color: white;
        position: relative;
        overflow: hidden;
    }}

    .hero-card:before {{
        content: "";
        position: absolute;
        right: -40px;
        top: -40px;
        width: 180px;
        height: 180px;
        border-radius: 50%;
        background: rgba(255,255,255,0.08);
    }}

    .hero-title {{
        font-size: 30px;
        font-weight: 900;
        line-height: 1.05;
        margin-bottom: 10px;
        position: relative;
        z-index: 1;
    }}

    .hero-desc {{
        font-size: 17px;
        line-height: 1.6;
        color: rgba(255,255,255,0.92);
        max-width: 720px;
        position: relative;
        z-index: 1;
    }}

    .section-title {{
        font-size: 22px;
        font-weight: 800;
        color: var(--primary-dark);
        margin-bottom: 8px;
    }}

    .section-desc {{
        color: var(--muted);
        font-size: 15px;
        line-height: 1.6;
        margin-bottom: 18px;
    }}

    .choice-card {{
        background: white;
        border-radius: 24px;
        padding: 24px;
        border: 2px solid rgba(52,110,76,0.10);
        min-height: 220px;
        transition: all 0.2s ease;
    }}

    .choice-card:hover {{
        border-color: rgba(52,110,76,0.28);
        transform: translateY(-2px);
    }}

    .choice-icon {{
        width:72px;
        height:72px;
        border-radius:20px;
        background: rgba(52,110,76,0.12);
        display:flex;
        align-items:center;
        justify-content:center;
        font-size:36px;
        margin-bottom:16px;
    }}

    .choice-title {{
        font-size:24px;
        font-weight:800;
        color: var(--primary-dark);
        margin-bottom:8px;
    }}

    .choice-desc {{
        font-size:15px;
        color: var(--muted);
        line-height:1.6;
        min-height:72px;
    }}

    .risk-badge {{
        display:inline-flex;
        align-items:center;
        justify-content:center;
        gap:8px;
        padding: 12px 18px;
        border-radius: 999px;
        font-size: 18px;
        font-weight: 900;
        letter-spacing: 0.3px;
    }}

    .risk-safe {{
        background: rgba(46,204,113,0.14);
        color: var(--safe);
    }}

    .risk-caution {{
        background: rgba(241,196,15,0.18);
        color: #A37700;
    }}

    .risk-danger {{
        background: rgba(231,76,60,0.14);
        color: var(--danger);
    }}

    .angle-panel {{
        text-align:center;
        padding: 30px 16px 18px 16px;
        border-radius: 28px;
        background: linear-gradient(180deg, #F8FBF9 0%, #F1F6F2 100%);
        border: 1px solid rgba(52,110,76,0.08);
    }}

    .angle-value {{
        font-size: 78px;
        font-weight: 900;
        line-height: 1;
        margin: 10px 0 8px 0;
        letter-spacing: -2px;
    }}

    .angle-label {{
        font-size: 15px;
        color: var(--muted);
        font-weight: 600;
    }}

    .gauge-wrap {{
        margin-top: 18px;
        margin-bottom: 8px;
    }}

    .gauge-bar {{
        height: 18px;
        border-radius: 999px;
        background:
          linear-gradient(90deg,
            var(--safe) 0%,
            var(--safe) 30%,
            var(--caution) 30%,
            var(--caution) 70%,
            var(--danger) 70%,
            var(--danger) 100%);
        position: relative;
        overflow: visible;
    }}

    .gauge-marker {{
        position: absolute;
        top: -10px;
        width: 24px;
        height: 38px;
        border-radius: 16px;
        background: #111;
        border: 4px solid white;
        box-shadow: 0 6px 16px rgba(0,0,0,0.18);
        transform: translateX(-50%);
    }}

    .gauge-labels {{
        display:flex;
        justify-content:space-between;
        font-size: 13px;
        color: var(--muted);
        margin-top: 8px;
        font-weight: 700;
    }}

    .metric-card {{
        text-align:center;
        min-height:140px;
        display:flex;
        flex-direction:column;
        justify-content:center;
    }}

    .metric-title {{
        font-size:14px;
        color: var(--muted);
        font-weight:700;
        margin-bottom:8px;
    }}

    .metric-value {{
        font-size:36px;
        font-weight:900;
        color: var(--primary-dark);
    }}

    .subtle {{
        color: var(--muted);
        font-size: 14px;
        line-height: 1.6;
    }}

    .footer-hint {{
        margin-top: 16px;
        padding: 14px 16px;
        border-radius: 18px;
        background: rgba(52,110,76,0.06);
        color: var(--primary-dark);
        font-weight: 600;
        font-size: 14px;
    }}

    .splash-wrap {{
        min-height: 71vh;
        display:flex;
        align-items:center;
        justify-content:center;
    }}

    .splash-card {{
        width: 100%;
        max-width: 760px;
        background: linear-gradient(135deg, rgba(52,110,76,0.98), rgba(35,70,49,1));
        color: white;
        border-radius: 40px;
        padding: 60px 40px;
        text-align:center;
        box-shadow: 0 22px 60px rgba(35,70,49,0.30);
        position: relative;
        overflow: hidden;
    }}

    .splash-card:before, .splash-card:after {{
        content:"";
        position:absolute;
        border-radius:50%;
        background: rgba(255,255,255,0.08);
    }}

    .splash-card:before {{
        width:240px; height:240px; top:-80px; right:-80px;
    }}

    .splash-card:after {{
        width:180px; height:180px; bottom:-70px; left:-70px;
    }}

    .splash-logo {{
        width:100px;
        height:100px;
        margin:0 auto 20px auto;
        border-radius:28px;
        background: rgba(255,255,255,0.14);
        display:flex;
        align-items:center;
        justify-content:center;
        font-size:48px;
        animation: pulse 1.8s infinite;
    }}

    .splash-title {{
        font-size:54px;
        font-weight:900;
        margin-bottom:10px;
    }}

    .splash-desc {{
        font-size:18px;
        color: rgba(255,255,255,0.90);
        margin-bottom:26px;
    }}

    .loading-dots span {{
        display:inline-block;
        width:10px;
        height:10px;
        margin:0 5px;
        border-radius:50%;
        background:white;
        opacity:.35;
        animation: blink 1.2s infinite;
    }}

    .loading-dots span:nth-child(2) {{ animation-delay: 0.2s; }}
    .loading-dots span:nth-child(3) {{ animation-delay: 0.4s; }}

    @keyframes blink {{
        0%, 80%, 100% {{ opacity: .25; transform: translateY(0); }}
        40% {{ opacity: 1; transform: translateY(-4px); }}
    }}

    @keyframes pulse {{
        0% {{ transform: scale(1); }}
        50% {{ transform: scale(1.04); }}
        100% {{ transform: scale(1); }}
    }}

    div.stButton > button {{
        width: 100%;
        min-height: 64px;
        border-radius: 18px;
        border: none;
        background: linear-gradient(135deg, var(--primary), var(--primary-dark));
        color: white;
        font-size: 18px;
        font-weight: 800;
        box-shadow: 0 12px 24px rgba(35,70,49,0.18);
    }}

    div.stButton > button:hover {{
        filter: brightness(1.03);
    }}

    .secondary-button-note {{
        color: var(--muted);
        font-size: 13px;
        margin-top: 8px;
    }}

    [data-testid="stFileUploader"] {{
        background: white;
        border: 2px dashed rgba(52,110,76,0.26);
        border-radius: 22px;
        padding: 18px;
    }}

    [data-testid="stMetric"] {{
        background: white;
        border-radius: 20px;
        padding: 14px;
        border: 1px solid rgba(52,110,76,0.08);
        box-shadow: var(--shadow);
    }}

    .img-panel {{
        background:white;
        border-radius:28px;
        padding:16px;
        border:1px solid rgba(52,110,76,0.08);
        box-shadow: var(--shadow);
    }}

    .tiny {{
        font-size:12px;
        color: var(--muted);
    }}
    
    .segmented-progress-wrap {{
        margin-top: 20px;
    }}

    .segmented-progress-track {{
        position: relative;
        width: 100%;
        height: 28px;
        border-radius: 999px;
        overflow: hidden;
        display: flex;
        background: #E9EFEA;
        border: 1px solid rgba(52,110,76,0.10);
    }}

    .seg-safe,
    .seg-caution,
    .seg-danger {{
        height: 100%;
    }}

    .seg-safe {{
        width: 25%;
        background: rgba(46, 204, 113, 0.20);
    }}

    .seg-caution {{
        width: 33.333%;
        background: rgba(241, 196, 15, 0.22);
    }}

    .seg-danger {{
        width: 41.667%;
        background: rgba(231, 76, 60, 0.18);
    }}

    .segmented-progress-fill {{
        position: absolute;
        top: 0;
        left: 0;
        height: 100%;
        width: 0;
        border-radius: 999px;
        z-index: 2;
        animation: fillBar 1.4s ease-out forwards;
        box-shadow: 0 4px 14px rgba(0,0,0,0.12);
    }}

    .fill-safe {{
        background: linear-gradient(90deg, #2ECC71, #27AE60);
    }}

    .fill-caution {{
        background: linear-gradient(90deg, #F1C40F, #D4AC0D);
    }}

    .fill-danger {{
        background: linear-gradient(90deg, #E74C3C, #C0392B);
    }}

    .segmented-progress-marker {{
        position: absolute;
        top: 50%;
        transform: translate(-50%, -50%);
        width: 18px;
        height: 18px;
        border-radius: 50%;
        background: #FFFFFF;
        border: 3px solid #1F2A24;
        z-index: 3;
        box-shadow: 0 2px 10px rgba(0,0,0,0.18);
        animation: markerFadeIn 1.5s ease-out forwards;
        opacity: 0;
    }}

    .segmented-progress-labels {{
        display: flex;
        justify-content: space-between;
        margin-top: 10px;
        gap: 8px;
    }}

    .segmented-progress-labels span {{
        flex: 1;
        text-align: center;
        font-size: 13px;
        font-weight: 800;
        color: #5F6B64;
        line-height: 1.35;
    }}

    @keyframes fillBar {{
        from {{
            width: 0;
        }}
        to {{
            width: var(--target-width);
        }}
    }}

    @keyframes markerFadeIn {{
        0% {{
            opacity: 0;
            transform: translate(-50%, -50%) scale(0.7);
        }}
        100% {{
            opacity: 1;
            transform: translate(-50%, -50%) scale(1);
        }}
    }}

    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------
# Session State
# ---------------------------------------------------
DEFAULTS = {
    "screen": "splash",
    "mode": None,
    "image_bytes": None,
    "image_name": None,
    "result": None,
    "server_url": "http://127.0.0.1:8000/predict",
}

for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ---------------------------------------------------
# Helpers
# ---------------------------------------------------
def reset_flow() -> None:
    st.session_state.screen = "select"
    st.session_state.mode = None
    st.session_state.image_bytes = None
    st.session_state.image_name = None
    st.session_state.result = None


def set_mode(mode: str) -> None:
    st.session_state.mode = mode
    st.session_state.image_bytes = None
    st.session_state.image_name = None
    st.session_state.result = None
    st.session_state.screen = "input"


def go_result() -> None:
    st.session_state.screen = "result"

def go_processing() -> None:
    st.session_state.screen = "processing"

# def save_uploaded_image(uploaded_file) -> None:
#     if uploaded_file is None:
#         return
#     st.session_state.image_bytes = uploaded_file.getvalue()
#     st.session_state.image_name = uploaded_file.name
#     st.session_state.result = None

def save_uploaded_image(uploaded_file) -> None:
    if uploaded_file is None:
        return

    # 1) 원본 바이트 읽기
    raw_bytes = uploaded_file.getvalue()

    # 2) PIL 로드
    image = Image.open(BytesIO(raw_bytes))

    # 3) EXIF Orientation 반영하여 실제 픽셀 회전/전치
    image = ImageOps.exif_transpose(image)

    # 4) 필요 시 색상 모드 정리
    if image.mode not in ("RGB", "RGBA"):
        image = image.convert("RGB")

    # 5) 다시 바이트로 저장
    output = BytesIO()
    save_format = "PNG" if image.mode == "RGBA" else "JPEG"
    image.save(output, format=save_format)

    st.session_state.image_bytes = output.getvalue()
    st.session_state.image_name = uploaded_file.name
    st.session_state.result = None

def save_camera_image(camera_file) -> None:
    if camera_file is None:
        return
    st.session_state.image_bytes = camera_file.getvalue()
    st.session_state.image_name = "camera_capture.jpg"
    st.session_state.result = None

# def image_from_session() -> Optional[Image.Image]:
#     if not st.session_state.image_bytes:
#         return None
#     return Image.open(BytesIO(st.session_state.image_bytes))

def image_from_session() -> Optional[Image.Image]:
    if not st.session_state.image_bytes:
        return None
    return Image.open(BytesIO(st.session_state.image_bytes))

def classify_risk(angle: float) -> str:
    if angle < 3:
        return "SAFE"
    if angle < 7:
        return "CAUTION"
    return "DANGER"


def risk_color(angle: float) -> str:
    risk = classify_risk(angle)
    if risk == "SAFE":
        return SAFE
    if risk == "CAUTION":
        return "#C99000"
    return DANGER


def risk_class(angle: float) -> str:
    risk = classify_risk(angle)
    if risk == "SAFE":
        return "risk-safe"
    if risk == "CAUTION":
        return "risk-caution"
    return "risk-danger"


def gauge_left_percent(angle: float) -> float:
    capped = min(max(angle, 0), 12)
    return (capped / 12.0) * 100.0

# UPDATE
def request_prediction(image_bytes: bytes, server_url: str, image_name: str) -> Dict:
    files = {
        "file": (image_name, image_bytes, "image/jpeg")
    }

    response = requests.post(server_url, files=files, timeout=60)
    response.raise_for_status()

    data = response.json()

    angle = float(data["predicted_angle"])

    return {
        "filename": data.get("filename", image_name),
        "angle": angle,
        "risk": classify_risk(angle),
        "unit": data.get("unit", "degree"),
        "grad_cam_img": data.get("grad_cam_img"),
    }

def processing_pipeline() -> None:
    if not st.session_state.image_bytes:
        st.warning("먼저 이미지를 등록해주세요.")
        st.session_state.screen = "input"
        st.rerun()
    try:
        with st.spinner("도로 경사도를 분석하는 중입니다..."):
            progress_slot = st.empty()
            bar = progress_slot.progress(0, text="이미지 업로드 준비 중...")
            time.sleep(0.2)
            bar.progress(15, text="이미지 전처리 중...")
            time.sleep(0.2)
            bar.progress(35, text="서버로 이미지 전송 중...")
            time.sleep(0.2)
            bar.progress(60, text="모델 추론 요청 중...")

            result = request_prediction(
                image_bytes=st.session_state.image_bytes,
                server_url=st.session_state.server_url,
                image_name=st.session_state.image_name or "input.jpg",
            )

            bar.progress(85, text="Grad-CAM 결과 정리 중...")
            time.sleep(0.2)

            st.session_state.result = result

            bar.progress(100, text="결과 생성 완료")
            time.sleep(0.3)

        go_result()
        st.rerun()

    except requests.RequestException as e:
        st.error(f"서버 요청 중 오류가 발생했습니다: {e}")
        st.session_state.screen = "input"
        st.rerun()

    except Exception as e:
        st.error(f"분석 중 오류가 발생했습니다: {e}")
        st.session_state.screen = "input"
        st.rerun()

# ---------------------------------------------------
# Shared Layout
# ---------------------------------------------------
def render_topbar(current_label: str) -> None:
    st.markdown(
        f"""
        <div class="topbar">
            <div class="brand">
                <div class="brand-icon">♿</div>
                <div>
                    <div class="brand-title">Wheel-Safe</div>
                </div>
            </div>
            <div class="status-chip">{current_label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_shell_start(label: str) -> None:
    st.markdown('<div class="app-shell">', unsafe_allow_html=True)
    render_topbar(label)


def render_shell_end() -> None:
    st.markdown('</div>', unsafe_allow_html=True)


# ---------------------------------------------------
# Screens
# ---------------------------------------------------
def screen_splash() -> None:
    st.markdown(
        """
        <div class="splash-wrap">
            <div class="splash-card">
                <div class="splash-logo">♿</div>
                <div class="splash-title">Wheel-Safe</div>
                <div class="splash-desc">
                    스마트폰 이미지 기반 도로 경사도 추정 시스템<br/>
                    휠체어 사용자를 위한 안전한 이동 의사결정 UI
                </div>
                <div class="loading-dots"><span></span><span></span><span></span></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # col1, col2, col3 = st.columns([1.2, 1, 1.2]) # 화면을 좌우로 분할할 때 사용
    # with col2:
    #     left, center, right = st.columns([1,2,1])
    #     with center:

    with stylable_container(
        key="center-button",
        css_styles="""
        {
            display: flex;
            justify-content: center;
            align-items: center;
            margin-top: 0px;
        }
        """
    ):
        if st.button("시작하기", key="start_app"):
            # 현재 앱의 화면 상태를 select로 변경
            # st.session_state, stream lit에서 세션 동안 유지되는 상태 저장소
            st.session_state.screen = "select"
            st.rerun()


def screen_select() -> None:
    render_shell_start("입력 방식 선택")

    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">안전한 경사 판별,<br/>존중받는 휠체어</div>
            <div class="hero-desc">
                촬영 또는 갤러리 이미지를 선택한 뒤 이미지를 기반으로 도로 경사도를 추정합니다.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.markdown(
            """
            <div class="choice-card">
                <div class="choice-icon">📷</div>
                <div class="choice-title">실시간 촬영</div>
                <div class="choice-desc">
                    현장 도로를 즉시 촬영해 경사도를 분석합니다.
                    실제 운영에서는 비디오 스트림 연동으로 확장 가능합니다.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        with stylable_container(
        key="left-camera_mode",
        css_styles="""
        {
            margin-top: 10px;
        }
        """
        ):
            if st.button("카메라 모드 시작", key="camera_mode"):
                set_mode("camera")
                st.rerun()

    with c2:
        st.markdown(
            """
            <div class="choice-card">
                <div class="choice-icon">🖼️</div>
                <div class="choice-title">저장된 사진 불러오기</div>
                <div class="choice-desc">
                    갤러리 이미지를 업로드하여 경사도를 추정합니다.
                    테스트 데이터셋 검증이나 시연에 적합합니다.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with stylable_container(
        key="right-camera_mode",
        css_styles="""
        {
            margin-top: 10px;
        }
        """
        ):
            if st.button("갤러리 모드 시작", key="gallery_mode"):
                set_mode("gallery")
                st.rerun()

    st.markdown(
        """
        <div class="footer-hint">
            추천 화면 비율: 1280×800 태블릿 기준 · 한 손 조작을 고려한 대형 버튼 UI
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_shell_end()


def screen_input() -> None:
    mode = st.session_state.mode or "gallery"
    label = "실시간 촬영" if mode == "camera" else "이미지 업로드"
    render_shell_start(label)

    left, right = st.columns([1.15, 0.85], gap="large")

    with left:
        st.markdown(
            f"""
            <div class="content-card">
                <div class="section-title">{label}</div>
                <div class="section-desc">
                    {'카메라로 도로 사진을 촬영하고 분석을 시작하세요.' if mode == 'camera' else '분석할 도로 이미지를 업로드하세요.'}
                </div>
            """,
            unsafe_allow_html=True,
        )

        if mode == "camera":
            camera_file = st.camera_input("도로 촬영", key="camera_input_main")
            if camera_file is not None:
                save_camera_image(camera_file)
        else:
            uploaded = st.file_uploader(
                "이미지 업로드",
                type=["png", "jpg", "jpeg", "webp"],
                key="gallery_upload",
                help="도로가 잘 보이는 정면 또는 진행 방향 이미지를 권장합니다.",
            )
            if uploaded is not None:
                save_uploaded_image(uploaded)

        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown(
            """
            <div class="soft-card">
                <div class="section-title">입력 가이드</div>
                <div class="section-desc">
                    경사 추정 정확도를 위해 도로 진행 방향이 명확하고,
                    바닥 면적이 충분히 보이는 이미지를 사용하는 것이 좋습니다.
                </div>
                <div class="subtle">
                    • 보도/경사로가 중앙에 보이게 촬영<br/>
                    • 너무 어두운 사진은 피하기<br/>
                    • 흐림이 적은 이미지 권장<br/>
                    • 경계석, 턱, 장애물이 함께 보이면 해석 보조 가능
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

    preview = image_from_session()
    if preview:
        p1, p2 = st.columns([1.15, 0.85], gap="large")
        with p1:
            st.markdown('<div class="img-panel">', unsafe_allow_html=True)
            st.image(preview, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with p2:
            st.markdown(
                """
                <div class="content-card">
                    <div class="section-title">분석 준비 완료</div>
                    <div class="section-desc">
                        이미지를 서버로 전송하여 경사 각도와 위험도를 분석합니다.
                    </div>
                """,
                unsafe_allow_html=True,
            )

            if st.button("경사 분석 시작", key="start_analysis"):
                go_processing()
                st.rerun()

            if st.button("다시 선택하기", key="back_to_select_1"):
                reset_flow()
                st.rerun()
    else:
        c1, c2, c3 = st.columns([1, 1, 1])
        with c2:
            if st.button("메인 화면 이동", key="back_to_select_2"):
                reset_flow()
                st.rerun()

    render_shell_end()


def screen_processing() -> None:
    render_shell_start("AI 분석 중")

    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">이미지를 분석하고 있습니다</div>
            <div class="hero-desc">
                도로 영역 검출, 원근 기반 기울기 계산, 위험도 분류를 순차적으로 수행합니다.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([0.15, 0.7, 0.15])
    with col2:
        st.markdown(
            """
            <div class="content-card" style="text-align:center;">
                <div style="font-size:68px; margin-bottom:12px;">🛰️</div>
                <div class="section-title">Road Slope Estimation Pipeline</div>
                <div class="section-desc">잠시만 기다려주세요. 분석이 완료되면 결과 화면으로 이동합니다.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    processing_pipeline()
    render_shell_end()


def screen_result() -> None:
    result = st.session_state.result
    if not result:
        st.warning("결과가 없습니다. 먼저 분석을 진행해주세요.")
        st.session_state.screen = "select"
        st.rerun()

    angle = float(result["angle"])
    risk = result["risk"]
    grad_cam_img = result.get("grad_cam_img")
    
    color = risk_color(angle)
    risk_css = risk_class(angle)

    render_shell_start("결과 확인")

    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">도로 경사 분석 결과</div>
            <div class="hero-desc">
                각도와 위험도를 가장 우선적으로 보여줍니다.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

    left, right = st.columns([1.05, 0.95], gap="large")

    with left:
        st.markdown('<div class="img-panel">', unsafe_allow_html=True)
        if grad_cam_img:
            cam_bytes = base64.b64decode(grad_cam_img)
            cam_image = Image.open(BytesIO(cam_bytes))
            st.image(cam_image, use_container_width=True)
        else:
            st.info("Grad-CAM 이미지가 없습니다.")
        st.markdown("</div>", unsafe_allow_html=True)
        # original_col, cam_col = st.columns(2, gap="medium")

        # with original_col:
        #     st.markdown('<div class="img-panel">', unsafe_allow_html=True)
        #     img = image_from_session()
        #     if img:
        #         st.image(img, use_container_width=True)
        #     st.markdown("</div>", unsafe_allow_html=True)

        # with cam_col:
        #     st.markdown('<div class="img-panel">', unsafe_allow_html=True)
        #     if grad_cam_img:
        #         cam_bytes = base64.b64decode(grad_cam_img)
        #         cam_image = Image.open(BytesIO(cam_bytes))
        #         st.image(cam_image, use_container_width=True)
        #     else:
        #         st.info("Grad-CAM 이미지가 없습니다.")
        #     st.markdown("</div>", unsafe_allow_html=True)

    ui_max_angle = 12.0
    fill_percent = min(max((angle / ui_max_angle) * 100, 0), 100)

    if risk == "SAFE":
        fill_class = "fill-safe"
    elif risk == "CAUTION":
        fill_class = "fill-caution"
    else:
        fill_class = "fill-danger"
        
    with right:
        st.markdown(
            f"""
            <div class="result-card">
            <div class="angle-panel">
            <div class="angle-label">Estimated Road Slope</div>
            <div class="angle-value" style="color:{color};">{angle:.1f}°</div>
            <div class="risk-badge {risk_css}">{risk}</div>

            <div class="segmented-progress-wrap">
            <div class="segmented-progress-track">
            <div class="seg-safe"></div>
            <div class="seg-caution"></div>
            <div class="seg-danger"></div>

            <div
                class="segmented-progress-fill {fill_class}"
                style="--target-width: {fill_percent:.1f}%;"
            ></div>
            </div>

            <div class="segmented-progress-labels">
                <span>SAFE<br/>0°–3°</span>
                <span>CAUTION<br/>3°–7°</span>
                <span>DANGER<br/>&gt;7°</span>
            </div>
            </div>
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

    c1, c2 = st.columns([0.7, 0.3], gap="large")

    with c1:
        print('test!')
        # st.markdown(
        #     f"""
        #     <div class="content-card">
        #         <div class="section-title">서버 응답 정보</div>
        #         <div class="section-desc">
        #             실제 서비스에서는 아래 응답이 모델 서버에서 전달된다고 가정합니다.
        #         </div>
        #         <div class="subtle">
        #             • request_id: <b>{result.get("request_id", "-")}</b><br/>
        #             • model_version: <b>{result.get("model_version", "-")}</b><br/>
        #             • api_url: <b>{st.session_state.server_url}</b><br/>
        #             • recommended_action:
        #                 <b>{
        #                     "이동 가능" if risk == "SAFE"
        #                     else "주의 이동" if risk == "CAUTION"
        #                     else "우회 권장"
        #                 }</b>
        #         </div>
        #     </div>
        #     """,
        #     unsafe_allow_html=True,
        # )

    with c2:
        if st.button("메인 화면 이동", key="analyze_another"):
            st.session_state.image_bytes = None
            st.session_state.image_name = None
            st.session_state.result = None
            st.session_state.screen = "select"
            st.rerun()

        if st.button("다른 이미지 사용", key="same_mode_again"):
            st.session_state.image_bytes = None
            st.session_state.image_name = None
            st.session_state.result = None
            st.session_state.screen = "input"
            st.rerun()

    render_shell_end()


# ---------------------------------------------------
# Router
# ---------------------------------------------------
screen = st.session_state.screen

if screen == "splash":
    screen_splash()
elif screen == "select":
    screen_select()
elif screen == "input":
    screen_input()
elif screen == "processing":
    screen_processing()
elif screen == "result":
    screen_result()
else:
    st.session_state.screen = "splash"
    st.rerun()