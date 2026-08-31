import numpy as np
import streamlit as st
import joblib

st.set_page_config(page_title="제주 감귤 당도 예측", page_icon="🍊", layout="centered")

st.markdown(
    """
    <style>
        :root {
            --bg-main: #f9f2ea;
            --bg-soft: #fffaf5;
            --panel: rgba(255, 255, 255, 0.7);
            --primary: #d78b4a;
            --primary-deep: #b86b3d;
            --accent: #f3c89d;
            --text: #4d352b;
            --muted: #7a5d4d;
            --line: rgba(184, 107, 61, 0.18);
            --success-bg: rgba(134, 172, 109, 0.12);
            --info-bg: rgba(212, 142, 87, 0.12);
        }

        .stApp {
            background: linear-gradient(180deg, #f9f2ea 0%, #f4e6d6 100%);
            color: var(--text);
        }

        .block-container {
            position: relative;
            z-index: 1;
            padding-top: 2.5rem;
            padding-bottom: 3rem;
        }

        h1, h2, h3 {
            color: var(--primary-deep) !important;
            font-weight: 700;
        }

        .stCaptionContainer {
            color: var(--muted) !important;
        }

        div[data-testid="stForm"] {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 20px;
            padding: 1.2rem;
            box-shadow: 0 8px 24px rgba(122, 93, 77, 0.08);
        }

        div[data-testid="stVerticalBlock"] > div {
            gap: 0.8rem;
        }

        div.stNumberInput > div,
        div.stTextInput > div,
        div.stSelectbox > div,
        div.stDateInput > div {
            background: rgba(255, 255, 255, 0.7);
            border: 1px solid var(--line);
            border-radius: 12px;
            box-shadow: none;
        }

        .stButton > button {
            background: linear-gradient(135deg, #e9b27b 0%, var(--primary) 100%);
            color: #fffaf4;
            border: none;
            border-radius: 12px;
            padding: 0.6rem 1.4rem;
            font-weight: 700;
            box-shadow: 0 8px 20px rgba(184, 107, 61, 0.25);
        }

        .stButton > button:hover {
            background: linear-gradient(135deg, #f2c38d 0%, var(--primary-deep) 100%);
            box-shadow: 0 10px 24px rgba(184, 107, 61, 0.32);
        }

        .stSuccess > div {
            background: var(--success-bg);
            border: 1px solid rgba(134, 172, 109, 0.3);
            color: #46603d;
            border-radius: 12px;
        }

        .stInfo > div {
            background: var(--info-bg);
            border: 1px solid rgba(215, 139, 74, 0.28);
            color: var(--text);
            border-radius: 12px;
        }

        .stMarkdown p, .stMarkdown li {
            color: var(--text);
        }
    </style>

    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    return joblib.load("brix_model.joblib")


model = load_model()
feature_names = getattr(model, "feature_names_in_", ["평균기온", "최저기온", "가조시간", "최저초상온도"])

st.title("🍊 제주도 성산지역 감귤 당도 예측")
st.caption("회귀 모델 기반으로 평균기온, 최저기온, 가조시간, 최저초상온도를 입력하면 당도를 예측합니다.")

with st.form("brix_form"):
    cols = st.columns(2)

    with cols[0]:
        avg_temp = st.number_input("평균기온 (°C)", min_value=-20.0, max_value=50.0, value=22.0, step=0.1)
        min_temp = st.number_input("최저기온 (°C)", min_value=-20.0, max_value=40.0, value=16.0, step=0.1)

    with cols[1]:
        sunshine_hours = st.number_input("가조시간 (시간)", min_value=0.0, max_value=20.0, value=9.5, step=0.1)
        min_ground_temp = st.number_input("최저 초상온도 (°C)", min_value=-20.0, max_value=40.0, value=12.0, step=0.1)

    submitted = st.form_submit_button("당도 예측하기")

if submitted:
    inputs = np.array(
        [[avg_temp, min_temp, sunshine_hours, min_ground_temp]],
        dtype=float,
    )

    prediction = float(model.predict(inputs)[0])

    st.subheader("예측 결과")
    st.success(f"예측 당도(Brix): {prediction:.2f} °Bx")

    st.write("입력값 요약")
    for name, value in zip(feature_names, inputs[0]):
        st.write(f"- {name}: {value:.2f}")

    if prediction < 10:
        st.info("당도가 낮은 편입니다. 향후 수확 시기와 관리 조건을 점검해 보세요.")
    elif prediction < 13:
        st.info("적정 수준의 당도입니다.")
    else:
        st.info("당도가 높은 편입니다. 우수한 품질의 감귤로 예상됩니다.")
