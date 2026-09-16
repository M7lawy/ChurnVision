"""
ChurnVision — Customer Churn Prediction App
Built on top of the trained Random Forest pipeline from the ChurnVision project
(https://github.com/M7lawy/ChurnVision).

Run with:
    streamlit run app.py
"""

import os
import joblib
import pandas as pd
import streamlit as st

from utils.translations import TXT
from utils.styles import get_css


# ------------------------------------------------------------------
# Configurations & Setup
# ------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
CAT_COLS = ["InternetService", "Contract", "PaymentMethod"]
NUM_COLS = ["tenure", "MonthlyCharges", "TotalServicesCount"]
BINARY_COLS = ["SeniorCitizen", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]


def setup_page():
    st.set_page_config(
        page_title="ChurnVision",
        page_icon="🔭",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    if "lang" not in st.session_state:
        st.session_state.lang = "en"


def toggle_lang():
    st.session_state.lang = "ar" if st.session_state.lang == "en" else "en"


@st.cache_resource
def load_artifacts():
    rf_model = joblib.load('./data/models/random_forest_model.pkl')
    ohe = joblib.load("./data/models/one_hot_encoder.pkl")
    scaler = joblib.load("./data/models/standard_scaler.pkl")
    return rf_model, ohe, scaler


# ------------------------------------------------------------------
# UI Components
# ------------------------------------------------------------------
def render_sidebar(L):
    with st.sidebar:
        st.markdown("### 🔭 ChurnVision")
        st.markdown(f"**{L['sidebar_about_title']}**")
        st.write(L["sidebar_about_body"])
        st.markdown("---")
        st.markdown(f"**{L['sidebar_model_title']}**")
        st.write(L["sidebar_model_body"])
        st.markdown("---")
        st.button(f"🌐 {L['lang_button']}", on_click=toggle_lang, key="lang_toggle_btn")


def render_header(L):
    st.markdown(
        f"""
        <div class="hero">
            <h1>🔭 {L['app_title']}</h1>
            <p>{L['app_subtitle']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_form(L, is_ar):
    col_left, col_right = st.columns(2, gap="large")

    with col_left:
        st.markdown(f'<div class="section-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="section-title">{L["section_customer"]}</div>', unsafe_allow_html=True)
        senior_citizen = st.toggle(L["senior_citizen"], value=False, key="senior")
        partner = st.toggle(L["partner"], value=False, key="partner")
        dependents = st.toggle(L["dependents"], value=False, key="dependents")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(f'<div class="section-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="section-title">{L["section_account"]}</div>', unsafe_allow_html=True)
        tenure = st.slider(L["tenure"], min_value=0, max_value=72, value=12, step=1)
        monthly_charges = st.slider(L["monthly_charges"], min_value=18.0, max_value=120.0, value=65.0, step=0.5)
        phone_service = st.toggle(L["phone_service"], value=True, key="phone")
        paperless_billing = st.toggle(L["paperless_billing"], value=True, key="paperless")

        contract_opts = {L["contract_m2m"]: "Month-to-month", L["contract_1y"]: "One year", L["contract_2y"]: "Two year"}
        contract = contract_opts[st.selectbox(L["contract"], list(contract_opts.keys()))]

        payment_opts = {
            L["pay_electronic"]: "Electronic check", L["pay_mailed"]: "Mailed check",
            L["pay_bank"]: "Bank transfer (automatic)", L["pay_credit"]: "Credit card (automatic)"
        }
        payment_method = payment_opts[st.selectbox(L["payment_method"], list(payment_opts.keys()))]
        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        st.markdown(f'<div class="section-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="section-title">{L["section_services"]}</div>', unsafe_allow_html=True)

        internet_opts = {L["internet_dsl"]: "DSL", L["internet_fiber"]: "Fiber optic", L["internet_none"]: "No"}
        internet_service = internet_opts[st.selectbox(L["internet_service"], list(internet_opts.keys()))]

        st.markdown("<br>", unsafe_allow_html=True)

        if internet_service == "No":
            msg = "الخدمات دي متاحة بس لو العميل مشترك في إنترنت" if is_ar else "Add-on services require an active internet subscription"
            st.caption(f"🔒 {msg}")
            online_security = online_backup = device_protection = tech_support = streaming_tv = streaming_movies = False
        else:
            online_security = st.toggle(L["online_security"], value=False, key="online_security")
            online_backup = st.toggle(L["online_backup"], value=False, key="online_backup")
            device_protection = st.toggle(L["device_protection"], value=False, key="device_protection")
            tech_support = st.toggle(L["tech_support"], value=False, key="tech_support")
            streaming_tv = st.toggle(L["streaming_tv"], value=False, key="streaming_tv")
            streaming_movies = st.toggle(L["streaming_movies"], value=False, key="streaming_movies")

        total_services_count = sum([
            online_security, online_backup, device_protection, tech_support, streaming_tv, streaming_movies
        ])

        st.markdown("<br>", unsafe_allow_html=True)
        st.metric(L["total_services"], total_services_count)
        st.markdown("</div>", unsafe_allow_html=True)

    return {
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalServicesCount": total_services_count,
        "InternetService": internet_service,
        "Contract": contract,
        "PaymentMethod": payment_method,
        "SeniorCitizen": int(senior_citizen),
        "Partner": int(partner),
        "Dependents": int(dependents),
        "PhoneService": int(phone_service),
        "PaperlessBilling": int(paperless_billing),
    }


# ------------------------------------------------------------------
# Prediction Logic
# ------------------------------------------------------------------
def predict_and_display(L, raw_input_dict, rf_model, ohe, scaler):
    predict_clicked = st.button(f"🔮 {L['predict_button']}", use_container_width=True)
    if not predict_clicked:
        return

    raw_input = pd.DataFrame([raw_input_dict])

    # Scale numeric columns
    nums_scaled = scaler.transform(raw_input[NUM_COLS])
    nums_df = pd.DataFrame(nums_scaled, columns=scaler.get_feature_names_out(NUM_COLS), index=raw_input.index)

    # One-hot encode categorical columns
    cat_encoded = ohe.transform(raw_input[CAT_COLS])
    cat_df = pd.DataFrame(cat_encoded, columns=ohe.get_feature_names_out(CAT_COLS), index=raw_input.index)

    # Assemble final feature matrix
    final_input = pd.concat([nums_df, cat_df, raw_input[BINARY_COLS]], axis=1)
    final_input = final_input[rf_model.feature_names_in_]

    churn_proba = float(rf_model.predict_proba(final_input)[0][1])
    stay_proba = 1 - churn_proba

    if churn_proba < 0.35:
        risk_class, risk_label, risk_msg = "risk-low", L["risk_low"], L["risk_low_msg"]
    elif churn_proba < 0.65:
        risk_class, risk_label, risk_msg = "risk-medium", L["risk_medium"], L["risk_medium_msg"]
    else:
        risk_class, risk_label, risk_msg = "risk-high", L["risk_high"], L["risk_high_msg"]

    st.markdown(
        f"""
        <div class="result-card">
            <div class="risk-badge {risk_class}">{risk_label}</div>
            <div style="font-size:2.6rem; font-weight:800; color:var(--accent-light); margin:6px 0;">
                {churn_proba*100:.1f}%
            </div>
            <div style="color:var(--text-secondary);">{L['churn_prob_label']}</div>
            <div class="result-msg">{risk_msg}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    m1, m2 = st.columns(2)
    m1.metric(L["churn_prob_label"], f"{churn_proba*100:.1f}%")
    m2.metric(L["stay_prob_label"], f"{stay_proba*100:.1f}%")
    st.progress(churn_proba)


# ------------------------------------------------------------------
# Main Application
# ------------------------------------------------------------------
def main():
    setup_page()

    L = TXT[st.session_state.lang]
    is_ar = (st.session_state.lang == "ar")
    dir_str = "rtl" if is_ar else "ltr"
    align_str = "right" if is_ar else "left"

    st.markdown(get_css(is_ar, dir_str, align_str), unsafe_allow_html=True)

    try:
        rf_model, ohe, scaler = load_artifacts()
    except Exception as e:
        st.error(f"Could not load model files. Details: {e}")
        st.stop()

    render_sidebar(L)
    render_header(L)

    raw_input_dict = render_form(L, is_ar)
    predict_and_display(L, raw_input_dict, rf_model, ohe, scaler)

    st.markdown(f'<div class="footer-note">{L["footer"]}</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
