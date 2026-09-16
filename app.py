"""
ChurnVision — Customer Churn Prediction App
Built on top of the trained Random Forest pipeline from the ChurnVision project
(https://github.com/M7lawy/ChurnVision).

Run with:
    streamlit run app.py
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ------------------------------------------------------------------
# Page config (must be the first Streamlit call)
# ------------------------------------------------------------------
st.set_page_config(
    page_title="ChurnVision",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# ------------------------------------------------------------------
# Translations
# ------------------------------------------------------------------
TXT = {
    "ar": {
        "app_title": "ChurnVision",
        "app_subtitle": "توقع احتمالية فقدان العميل باستخدام الذكاء الاصطناعي",
        "lang_button": "English",
        "sidebar_about_title": "عن المشروع",
        "sidebar_about_body": (
            "ChurnVision بيستخدم نموذج Random Forest اتدرب على بيانات شركة اتصالات "
            "لتوقع احتمالية ترك العميل للخدمة (Churn)، بناءً على بيانات العقد والخدمات "
            "والفواتير الخاصة بيه."
        ),
        "sidebar_model_title": "الموديل المستخدم",
        "sidebar_model_body": "Random Forest Classifier",
        "section_customer": "بيانات العميل",
        "section_account": "بيانات الحساب والعقد",
        "section_services": "الخدمات المشترك فيها",
        "senior_citizen": "عميل كبير السن (65+)",
        "partner": "لديه شريك/زوج",
        "dependents": "لديه معالين (أطفال/أفراد أسرة)",
        "tenure": "عدد شهور الاشتراك",
        "monthly_charges": "الفاتورة الشهرية ($)",
        "phone_service": "مشترك في خدمة الهاتف",
        "paperless_billing": "فاتورة إلكترونية (بدون ورق)",
        "internet_service": "نوع خدمة الإنترنت",
        "contract": "نوع العقد",
        "payment_method": "طريقة الدفع",
        "online_security": "حماية أونلاين (Online Security)",
        "online_backup": "نسخ احتياطي أونلاين (Online Backup)",
        "device_protection": "حماية الأجهزة (Device Protection)",
        "tech_support": "دعم فني (Tech Support)",
        "streaming_tv": "بث تلفزيوني (Streaming TV)",
        "streaming_movies": "بث أفلام (Streaming Movies)",
        "predict_button": "توقع الآن",
        "result_title": "نتيجة التوقع",
        "churn_prob_label": "احتمالية ترك الخدمة",
        "risk_low": "خطورة منخفضة",
        "risk_medium": "خطورة متوسطة",
        "risk_high": "خطورة عالية",
        "risk_low_msg": "العميل مستقر غالبًا — احتمالية تركه للخدمة ضعيفة.",
        "risk_medium_msg": "في إشارات تستحق المتابعة — ينصح بالتواصل مع العميل ومراجعة عرضه.",
        "risk_high_msg": "العميل معرض بشدة لترك الخدمة — ينصح بالتدخل فورًا (عرض خاص، تواصل مباشر).",
        "stay_prob_label": "احتمالية الاستمرار",
        "internet_dsl": "DSL",
        "internet_fiber": "فايبر بصري (Fiber optic)",
        "internet_none": "بدون إنترنت",
        "contract_m2m": "شهري (Month-to-month)",
        "contract_1y": "سنة واحدة",
        "contract_2y": "سنتين",
        "pay_bank": "تحويل بنكي (تلقائي)",
        "pay_credit": "بطاقة ائتمان (تلقائي)",
        "pay_electronic": "شيك إلكتروني",
        "pay_mailed": "شيك بالبريد",
        "yes": "نعم",
        "no": "لا",
        "footer": "مبني فوق نموذج Random Forest من مشروع ChurnVision",
        "total_services": "إجمالي الخدمات الإضافية المفعّلة",
    },
    "en": {
        "app_title": "ChurnVision",
        "app_subtitle": "AI-powered customer churn risk prediction",
        "lang_button": "عربي",
        "sidebar_about_title": "About",
        "sidebar_about_body": (
            "ChurnVision uses a Random Forest model trained on telecom customer data "
            "to predict the probability that a customer will churn, based on their "
            "contract, services and billing information."
        ),
        "sidebar_model_title": "Model in use",
        "sidebar_model_body": "Random Forest Classifier",
        "section_customer": "Customer Info",
        "section_account": "Account & Contract",
        "section_services": "Subscribed Services",
        "senior_citizen": "Senior citizen (65+)",
        "partner": "Has a partner",
        "dependents": "Has dependents",
        "tenure": "Tenure (months)",
        "monthly_charges": "Monthly charges ($)",
        "phone_service": "Has phone service",
        "paperless_billing": "Paperless billing",
        "internet_service": "Internet service type",
        "contract": "Contract type",
        "payment_method": "Payment method",
        "online_security": "Online Security",
        "online_backup": "Online Backup",
        "device_protection": "Device Protection",
        "tech_support": "Tech Support",
        "streaming_tv": "Streaming TV",
        "streaming_movies": "Streaming Movies",
        "predict_button": "Predict Now",
        "result_title": "Prediction Result",
        "churn_prob_label": "Churn probability",
        "risk_low": "Low risk",
        "risk_medium": "Medium risk",
        "risk_high": "High risk",
        "risk_low_msg": "This customer looks stable — low probability of churning.",
        "risk_medium_msg": "Some warning signs — worth reaching out and reviewing their offer.",
        "risk_high_msg": "High churn risk — recommend immediate action (retention offer, direct contact).",
        "stay_prob_label": "Retention probability",
        "internet_dsl": "DSL",
        "internet_fiber": "Fiber optic",
        "internet_none": "No internet",
        "contract_m2m": "Month-to-month",
        "contract_1y": "One year",
        "contract_2y": "Two year",
        "pay_bank": "Bank transfer (automatic)",
        "pay_credit": "Credit card (automatic)",
        "pay_electronic": "Electronic check",
        "pay_mailed": "Mailed check",
        "yes": "Yes",
        "no": "No",
        "footer": "Built on the Random Forest model from the ChurnVision project",
        "total_services": "Total active add-on services",
    },
}

# ------------------------------------------------------------------
# Session state — language toggle
# ------------------------------------------------------------------
if "lang" not in st.session_state:
    st.session_state.lang = "ar"


def toggle_lang():
    st.session_state.lang = "en" if st.session_state.lang == "ar" else "ar"


L = TXT[st.session_state.lang]
IS_AR = st.session_state.lang == "ar"
DIR = "rtl" if IS_AR else "ltr"
ALIGN = "right" if IS_AR else "left"

# ------------------------------------------------------------------
# Theme — professional blue / dark
# ------------------------------------------------------------------
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;800&family=Inter:wght@400;600;700;800&display=swap');

    :root {{
        --bg-primary: #0b1220;
        --bg-secondary: #111a2e;
        --card-bg: #16213a;
        --card-border: #223353;
        --accent: #3b82f6;
        --accent-light: #60a5fa;
        --accent-dark: #1d4ed8;
        --text-primary: #e8edf7;
        --text-secondary: #93a3c2;
        --success: #22c55e;
        --warning: #f59e0b;
        --danger: #ef4444;
    }}

    html, body, [class*="css"] {{
        font-family: {"'Cairo', sans-serif" if IS_AR else "'Inter', sans-serif"};
    }}

    .stApp {{
        background: radial-gradient(circle at top left, var(--bg-secondary), var(--bg-primary) 60%);
        color: var(--text-primary);
        direction: {DIR};
    }}

    section[data-testid="stSidebar"] {{
        background: var(--bg-secondary);
        border-{"left" if IS_AR else "right"}: 1px solid var(--card-border);
    }}

    h1, h2, h3, h4, p, label, span, div {{
        text-align: {ALIGN};
    }}

    .hero {{
        background: linear-gradient(135deg, #14213d 0%, #0b1220 100%);
        border: 1px solid var(--card-border);
        border-radius: 18px;
        padding: 28px 32px;
        margin-bottom: 24px;
        box-shadow: 0 8px 30px rgba(0,0,0,0.35);
    }}
    .hero h1 {{
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        background: linear-gradient(90deg, var(--accent-light), #93c5fd);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
    .hero p {{
        color: var(--text-secondary);
        margin-top: 6px;
        font-size: 1.05rem;
    }}

    .section-card {{
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        border-radius: 16px;
        padding: 22px 24px;
        margin-bottom: 20px;
    }}
    .section-title {{
        font-size: 1.15rem;
        font-weight: 700;
        color: var(--accent-light);
        margin-bottom: 14px;
        padding-bottom: 10px;
        border-bottom: 1px solid var(--card-border);
    }}

    div[data-testid="stMetric"] {{
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        border-radius: 14px;
        padding: 14px 18px;
    }}

    .stButton > button {{
        background: linear-gradient(90deg, var(--accent-dark), var(--accent));
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.7rem 1.6rem;
        font-weight: 700;
        font-size: 1.05rem;
        width: 100%;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
        box-shadow: 0 4px 18px rgba(59,130,246,0.35);
    }}
    .stButton > button:hover {{
        transform: translateY(-1px);
        box-shadow: 0 6px 22px rgba(59,130,246,0.5);
    }}

    .lang-toggle > button {{
        background: transparent !important;
        color: var(--accent-light) !important;
        border: 1px solid var(--card-border) !important;
        box-shadow: none !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        padding: 0.35rem 1rem !important;
        width: auto !important;
    }}

    .risk-badge {{
        display: inline-block;
        padding: 6px 18px;
        border-radius: 999px;
        font-weight: 800;
        font-size: 1rem;
        margin-bottom: 10px;
    }}
    .risk-low {{ background: rgba(34,197,94,0.15); color: var(--success); border: 1px solid var(--success); }}
    .risk-medium {{ background: rgba(245,158,11,0.15); color: var(--warning); border: 1px solid var(--warning); }}
    .risk-high {{ background: rgba(239,68,68,0.15); color: var(--danger); border: 1px solid var(--danger); }}

    .result-card {{
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        border-radius: 18px;
        padding: 26px 28px;
        text-align: center;
        margin-top: 10px;
    }}
    .result-msg {{
        color: var(--text-secondary);
        font-size: 1rem;
        margin-top: 8px;
    }}

    footer {{visibility: hidden;}}
    #MainMenu {{visibility: hidden;}}

    hr {{ border-color: var(--card-border); }}

    .footer-note {{
        text-align: center;
        color: var(--text-secondary);
        font-size: 0.85rem;
        margin-top: 30px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# Load model artifacts (cached)
# ------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    rf_model = joblib.load('./data/models/random_forest_model.pkl')
    ohe = joblib.load("./data/models/one_hot_encoder.pkl")
    scaler = joblib.load("./data/models/standard_scaler.pkl")
    return rf_model, ohe, scaler


try:
    rf_model, ohe, scaler = load_artifacts()
    ARTIFACTS_OK = True
except Exception as e:
    ARTIFACTS_OK = False
    LOAD_ERROR = str(e)

CAT_COLS = ["InternetService", "Contract", "PaymentMethod"]
NUM_COLS = ["tenure", "MonthlyCharges", "TotalServicesCount"]
BINARY_COLS = ["SeniorCitizen", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]

# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown(f"### 🔭 ChurnVision")
    st.markdown(f"**{L['sidebar_about_title']}**")
    st.write(L["sidebar_about_body"])
    st.markdown("---")
    st.markdown(f"**{L['sidebar_model_title']}**")
    st.write(L["sidebar_model_body"])
    st.markdown("---")
    st.button(f"🌐 {L['lang_button']}", on_click=toggle_lang, key="lang_toggle_btn")

# ------------------------------------------------------------------
# Hero header
# ------------------------------------------------------------------
st.markdown(
    f"""
    <div class="hero">
        <h1>🔭 {L['app_title']}</h1>
        <p>{L['app_subtitle']}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not ARTIFACTS_OK:
    st.error(f"Could not load model files from '{MODELS_DIR}'. Details: {LOAD_ERROR}")
    st.stop()

# ------------------------------------------------------------------
# Input form
# ------------------------------------------------------------------
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

    contract_options = {
        L["contract_m2m"]: "Month-to-month",
        L["contract_1y"]: "One year",
        L["contract_2y"]: "Two year",
    }
    contract_label = st.selectbox(L["contract"], list(contract_options.keys()))
    contract = contract_options[contract_label]

    payment_options = {
        L["pay_electronic"]: "Electronic check",
        L["pay_mailed"]: "Mailed check",
        L["pay_bank"]: "Bank transfer (automatic)",
        L["pay_credit"]: "Credit card (automatic)",
    }
    payment_label = st.selectbox(L["payment_method"], list(payment_options.keys()))
    payment_method = payment_options[payment_label]

    st.markdown("</div>", unsafe_allow_html=True)

with col_right:
    st.markdown(f'<div class="section-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="section-title">{L["section_services"]}</div>', unsafe_allow_html=True)

    internet_options = {
        L["internet_dsl"]: "DSL",
        L["internet_fiber"]: "Fiber optic",
        L["internet_none"]: "No",
    }
    internet_label = st.selectbox(L["internet_service"], list(internet_options.keys()))
    internet_service = internet_options[internet_label]

    st.markdown("<br>", unsafe_allow_html=True)

    if internet_service == "No":
        st.caption(
            "🔒 " + (
                "الخدمات دي متاحة بس لو العميل مشترك في إنترنت"
                if IS_AR else
                "Add-on services require an active internet subscription"
            )
        )
        online_security = online_backup = device_protection = False
        tech_support = streaming_tv = streaming_movies = False
    else:
        online_security = st.toggle(L["online_security"], value=False, key="online_security")
        online_backup = st.toggle(L["online_backup"], value=False, key="online_backup")
        device_protection = st.toggle(L["device_protection"], value=False, key="device_protection")
        tech_support = st.toggle(L["tech_support"], value=False, key="tech_support")
        streaming_tv = st.toggle(L["streaming_tv"], value=False, key="streaming_tv")
        streaming_movies = st.toggle(L["streaming_movies"], value=False, key="streaming_movies")

    total_services_count = sum([
        online_security, online_backup, device_protection,
        tech_support, streaming_tv, streaming_movies,
    ])

    st.markdown("<br>", unsafe_allow_html=True)
    st.metric(L["total_services"], total_services_count)

    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------------------------
# Predict
# ------------------------------------------------------------------
predict_clicked = st.button(f"🔮 {L['predict_button']}", use_container_width=True)

if predict_clicked:
    raw_input = pd.DataFrame([{
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
    }])

    # Scale numeric columns
    nums_scaled = scaler.transform(raw_input[NUM_COLS])
    nums_df = pd.DataFrame(nums_scaled, columns=scaler.get_feature_names_out(NUM_COLS), index=raw_input.index)

    # One-hot encode categorical columns
    cat_encoded = ohe.transform(raw_input[CAT_COLS])
    cat_df = pd.DataFrame(cat_encoded, columns=ohe.get_feature_names_out(CAT_COLS), index=raw_input.index)

    # Assemble final feature matrix in the exact order the model was trained on
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

st.markdown(f'<div class="footer-note">{L["footer"]}</div>', unsafe_allow_html=True)
