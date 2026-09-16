def get_css(is_ar, dir_str, align_str):
    return f"""
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
        font-family: {"'Cairo', sans-serif" if is_ar else "'Inter', sans-serif"};
    }}

    .stApp {{
        background: radial-gradient(circle at top left, var(--bg-secondary), var(--bg-primary) 60%);
        color: var(--text-primary);
        direction: {dir_str};
    }}

    section[data-testid="stSidebar"] {{
        background: var(--bg-secondary);
        border-{"left" if is_ar else "right"}: 1px solid var(--card-border);
    }}

    h1, h2, h3, h4, p, label, span, div {{
        text-align: {align_str};
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
    """

