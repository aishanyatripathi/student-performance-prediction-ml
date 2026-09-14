def get_custom_css() -> str:
    """Returns production enterprise SaaS CSS inspired by Stripe, Linear, and Vercel."""
    return """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --bg-main: #F8FAFC;
        --bg-card: #FFFFFF;
        --text-primary: #0F172A;
        --text-secondary: #475569;
        --text-muted: #64748B;
        --border-color: #E2E8F0;
        --border-color-hover: #CBD5E1;
        --accent: #4F46E5;
        --accent-hover: #4338CA;
        --accent-light: #EEF2FF;
        --accent-border: #C7D2FE;
        --radius-sm: 8px;
        --radius-md: 12px;
        --radius-lg: 16px;
        --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -2px rgba(0, 0, 0, 0.05);
    }

    /* Global reset & typography */
    .stApp {
        background-color: var(--bg-main);
        color: var(--text-primary);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        -webkit-font-smoothing: antialiased;
    }

    .main .block-container {
        padding-top: 2.5rem;
        padding-bottom: 4rem;
        max-width: 1200px;
    }

    /* Typography Hierarchy */
    h1, .page-title {
        font-size: 32px !important;
        font-weight: 800 !important;
        color: var(--text-primary) !important;
        letter-spacing: -0.025em !important;
        line-height: 1.25 !important;
        margin-bottom: 8px !important;
    }

    h2, .section-title {
        font-size: 24px !important;
        font-weight: 700 !important;
        color: var(--text-primary) !important;
        letter-spacing: -0.02em !important;
        line-height: 1.3 !important;
        margin-bottom: 16px !important;
    }

    h3, .card-title {
        font-size: 18px !important;
        font-weight: 600 !important;
        color: var(--text-primary) !important;
        letter-spacing: -0.01em !important;
        line-height: 1.4 !important;
    }

    p, body {
        font-size: 15px;
        color: var(--text-secondary);
        line-height: 1.6;
    }

    .caption-text {
        font-size: 13px;
        color: var(--text-muted);
        font-weight: 500;
    }

    /* Header Container (Linear/Vercel Aesthetic) */
    .app-header {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: var(--radius-lg);
        padding: 24px 32px;
        margin-bottom: 32px;
        box-shadow: var(--shadow-sm);
    }

    .app-header-top {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 8px;
    }

    .header-tag {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background-color: var(--accent-light);
        color: var(--accent);
        border: 1px solid var(--accent-border);
        border-radius: 6px;
        padding: 4px 10px;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }

    .header-sub {
        color: var(--text-secondary);
        font-size: 15px;
        font-weight: 400;
        margin: 0;
    }

    /* Compact Stat Metric Cards */
    .metric-card {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: var(--radius-md);
        padding: 20px 24px;
        box-shadow: var(--shadow-sm);
        transition: border-color 0.15s ease, box-shadow 0.15s ease;
    }

    .metric-card:hover {
        border-color: var(--border-color-hover);
        box-shadow: var(--shadow-md);
    }

    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: var(--text-primary);
        letter-spacing: -0.02em;
        line-height: 1.2;
    }

    .metric-label {
        font-size: 13px;
        font-weight: 600;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-top: 6px;
    }

    /* Risk Status Badges */
    .risk-badge {
        display: inline-flex;
        align-items: center;
        padding: 4px 12px;
        border-radius: 6px;
        font-size: 13px;
        font-weight: 600;
    }

    .risk-critical {
        background-color: #FEF2F2;
        color: #991B1B;
        border: 1px solid #FCA5A5;
    }

    .risk-high {
        background-color: #FFF7ED;
        color: #9A3412;
        border: 1px solid #FDBA74;
    }

    .risk-medium {
        background-color: #FEFCE8;
        color: #854D0E;
        border: 1px solid #FDE047;
    }

    .risk-low {
        background-color: #F0FDF4;
        color: #166534;
        border: 1px solid #86EFAC;
    }

    /* Recommendation Items */
    .recommendation-item {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-left: 3px solid var(--accent);
        border-radius: var(--radius-sm);
        padding: 16px 20px;
        margin-bottom: 12px;
    }

    .rec-category {
        font-weight: 600;
        color: var(--accent);
        font-size: 14px;
    }

    .rec-action {
        color: var(--text-primary);
        font-size: 14px;
        margin-top: 4px;
        line-height: 1.5;
    }

    /* Navigation Tabs (Linear/Vercel Underline Tab Bar) */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        padding: 0;
        border-bottom: 1px solid var(--border-color) !important;
        margin-bottom: 32px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 40px;
        padding: 0 4px;
        background-color: transparent !important;
        border: none !important;
        border-radius: 0 !important;
        color: var(--text-muted) !important;
        font-weight: 500 !important;
        font-size: 15px !important;
        box-shadow: none !important;
        transition: color 0.15s ease !important;
    }

    .stTabs [data-baseweb="tab"]:hover {
        color: var(--text-primary) !important;
    }

    .stTabs [aria-selected="true"] {
        color: var(--accent) !important;
        font-weight: 600 !important;
        border-bottom: 2px solid var(--accent) !important;
        background-color: transparent !important;
    }

    /* Form & Container Layout */
    .stForm {
        background: var(--bg-card);
        border: 1px solid var(--border-color) !important;
        border-radius: var(--radius-lg) !important;
        padding: 32px !important;
        box-shadow: var(--shadow-sm) !important;
    }

    /* High-contrast Form Submit & Action Buttons */
    .stFormSubmitButton > button,
    button[kind="primaryFormSubmit"],
    button[data-testid="baseButton-primaryFormSubmit"],
    .stButton > button {
        background-color: var(--accent) !important;
        color: #FFFFFF !important;
        border-radius: var(--radius-sm) !important;
        border: 1px solid var(--accent) !important;
        font-weight: 700 !important;
        font-size: 15px !important;
        padding: 10px 20px !important;
        box-shadow: var(--shadow-sm) !important;
        transition: all 0.15s ease !important;
        height: 44px !important;
        width: 100% !important;
    }

    .stFormSubmitButton > button *,
    button[kind="primaryFormSubmit"] *,
    button[data-testid="baseButton-primaryFormSubmit"] *,
    .stButton > button * {
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }

    .stFormSubmitButton > button:hover,
    button[kind="primaryFormSubmit"]:hover,
    button[data-testid="baseButton-primaryFormSubmit"]:hover,
    .stButton > button:hover {
        background-color: var(--accent-hover) !important;
        border-color: var(--accent-hover) !important;
        color: #FFFFFF !important;
    }

    .stDownloadButton > button {
        background-color: #FFFFFF !important;
        color: var(--text-primary) !important;
        border-radius: var(--radius-sm) !important;
        border: 1px solid var(--border-color) !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        padding: 10px 20px !important;
        box-shadow: var(--shadow-sm) !important;
        transition: all 0.15s ease !important;
        height: 40px !important;
    }

    .stDownloadButton > button:hover {
        background-color: var(--bg-main) !important;
        border-color: var(--border-color-hover) !important;
        color: var(--text-primary) !important;
    }

    /* Inputs, Number Inputs, Selectboxes & Values (White Text for High Contrast) */
    .stTextInput input, 
    .stNumberInput input, 
    [data-testid="stNumberInput"] input,
    input[type="number"],
    .stNumberInput button *,
    .stNumberInput span,
    .stSelectbox select, 
    [data-baseweb="select"] *,
    [data-testid="stSelectbox"] *,
    div[data-baseweb="select"] *,
    div[data-baseweb="select"] p,
    div[data-baseweb="select"] span {
        color: #FFFFFF !important;
        font-size: 14px !important;
        font-weight: 600 !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    [data-testid="stWidgetLabel"],
    [data-testid="stWidgetLabel"] label, 
    [data-testid="stWidgetLabel"] p, 
    .stSlider label, 
    .stNumberInput label, 
    .stSelectbox label {
        color: var(--text-primary) !important;
        font-size: 14px !important;
        font-weight: 600 !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    .stDataFrame {
        border: 1px solid var(--border-color);
        border-radius: var(--radius-md);
    }



    /* Alert Callouts */
    .stAlert {
        border-radius: var(--radius-md) !important;
        border: 1px solid var(--border-color) !important;
        background-color: var(--bg-card) !important;
    }

    /* Footer */
    .app-footer {
        text-align: center;
        color: var(--text-muted);
        font-size: 13px;
        padding: 32px 0 16px 0;
        border-top: 1px solid var(--border-color);
        margin-top: 48px;
    }
    </style>
    """
