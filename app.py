# ==========================================
# IMPORTS
# ==========================================

from pathlib import Path
from datetime import datetime

import streamlit as st

from utils import predict_news


# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ==========================================
# LOAD CSS
# ==========================================

def load_css(css_file):
    css_path = Path(css_file)

    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(
                f"<style>{f.read()}</style>",
                unsafe_allow_html=True,
            )


load_css("assets/style.css")


# ==========================================
# HEADER
# ==========================================

today = datetime.now()

st.markdown(
    f"""
<div style="text-align:center;">

<h1>
📰 Daily News Dashboard
</h1>

<h3>
Top Headlines Today
</h3>

<p style="
font-size:18px;
font-weight:600;
color:#444;
margin-top:-8px;">
{today.strftime("%A, %d %B %Y")}
</p>

</div>
""",
    unsafe_allow_html=True,
)

st.markdown("<hr>", unsafe_allow_html=True)


# ==========================================
# BREAKING NEWS
# ==========================================

breaking_news = (
    "AI breakthrough improves fake news detection accuracy "
    "• Global markets react to new economic policies "
    "• Tech giants race toward next-generation AI systems "
    "• Climate summit reaches historic agreement "
    "• Sports finals enter exciting final stage "
    "• Cybersecurity agencies warn against AI-generated fake news "
)

st.markdown(
    f"""
<div class="breaking-container">

<div class="breaking-label">
🔴 LIVE
</div>

<div class="breaking-ticker">

{breaking_news}

</div>

</div>
""",
    unsafe_allow_html=True,
)


# ==========================================
# FEATURED NEWS
# ==========================================

featured_news = [

    {
        "category": "POLITICS",
        "title": "Election results shake political landscape",
        "description":
        "Major updates are coming in from the national election with unexpected outcomes across several key regions."
    },

    {
        "category": "TECH",
        "title": "AI revolution is transforming industries",
        "description":
        "Companies worldwide continue adopting AI solutions to improve efficiency and productivity."
    },

    {
        "category": "WORLD",
        "title": "Global markets show mixed trends",
        "description":
        "Financial markets fluctuate as investors respond to changing economic indicators around the world."
    }

]

for news in featured_news:

    st.markdown(
        f"""
<div class="news-card">

<div class="tag">
{news["category"]}
</div>

<h3>
{news["title"]}
</h3>

<p>
{news["description"]}
</p>

</div>
""",
        unsafe_allow_html=True,
    )


# ==========================================
# MAIN LAYOUT
# ==========================================

left, right = st.columns([2.3, 1])


# ==========================================
# LEFT PANEL
# ==========================================

with left:

    if "news_text" not in st.session_state:
        st.session_state.news_text = ""

    st.markdown(
        """
<h2 style="
font-size:34px;
font-weight:800;
color:#0b1f3a;
margin-bottom:10px;">
🧠 Analyze News Article
</h2>
""",
        unsafe_allow_html=True,
    )

    news_text = st.text_area(
        "",
        key="news_text",
        height=350,
        placeholder="Paste any news article here...",
        label_visibility="collapsed",
    )

    col1, col2 = st.columns(2)

    with col1:

        detect = st.button(
            "🚀 Detect News",
            use_container_width=True,
        )

    with col2:

        clear = st.button(
            "🗑 Clear",
            use_container_width=True,
        )

    if clear:

        st.session_state.news_text = ""

        st.rerun()

# ==========================================
# RIGHT PANEL
# ==========================================

with right:

    st.markdown(
        """
<h2 style="
font-size:30px;
font-weight:800;
color:#0b1f3a;
margin-bottom:12px;">
📌 Features
</h2>
""",
        unsafe_allow_html=True,
    )

    st.markdown("""
<div style="
background:rgba(255,255,255,.92);
padding:18px;
border-radius:15px;
border:1px solid rgba(0,0,0,.08);
box-shadow:0 6px 14px rgba(0,0,0,.08);
font-size:18px;
font-weight:700;
line-height:2;
color:#111;">

✅ NLP Text Cleaning<br>

✅ TF-IDF Vectorizer<br>

✅ Ensemble Machine Learning<br>

✅ Confidence Score<br>

✅ Trusted News Verification<br>

✅ Professional Dashboard

</div>
""", unsafe_allow_html=True)

    st.markdown(
        """
<h2 style="
font-size:30px;
font-weight:800;
color:#0b1f3a;
margin-top:20px;
margin-bottom:12px;">
📊 Model
</h2>
""",
        unsafe_allow_html=True,
    )

    st.markdown("""
<div style="
background:rgba(255,255,255,.92);
padding:18px;
border-radius:15px;
border:1px solid rgba(0,0,0,.08);
font-size:18px;
font-weight:700;
color:#111;
">

🤖 Ensemble Machine Learning Classifier

</div>
""", unsafe_allow_html=True)

    st.markdown(
        """
<h2 style="
font-size:30px;
font-weight:800;
color:#0b1f3a;
margin-top:20px;
margin-bottom:12px;">
💡 Tip
</h2>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
<div style="
background:rgba(255,255,255,.88);
padding:16px;
border-radius:14px;
border:1px solid rgba(0,0,0,.08);
font-size:17px;
font-weight:600;
line-height:1.7;
color:#111;">
Paste a complete news article instead of a short sentence to obtain more reliable predictions.
</div>
""",
        unsafe_allow_html=True,
    )


# ==========================================
# PREDICTION
# ==========================================

if detect:

    if news_text.strip() == "":

        st.warning("⚠ Please enter a news article.")

    else:

        with st.spinner("🔍 Analyzing article..."):

            prediction, confidence = predict_news(news_text)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(
            """
<h2 style="
font-size:34px;
font-weight:800;
color:#0b1f3a;">
📰 Detection Result
</h2>
""",
            unsafe_allow_html=True,
        )

        if prediction == 1:

            result_color = "#d71920"
            result_icon = "🚨"
            result_text = "FAKE NEWS DETECTED"

        else:

            result_color = "#16a34a"
            result_icon = "✅"
            result_text = "REAL NEWS DETECTED"

        st.markdown(
            f"""
<div style="
background:rgba(255,255,255,.90);
padding:22px;
border-left:8px solid {result_color};
border-radius:16px;
box-shadow:0 6px 18px rgba(0,0,0,.08);
margin-bottom:18px;">

<h2 style="
margin:0;
color:{result_color};
font-size:32px;
font-weight:800;">

{result_icon} {result_text}

</h2>

</div>
""",
            unsafe_allow_html=True,
        )

        st.metric(
            label="Confidence Score",
            value=f"{confidence:.2f}%"
        )

        st.progress(min(int(confidence), 100))

        st.markdown(
            """
<h2 style="
font-size:30px;
font-weight:800;
color:#0b1f3a;
margin-top:25px;">
📄 Submitted Article
</h2>
""",
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
<div style="
background:rgba(255,255,255,.85);
padding:20px;
border-radius:16px;
border:1px solid rgba(0,0,0,.08);
box-shadow:0 4px 12px rgba(0,0,0,.08);
font-size:17px;
line-height:1.8;
color:#111;
white-space:pre-wrap;">

{news_text}

</div>
""",
            unsafe_allow_html=True,
        )

# ==========================================
# FOOTER
# ==========================================

st.markdown("<br><br>", unsafe_allow_html=True)

st.markdown(
    """
<hr style="margin-top:40px;margin-bottom:20px;">

<div style="
text-align:center;
padding:20px;
">

<h3 style="
color:#0b1f3a;
font-size:24px;
font-weight:800;
margin-bottom:10px;">
📰 Fake News Detection Dashboard
</h3>

<p style="
font-size:17px;
color:#333;
font-weight:600;
line-height:1.8;">

Developed using ❤️ with
<b>Streamlit</b>,
<b>Machine Learning</b>,
<b>NLP</b>,
<b>TF-IDF Vectorization</b>

</p>

<p style="
font-size:15px;
color:#666;">

© 2026 Pralay Bajkhan • All Rights Reserved

</p>

</div>
""",
    unsafe_allow_html=True,
)


# ==========================================
# SIDEBAR (OPTIONAL)
# ==========================================

with st.sidebar:

    st.markdown("## 📰 Dashboard")

    st.write("Fake News Detection using Machine Learning")

    st.markdown("---")

    st.write("### 🔧 Technologies")

    st.write("• Python")

    st.write("• Streamlit")

    st.write("• Scikit-learn")

    st.write("• TF-IDF Vectorizer")

    st.write("• NLP")

    st.write("• Ensemble Learning")

    st.markdown("---")

    st.info(
        "This project predicts whether a news article is REAL or FAKE using Natural Language Processing and Machine Learning."
    )


# ==========================================
# END OF APP
# ==========================================                