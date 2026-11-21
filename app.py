
import streamlit as st
import pandas as pd
import plotly.express as px
import re
from datetime import datetime

from classifier import train_model, predict_category, load_data


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Cynbit AI Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================
# CUSTOM CSS - BLACK PREMIUM THEME
# ==========================================

st.markdown("""
<style>

.stApp {
    background-color: #0B0B10;
    color: #F8FAFC;
}

[data-testid="stSidebar"] {
    background-color: #111118;
    border-right: 1px solid #292936;
}

.main-title {
    font-size: 34px;
    font-weight: 800;
    color: #FFFFFF;
}

.subtitle {
    color: #A1A1AA;
    font-size: 15px;
}

.kpi-card {
    background: linear-gradient(135deg, #191923, #242033);
    border: 1px solid #383347;
    padding: 22px;
    border-radius: 14px;
    min-height: 125px;
}

.kpi-label {
    color: #A1A1AA;
    font-size: 13px;
}

.kpi-value {
    color: #FFFFFF;
    font-size: 29px;
    font-weight: 800;
    margin-top: 8px;
}

.section-card {
    background-color: #15151E;
    border: 1px solid #292936;
    padding: 20px;
    border-radius: 14px;
}

div.stButton > button {
    background: linear-gradient(90deg, #7C3AED, #9333EA);
    color: white;
    border: none;
    border-radius: 9px;
    font-weight: 600;
    min-height: 45px;
}

div.stButton > button:hover {
    background: #6D28D9;
    color: white;
    border: none;
}

h1, h2, h3 {
    color: #FFFFFF;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# LOAD MACHINE LEARNING MODEL
# ==========================================

@st.cache_resource
def load_ml_model():
    return train_model()


model = load_ml_model()


# ==========================================
# SESSION STATE
# ==========================================

if "history" not in st.session_state:
    st.session_state.history = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None


# ==========================================
# DATASET
# ==========================================

X, y = load_data()

category_counts = (
    y.value_counts()
    .rename_axis("Category")
    .reset_index(name="Count")
)


# ==========================================
# REQUIREMENT EXTRACTION
# ==========================================

def extract_requirements(message):

    keywords = [
        "website",
        "e-commerce",
        "payment",
        "dashboard",
        "login",
        "chatbot",
        "automation",
        "email",
        "document",
        "android",
        "ios",
        "mobile app",
        "booking",
        "cloud",
        "hosting",
        "server",
        "AWS"
    ]

    return [
        word for word in keywords
        if re.search(
            r"\b" + re.escape(word) + r"\b",
            message,
            re.IGNORECASE
        )
    ]


# ==========================================
# LEAD PRIORITY
# ==========================================

def get_priority(message, requirements):

    high_intent_words = [
        "urgent",
        "as soon as possible",
        "immediately",
        "payment",
        "budget",
        "ready to start"
    ]

    if any(
        word in message.lower()
        for word in high_intent_words
    ):
        return "High"

    elif len(requirements) >= 3:
        return "Medium"

    return "Low"


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.markdown("## 🤖 CY NBIT AI")

    st.caption("INTELLIGENT PROJECT DISCOVERY")

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Dashboard",
            "AI Inquiry Analyzer",
            "Prediction History",
            "About Project"
        ]
    )

    st.divider()

    st.markdown("### Model Status")

    st.success("ML Model Loaded")

    st.caption("TF-IDF + Logistic Regression")

    st.divider()

    st.caption("Cynbit Technologies")
    st.caption("AI/ML Assessment Project")


# ==========================================
# HEADER
# ==========================================

st.markdown(
    '<div class="main-title">Cynbit AI Dashboard</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Intelligent Customer Inquiry Analysis & Project Discovery'
    '</div>',
    unsafe_allow_html=True
)

st.write("")


# ==========================================
# DASHBOARD OVERVIEW
# ==========================================

if page == "Dashboard":

    st.markdown("## 📊 Dashboard Overview")

    total = len(st.session_state.history)

    accuracy = getattr(model, "test_accuracy_", None)

    accuracy_text = (
        f"{accuracy:.1%}" if accuracy is not None else "N/A"
    )

    k1, k2, k3, k4 = st.columns(4)

    with k1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Total Inquiries</div>
            <div class="kpi-value">{total}</div>
        </div>
        """, unsafe_allow_html=True)

    with k2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Predictions Made</div>
            <div class="kpi-value">{total}</div>
        </div>
        """, unsafe_allow_html=True)

    with k3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Service Categories</div>
            <div class="kpi-value">{y.nunique()}</div>
        </div>
        """, unsafe_allow_html=True)

    with k4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Test Accuracy</div>
            <div class="kpi-value">{accuracy_text}</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.divider()

    st.markdown("## 📈 Analytics")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### Dataset Category Distribution")

        fig = px.pie(
            category_counts,
            names="Category",
            values="Count",
            hole=0.55,
            template="plotly_dark",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )

        fig.update_layout(
            paper_bgcolor="#15151E",
            plot_bgcolor="#15151E",
            font_color="white"
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:

        st.markdown("### Prediction History")

        if st.session_state.history:

            history_df = pd.DataFrame(
                st.session_state.history
            )

            chart_data = (
                history_df["Category"]
                .value_counts()
                .reset_index()
            )

            chart_data.columns = ["Category", "Predictions"]

            fig2 = px.bar(
                chart_data,
                x="Category",
                y="Predictions",
                color="Category",
                template="plotly_dark",
                text="Predictions"
            )

            fig2.update_layout(
                paper_bgcolor="#15151E",
                plot_bgcolor="#15151E",
                showlegend=False
            )

            st.plotly_chart(
                fig2,
                use_container_width=True
            )

        else:

            st.info(
                "No predictions yet. Go to AI Inquiry Analyzer "
                "to start analyzing customer messages."
            )

    st.divider()

    st.markdown("## 🧠 Model Evaluation")

    st.metric(
        "Holdout Test Accuracy",
        accuracy_text,
        help="Accuracy on the held-out test dataset."
    )

    with st.expander("View Classification Report"):

        report = getattr(model, "test_report_", None)

        if report:
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(
                report_df.round(3),
                use_container_width=True
            )


# ==========================================
# AI INQUIRY ANALYZER
# ==========================================

elif page == "AI Inquiry Analyzer":

    st.markdown("## 🤖 AI Inquiry Analyzer")

    st.write(
        "Enter a customer inquiry and let the ML model "
        "identify the required service."
    )

    sample_messages = [
        "I need an ecommerce website with payment integration",
        "I want an AI chatbot for customer support",
        "Build an Android mobile application",
        "I need AWS cloud hosting"
    ]

    selected = st.selectbox(
        "Choose a sample inquiry",
        ["Write my own inquiry"] + sample_messages
    )

    default_message = (
        "" if selected == "Write my own inquiry"
        else selected
    )

    with st.form("inquiry_form"):

        message = st.text_area(
            "Customer Message",
            value=default_message,
            placeholder="Enter customer requirements here...",
            height=140
        )

        submitted = st.form_submit_button(
            "🔍 Analyze Inquiry",
            use_container_width=True
        )

    if submitted:

        if not message.strip():

            st.warning("Please enter a customer inquiry.")

        elif len(message.strip()) < 10:

            st.warning("Please enter a more detailed inquiry.")

        else:

            category, confidence = predict_category(
                model,
                message
            )

            requirements = extract_requirements(message)

            priority = get_priority(
                message,
                requirements
            )

            response = (
                f"Thank you for contacting Cynbit Technologies. "
                f"We understand that you are looking for "
                f"{category.lower()} services. "
                f"Our team would be happy to understand your "
                f"requirements in more detail. "
                f"Could you please share your expected timeline "
                f"and project goals?"
            )

            result = {
                "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Message": message,
                "Category": category,
                "Confidence": round(confidence * 100, 2),
                "Priority": priority,
                "Requirements": ", ".join(requirements)
            }

            st.session_state.history.append(result)

            st.session_state.last_result = {
                **result,
                "Response": response
            }

    if st.session_state.last_result:

        result = st.session_state.last_result

        st.divider()

        st.markdown("## 🎯 AI Analysis Results")

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Predicted Service",
            result["Category"]
        )

        c2.metric(
            "Model Confidence",
            f'{result["Confidence"]:.2f}%'
        )

        c3.metric(
            "Lead Priority",
            result["Priority"]
        )

        st.progress(
            min(result["Confidence"] / 100, 1.0)
        )

        st.markdown("### 📌 Extracted Requirements")

        if result["Requirements"]:

            for item in result["Requirements"].split(", "):
                st.write("✅", item.title())

        else:

            st.info("No predefined keywords detected.")

        st.markdown("### 💬 Suggested Customer Response")

        st.info(result["Response"])

        st.caption(
            "Confidence is the model's estimated class probability, "
            "not a guarantee of prediction correctness. "
            "Lead priority uses simple keyword rules."
        )


# ==========================================
# PREDICTION HISTORY
# ==========================================

elif page == "Prediction History":

    st.markdown("## 📋 Prediction History")

    if st.session_state.history:

        history_df = pd.DataFrame(
            st.session_state.history
        )

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        st.divider()

        st.markdown("### 📥 Download Reports")

        csv_data = history_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="⬇️ Download CSV Report",
            data=csv_data,
            file_name="cynbit_prediction_history.csv",
            mime="text/csv",
            use_container_width=True
        )

        report_text = (
            "CYNBIT AI - PREDICTION REPORT\n"
            "==============================\n\n"
            + history_df.to_string(index=False)
        )

        st.download_button(
            label="⬇️ Download Text Report",
            data=report_text,
            file_name="cynbit_assessment_report.txt",
            mime="text/plain",
            use_container_width=True
        )

        if st.button("🗑️ Clear Prediction History"):

            st.session_state.history = []
            st.session_state.last_result = None

            st.rerun()

    else:

        st.info(
            "No prediction history available. "
            "Analyze a customer inquiry to create your first record."
        )


# ==========================================
# ABOUT PROJECT
# ==========================================

elif page == "About Project":

    st.markdown("## 💡 About Cynbit AI")

    st.write(
        "Cynbit AI Project Discovery Assistant is an "
        "AI/ML proof of concept that analyzes customer "
        "inquiries and identifies the software service "
        "required by the customer."
    )

    st.markdown("### 🚀 Key Features")

    st.write("""
    - Customer inquiry classification
    - AI-based service category prediction
    - Project requirement extraction
    - Lead priority detection
    - Automated customer response suggestions
    - Prediction analytics and report downloads
    """)

    st.markdown("### 🛠️ Technology Stack")

    st.write("""
    - Python
    - Streamlit
    - Pandas
    - Scikit-learn
    - TF-IDF Vectorization
    - Logistic Regression
    - Plotly
    """)

    st.markdown("### 📂 Supported Service Categories")

    for category in sorted(y.unique()):
        st.write("•", category)

    st.divider()

   #  st.caption("Cynbit Technologies - AI/ML Assessment Project")
