import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
import os
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Amazon Customer Analytics",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# Load ALL six model files
# Notebook saves:  churn_model.pkl  churn_scaler.pkl
#                  clv_model.pkl    clv_scaler.pkl
#                  kmeans_model.pkl kmeans_scaler.pkl
# ─────────────────────────────────────────────
@st.cache_resource
def load_models():
    base = os.getcwd()
    files = {
        "churn_model":   "churn_model.pkl",
        "churn_scaler":  "churn_scaler.pkl",   # notebook saves as churn_scaler.pkl
        "clv_model":     "clv_model.pkl",
        "clv_scaler":    "clv_scaler.pkl",
        "kmeans_model":  "kmeans_model.pkl",
        "kmeans_scaler": "kmeans_scaler.pkl",
    }
    loaded = {}
    for key, fname in files.items():
        path = os.path.join(base, fname)
        if not os.path.exists(path):
            st.error(f"Missing: `{fname}` — run the notebook first.")
            st.stop()
        loaded[key] = joblib.load(path)
    return loaded

models = load_models()
churn_model   = models["churn_model"]
churn_scaler  = models["churn_scaler"]
clv_model     = models["clv_model"]
clv_scaler    = models["clv_scaler"]
kmeans_model  = models["kmeans_model"]
kmeans_scaler = models["kmeans_scaler"]

# ─────────────────────────────────────────────
# Load dataset
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    path = os.path.join(os.getcwd(), "amazon.xlsx")
    try:
        df = pd.read_excel(path)
        df.columns = df.columns.str.strip()
        return df
    except FileNotFoundError:
        st.warning("amazon.xlsx not found — data visualisations will be skipped.")
        return pd.DataFrame()

df_original = load_data()

# ─────────────────────────────────────────────
# Feature lists — must match notebook EXACTLY
#
# CHURN model (Cell #18) — LogisticRegression
# actual_churn_features (after guard filter):
#   Age, Purchase_Amount, Payment_Method, Loyalty_Score,
#   Avg_Purchase_Value, Days_Since_Purchase, Rating,
#   Location, Product_Category, Return_Status, Customer_Segment
#   (CLV_Engineered excluded because it wasn't in df at save time)
#
# CLV model (Cell #17) — LinearRegression
#   Age, Purchase_Amount, Payment_Method, Loyalty_Score,
#   Rating, Gender, Return_Status, Customer_Segment,
#   Days_Since_Purchase
#
# KMeans (Cell #14/15):
#   Customer_Lifetime_Value, Loyalty_Score, Purchase_Amount
# ─────────────────────────────────────────────
CHURN_FEATURES = [
    "Age", "Purchase_Amount", "Payment_Method", "Loyalty_Score",
    "Avg_Purchase_Value", "Days_Since_Purchase", "Rating",
    "Location", "Product_Category", "Return_Status", "Customer_Segment"
]

CLV_FEATURES = [
    "Age", "Purchase_Amount", "Payment_Method", "Loyalty_Score",
    "Rating", "Gender", "Return_Status", "Customer_Segment",
    "Days_Since_Purchase"
]

KMEANS_FEATURES = ["Customer_Lifetime_Value", "Loyalty_Score", "Purchase_Amount"]

# Encoded mappings (match notebook LabelEncoder order — alphabetical fit)
PAYMENT_MAP     = {"Bank Transfer": 0, "Cash": 1, "Credit Card": 2, "PayPal": 3}
GENDER_MAP      = {"Female": 0, "Male": 1, "Other": 2}
LOCATION_MAP    = {"Chicago": 0, "Houston": 1, "Los Angeles": 2, "New York": 3, "San Francisco": 4}
PRODUCT_MAP     = {"Books": 0, "Clothing": 1, "Electronics": 2, "Home Appliances": 3, "Toys": 4}
RETURN_MAP      = {"No": 0, "Yes": 1}
SEGMENT_MAP     = {"New": 0, "Regular": 1, "VIP": 2}
CHANNEL_MAP     = {"Both": 0, "In-store": 1, "Online": 2}

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.title("🛒 Amazon Customer Analytics Dashboard")
st.markdown(
    "Predict **churn**, estimate **Customer Lifetime Value**, and identify "
    "**customer segment** — all from a single customer profile."
)
st.markdown("---")

# ─────────────────────────────────────────────
# Sidebar — customer profile input
# ─────────────────────────────────────────────
st.sidebar.header("Customer Profile")
st.sidebar.markdown("Fill in the customer details below.")

age              = st.sidebar.slider("Age", 18, 70, 35)
purchase_amount  = st.sidebar.number_input("Purchase Amount ($)", 10.0, 1000.0, 250.0, step=10.0)
loyalty_score    = st.sidebar.slider("Loyalty Score (1–100)", 1, 100, 50)
rating           = st.sidebar.slider("Rating (1–5)", 1, 5, 3)
avg_purchase_val = st.sidebar.number_input("Avg Purchase Value ($)", 10.0, 1000.0, 200.0, step=10.0)
days_since       = st.sidebar.number_input("Days Since Last Purchase", 0, 2000, 180, step=10)
clv_real         = st.sidebar.number_input("Customer Lifetime Value ($)", 50.0, 8000.0, 1500.0, step=50.0)

st.sidebar.markdown("---")
payment_method   = st.sidebar.selectbox("Payment Method", list(PAYMENT_MAP.keys()))
gender           = st.sidebar.selectbox("Gender", list(GENDER_MAP.keys()))
location         = st.sidebar.selectbox("Location", list(LOCATION_MAP.keys()))
product_cat      = st.sidebar.selectbox("Product Category", list(PRODUCT_MAP.keys()))
return_status    = st.sidebar.selectbox("Return Status", list(RETURN_MAP.keys()))
customer_segment = st.sidebar.selectbox("Customer Segment", list(SEGMENT_MAP.keys()))

predict_btn = st.sidebar.button("Run Predictions", type="primary", use_container_width=True)

# ─────────────────────────────────────────────
# Prediction logic
# ─────────────────────────────────────────────
def build_churn_input():
    """Build the 11-feature vector for churn model — exact order as CHURN_FEATURES."""
    return pd.DataFrame([{
        "Age":               age,
        "Purchase_Amount":   purchase_amount,
        "Payment_Method":    PAYMENT_MAP[payment_method],
        "Loyalty_Score":     loyalty_score,
        "Avg_Purchase_Value": avg_purchase_val,
        "Days_Since_Purchase": days_since,
        "Rating":            rating,
        "Location":          LOCATION_MAP[location],
        "Product_Category":  PRODUCT_MAP[product_cat],
        "Return_Status":     RETURN_MAP[return_status],
        "Customer_Segment":  SEGMENT_MAP[customer_segment],
    }], columns=CHURN_FEATURES)

def build_clv_input():
    """Build the 9-feature vector for CLV model — exact order as CLV_FEATURES."""
    return pd.DataFrame([{
        "Age":               age,
        "Purchase_Amount":   purchase_amount,
        "Payment_Method":    PAYMENT_MAP[payment_method],
        "Loyalty_Score":     loyalty_score,
        "Rating":            rating,
        "Gender":            GENDER_MAP[gender],
        "Return_Status":     RETURN_MAP[return_status],
        "Customer_Segment":  SEGMENT_MAP[customer_segment],
        "Days_Since_Purchase": days_since,
    }], columns=CLV_FEATURES)

def build_kmeans_input():
    """Build the 3-feature vector for KMeans — exact order as KMEANS_FEATURES."""
    return pd.DataFrame([{
        "Customer_Lifetime_Value": clv_real,
        "Loyalty_Score":           loyalty_score,
        "Purchase_Amount":         purchase_amount,
    }], columns=KMEANS_FEATURES)

# ─────────────────────────────────────────────
# Main content — tabs
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["Churn Prediction", "CLV Prediction", "Customer Segment", "Data Insights"]
)

# ── Tab 1: Churn ──────────────────────────────
with tab1:
    st.subheader("Churn Prediction")
    st.markdown(
        "Model: **Logistic Regression** with `class_weight='balanced'`  \n"
        f"Features used: `{', '.join(CHURN_FEATURES)}`  \n"
        "Trained accuracy: **85.58%** · F1 (churn class): **0.86**"
    )

    if predict_btn:
        churn_input  = build_churn_input()
        churn_scaled = churn_scaler.transform(churn_input)
        prediction   = churn_model.predict(churn_scaled)[0]
        proba        = churn_model.predict_proba(churn_scaled)[0]

        col_a, col_b = st.columns(2)

        with col_a:
            if prediction == 1:
                st.error("### Likely to Churn")
                st.metric("Churn probability", f"{proba[1]*100:.1f}%")
            else:
                st.success("### Not Likely to Churn")
                st.metric("Retention probability", f"{proba[0]*100:.1f}%")

        with col_b:
            fig_pie = go.Figure(go.Pie(
                labels=["No Churn", "Churn"],
                values=[proba[0], proba[1]],
                hole=0.45,
                marker_colors=["#2ecc71", "#e74c3c"]
            ))
            fig_pie.update_layout(
                title="Churn probability split",
                margin=dict(t=40, b=0, l=0, r=0),
                showlegend=True
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        # Bar chart
        fig_bar = go.Figure(go.Bar(
            x=["No Churn", "Churn"],
            y=[proba[0], proba[1]],
            marker_color=["#2ecc71", "#e74c3c"],
            text=[f"{p*100:.1f}%" for p in proba],
            textposition="auto",
            width=[0.35, 0.35]
        ))
        fig_bar.update_layout(
            title="Predicted probability breakdown",
            yaxis=dict(title="Probability", range=[0, 1]),
            xaxis_title="Outcome",
            bargap=0.5
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("**Input sent to model:**")
        st.dataframe(build_churn_input(), use_container_width=True)
    else:
        st.info("Fill in the customer profile on the left and click **Run Predictions**.")

# ── Tab 2: CLV ───────────────────────────────
with tab2:
    st.subheader("Customer Lifetime Value Prediction")
    st.markdown(
        "Model: **Linear Regression**  \n"
        f"Features used: `{', '.join(CLV_FEATURES)}`  \n"
        "Trained R²: **0.7648** · MAPE: **27.8%** · MAE: **594.60**"
    )

    if predict_btn:
        clv_input  = build_clv_input()
        clv_scaled = clv_scaler.transform(clv_input)
        clv_pred   = clv_model.predict(clv_scaled)[0]

        st.metric("Predicted Customer Lifetime Value", f"${clv_pred:,.2f}")

        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=clv_pred,
            delta={"reference": 2000, "label": "vs avg $2,000"},
            gauge={
                "axis": {"range": [0, 8000]},
                "bar":  {"color": "#3498db"},
                "steps": [
                    {"range": [0,    1500], "color": "#fadbd8"},
                    {"range": [1500, 4000], "color": "#fdebd0"},
                    {"range": [4000, 8000], "color": "#d5f5e3"},
                ],
                "threshold": {
                    "line":  {"color": "black", "width": 3},
                    "thickness": 0.75,
                    "value": 2000
                }
            },
            title={"text": "Predicted CLV ($)"}
        ))
        fig_gauge.update_layout(height=350)
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.caption(
            "CLV ranges: Low < $1,500 · Mid $1,500–$4,000 · High > $4,000  \n"
            "Black line = dataset average ($2,000)"
        )

        st.markdown("**Input sent to model:**")
        st.dataframe(build_clv_input(), use_container_width=True)
    else:
        st.info("Fill in the customer profile on the left and click **Run Predictions**.")

# ── Tab 3: Segment ────────────────────────────
with tab3:
    st.subheader("Customer Segment")
    st.markdown(
        "Model: **KMeans (k=3)** — Silhouette score: **0.348**  \n"
        f"Features used: `{', '.join(KMEANS_FEATURES)}`"
    )

    SEGMENT_LABELS = {
        0: ("Low-Value Customers",    "Low CLV, low loyalty, low spend. At-risk group.", "#e74c3c"),
        1: ("Mid-Value Customers",    "Moderate CLV and spend. Growth opportunity.",      "#f39c12"),
        2: ("High-Value Customers",   "High CLV, high loyalty, high spend. VIP group.",   "#2ecc71"),
    }

    if predict_btn:
        km_input  = build_kmeans_input()
        km_scaled = kmeans_scaler.transform(km_input)
        segment   = int(kmeans_model.predict(km_scaled)[0])

        label, desc, colour = SEGMENT_LABELS[segment]

        st.markdown(
            f"<div style='background:{colour}22;border-left:5px solid {colour};"
            f"padding:16px;border-radius:8px'>"
            f"<h3 style='color:{colour};margin:0'>Segment {segment} — {label}</h3>"
            f"<p style='margin:6px 0 0;color:var(--text-color)'>{desc}</p></div>",
            unsafe_allow_html=True
        )

        st.markdown("**Input sent to model:**")
        st.dataframe(build_kmeans_input(), use_container_width=True)

        st.markdown("---")
        st.markdown("#### All segment profiles")
        for seg_id, (lbl, dsc, col) in SEGMENT_LABELS.items():
            marker = "◀ **this customer**" if seg_id == segment else ""
            st.markdown(f"- **Segment {seg_id} — {lbl}**: {dsc} {marker}")
    else:
        st.info("Fill in the customer profile on the left and click **Run Predictions**.")

# ── Tab 4: Data Insights ─────────────────────
with tab4:
    st.subheader("Dataset Insights")

    if df_original.empty:
        st.warning("amazon.xlsx not found — place it in the same folder as app.py.")
    else:
        st.markdown(f"Dataset: **{df_original.shape[0]:,} rows × {df_original.shape[1]} columns**")
        st.dataframe(df_original.head(10), use_container_width=True)
        st.markdown("---")

        col1, col2 = st.columns(2)

        # Age distribution
        with col1:
            if "Age" in df_original.columns:
                fig_age = px.histogram(
                    df_original, x="Age", nbins=20,
                    title="Customer Age Distribution",
                    color_discrete_sequence=["#3498db"]
                )
                fig_age.update_layout(bargap=0.1, xaxis_title="Age", yaxis_title="Count")
                st.plotly_chart(fig_age, use_container_width=True)

        # Product category
        with col2:
            if "Product_Category" in df_original.columns:
                counts = df_original["Product_Category"].value_counts().reset_index()
                counts.columns = ["Product_Category", "Count"]
                fig_cat = px.pie(
                    counts, values="Count", names="Product_Category",
                    title="Product Category Distribution", hole=0.35,
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                fig_cat.update_traces(textposition="inside", textinfo="percent+label")
                st.plotly_chart(fig_cat, use_container_width=True)

        col3, col4 = st.columns(2)

        # Churn distribution
        with col3:
            if "Churn" in df_original.columns:
                churn_counts = df_original["Churn"].map({1: "Churn", 0: "No Churn"}).value_counts()
                fig_churn = px.bar(
                    x=churn_counts.index, y=churn_counts.values,
                    title="Churn Distribution (Real Labels)",
                    color=churn_counts.index,
                    color_discrete_map={"Churn": "#e74c3c", "No Churn": "#2ecc71"},
                    labels={"x": "Status", "y": "Count"}
                )
                st.plotly_chart(fig_churn, use_container_width=True)

        # CLV distribution
        with col4:
            if "Customer_Lifetime_Value" in df_original.columns:
                fig_clv = px.histogram(
                    df_original, x="Customer_Lifetime_Value", nbins=30,
                    title="Customer Lifetime Value Distribution",
                    color_discrete_sequence=["#9b59b6"]
                )
                fig_clv.update_layout(xaxis_title="CLV ($)", yaxis_title="Count")
                st.plotly_chart(fig_clv, use_container_width=True)

        # Payment method
        if "Payment_Method" in df_original.columns:
            pm = df_original["Payment_Method"].value_counts().reset_index()
            pm.columns = ["Payment_Method", "Count"]
            fig_pm = px.bar(
                pm, x="Payment_Method", y="Count",
                title="Payment Method Usage",
                color="Payment_Method",
                color_discrete_sequence=px.colors.qualitative.Plotly
            )
            st.plotly_chart(fig_pm, use_container_width=True)

        # Location
        if "Location" in df_original.columns:
            loc = df_original["Location"].value_counts().reset_index()
            loc.columns = ["Location", "Count"]
            fig_loc = px.bar(
                loc, x="Location", y="Count",
                title="Customers by Location",
                color="Location",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig_loc, use_container_width=True)

st.markdown("---")
st.caption("Amazon Customer Analytics · Built with amazonnew.ipynb · Model accuracy: Churn 85.58% · CLV R² 0.7648")
