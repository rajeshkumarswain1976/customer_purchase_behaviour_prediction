import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib

matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['axes.unicode_minus'] = False

# Load pre-segmented dataset
st.set_page_config(layout="wide")
df = pd.read_excel("Segmented_Customers.xlsx")

# Load models and scalers
churn_model = joblib.load("churn_model.pkl")
churn_scaler = joblib.load("churn_scaler.pkl")
clv_model = joblib.load("clv_model.pkl")
clv_scaler = joblib.load("clv_scaler.pkl")
kmeans_model = joblib.load("kmeans_model.pkl")
kmeans_scaler = joblib.load("kmeans_scaler.pkl")

# Assign segment at the start if not present
if 'Customer_Segment' not in df.columns:
    segment_features = df[['Age', 'Rating', 'Loyalty_Score']].dropna()
    segment_scaled = kmeans_scaler.transform(segment_features)
    df.loc[segment_features.index, 'Customer_Segment'] = kmeans_model.predict(segment_scaled)

# Create Loyalty Buckets (if not already there)
if 'Loyalty_Bucket' not in df.columns:
    loyalty_bins = [0, 25, 50, 75, 100]
    loyalty_labels = ['0-25', '25-50', '50-75', '75-100']
    df['Loyalty_Bucket'] = pd.cut(df['Loyalty_Score'], bins=loyalty_bins, labels=loyalty_labels, include_lowest=True)

# Sidebar navigation
st.sidebar.title("Navigation")
options = [
    "Overview",
    "CLV Table",
    "Churn Prediction",
    "Customer Segmentation",
    "📉 Average Churn Rate by Loyalty Bucket",
    "Average CLV by Loyalty Group",
    "CLV Distribution by Segment",
    "Visual Analytics"
]
choice = st.sidebar.radio("Go to", options)

if choice == "Overview":
    st.title("Customer Analytics Dashboard")
    st.write("This dashboard uses pre-computed churn, CLV, and segments from the uploaded dataset.")
    st.subheader("Uploaded Data")
    st.dataframe(df.head())

elif choice == "CLV Table":
    st.title("📋 CLV Table")
    st.dataframe(df[['Customer_ID', 'Age', 'Rating', 'Loyalty_Score', 'Customer_Lifetime_Value']].head(50))

elif choice == "Churn Prediction":
    st.title("📊 Churn Prediction")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.selectbox("Select Age", options=list(range(18, 81)), index=12)
    with col2:
        rating = st.radio("Rating", options=[round(x * 0.5, 1) for x in range(2, 11)], index=5, horizontal=True)
    with col3:
        loyalty = st.radio("Loyalty Score", options=list(range(0, 101, 10)), index=5, horizontal=True)

    if st.button("⚡ Predict Now"):
        input_array = np.array([[age, rating, loyalty]])

        churn_scaled = churn_scaler.transform(input_array)
        churn_prob = churn_model.predict_proba(churn_scaled)[0][1]

        segment_input = pd.DataFrame([[age, rating, loyalty]], columns=['Age', 'Rating', 'Loyalty_Score'])
        segment_scaled = kmeans_scaler.transform(segment_input)
        segment = kmeans_model.predict(segment_scaled)[0]

        st.markdown("### 📈 Prediction Results")
        st.metric("Churn Probability", f"{churn_prob * 100:.2f}%")
        st.metric("Customer Segment", f"Segment {segment + 1}")

        fig, ax = plt.subplots()
        labels = ['Churn', 'No Churn']
        values = [churn_prob, 1 - churn_prob]
        colors = ['#4da6ff', '#006bb3']
        ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
        ax.axis('equal')
        st.subheader("Churn Breakdown")
        st.pyplot(fig)

elif choice == "Customer Segmentation":
    st.title("Customer Segmentation")
    st.write(df[['Age', 'Customer_Lifetime_Value', 'Customer_Segment']].head())

elif choice == "📉 Average Churn Rate by Loyalty Bucket":
    st.title("📉 Average Churn Rate by Loyalty Bucket")
    churn_by_loyalty = df.groupby('Loyalty_Bucket')['Churn'].mean().reset_index()
    fig, ax = plt.subplots()
    sns.barplot(x='Loyalty_Bucket', y='Churn', data=churn_by_loyalty, ax=ax, palette='coolwarm')
    for i, row in churn_by_loyalty.iterrows():
        ax.text(i, row['Churn'], f"{row['Churn']:.2f}", ha='center', va='bottom')
    ax.set_title("Average Churn Rate by Loyalty Bucket")
    st.pyplot(fig)

elif choice == "Average CLV by Loyalty Group":
    st.title("Average CLV by Loyalty Group")
    clv_by_loyalty = df.groupby('Loyalty_Bucket')['Customer_Lifetime_Value'].mean().reset_index()
    fig, ax = plt.subplots()
    sns.barplot(x='Loyalty_Bucket', y='Customer_Lifetime_Value', data=clv_by_loyalty, ax=ax, palette='viridis')
    for i, row in clv_by_loyalty.iterrows():
        ax.text(i, row['Customer_Lifetime_Value'], f"{row['Customer_Lifetime_Value']:.0f}", ha='center', va='bottom')
    ax.set_title("Average CLV by Loyalty Group")
    ax.set_ylabel("Avg Customer Lifetime Value")
    st.pyplot(fig)

elif choice == "CLV Distribution by Segment":
    st.title("CLV Distribution by Segment")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=df, x="Customer_Lifetime_Value", hue="Customer_Segment", multiple="stack", kde=True, ax=ax)
    ax.set_title("CLV Distribution by Customer Segment")
    ax.set_xlabel("Customer Lifetime Value")
    st.pyplot(fig)

elif choice == "Visual Analytics":
    st.title("Visual Analytics")

    st.subheader("Elbow Method for Optimal K")
    sse = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        segment_features = df[['Age', 'Rating', 'Loyalty_Score']].dropna()
        scaled = kmeans_scaler.transform(segment_features)
        kmeans.fit(scaled)
        sse.append(kmeans.inertia_)

    fig1, ax1 = plt.subplots()
    ax1.plot(range(1, 11), sse, marker='o')
    ax1.set_xlabel('Number of clusters')
    ax1.set_ylabel('SSE')
    ax1.set_title('Elbow Method for Optimal K')
    st.pyplot(fig1)

    st.subheader("CLV vs Loyalty Score by Cluster")
    segment_colors = pd.Categorical(df['Customer_Segment']).codes

    fig2, ax2 = plt.subplots()
    scatter = ax2.scatter(
        df['Customer_Lifetime_Value'],
        df['Loyalty_Score'],
        c=segment_colors,
        cmap='viridis',
        alpha=0.7
    )
    ax2.set_xlabel("Customer Lifetime Value")
    ax2.set_ylabel("Loyalty Score")
    ax2.set_title("CLV vs Loyalty Score by Cluster")

    categories = pd.Categorical(df['Customer_Segment']).categories
    handles = [
        plt.Line2D([0], [0], marker='o', color='w',
                   label=cat, markersize=10,
                   markerfacecolor=plt.cm.viridis(i / len(categories)))
        for i, cat in enumerate(categories)
    ]
    ax2.legend(handles=handles, title='Customer Segment')

    st.pyplot(fig2)

    st.subheader("CLV Distribution")
    fig3, ax3 = plt.subplots()
    sns.histplot(df['Customer_Lifetime_Value'], kde=True, ax=ax3, color='skyblue')
    ax3.set_title("Overall CLV Distribution")
    ax3.set_ylabel("Count")
    st.pyplot(fig3)
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f8f9fa;
        text-align: center;
        color: #888;
        font-size: 13px;
        padding: 5px 0;
        z-index: 100;
    }
    </style>
    <div class="footer">© 2025 | Developed by Rajesh</div>
    """,
    unsafe_allow_html=True
)


