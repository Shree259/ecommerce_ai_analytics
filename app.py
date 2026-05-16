import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="AI E-Commerce Analytics", layout="wide")

st.title("🛒 AI-Powered E-Commerce Analytics Dashboard")

# Upload CSV
uploaded_file = st.file_uploader("Upload Customer CSV File", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    required_columns = [
        "CustomerID",
        "Age",
        "PurchaseAmount",
        "Orders",
        "Country"
    ]

    missing_cols = [col for col in required_columns if col not in df.columns]

    if missing_cols:
        st.error(f"Missing columns: {missing_cols}")
        st.stop()

    st.subheader("Dataset Preview")
    st.dataframe(df)

    
    # BASIC ANALYTICS
    

    total_revenue = df["PurchaseAmount"].sum()
    total_orders = df["Orders"].sum()
    avg_purchase = df["PurchaseAmount"].mean()

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Revenue", f"${total_revenue:,.2f}")
    col2.metric("Total Orders", total_orders)
    col3.metric("Average Purchase", f"${avg_purchase:,.2f}")

    
    # COUNTRY ANALYSIS
    

    st.subheader("Country-wise Revenue")

    country_sales = df.groupby("Country")["PurchaseAmount"].sum().reset_index()

    fig = px.bar(
        country_sales,
        x="Country",
        y="PurchaseAmount",
        title="Revenue by Country"
    )

    st.plotly_chart(fig, use_container_width=True)


    # CUSTOMER SEGMENTATION

    st.subheader("AI Customer Segmentation")

    X = df[["Age", "PurchaseAmount", "Orders"]]

    kmeans = KMeans(n_clusters=3, random_state=42)
    df["Cluster"] = kmeans.fit_predict(X)

    fig2 = px.scatter(
        df,
        x="PurchaseAmount",
        y="Orders",
        color=df["Cluster"].astype(str),
        hover_data=["CustomerID"],
        title="Customer Segmentation"
    )

    st.plotly_chart(fig2, use_container_width=True)

    
    # SALES PREDICTION
    

    st.subheader("AI Sales Prediction")

    X_pred = df[["Orders"]]
    y_pred = df["PurchaseAmount"]

    model = LinearRegression()
    model.fit(X_pred, y_pred)

    future_orders = pd.DataFrame([[25]], columns=["Orders"])

    predicted_sales = model.predict(future_orders)

    st.success(f"Predicted Sales for 25 Orders: ${predicted_sales[0]:.2f}")


    # DOWNLOAD CSV

    csv = df.to_csv(index=False)

    st.download_button(
        label="Download Processed Data",
        data=csv,
        file_name="processed_data.csv",
        mime="text/csv"
    )

else:
    st.info("Please upload a CSV file.")