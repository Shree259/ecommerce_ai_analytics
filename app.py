import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Universal AI Analytics Dashboard", layout="wide")

st.title("📊 Universal AI-Powered Analytics Dashboard")

# Upload CSV
uploaded_file = st.file_uploader("Upload Any CSV File", type=["csv"])

if uploaded_file:

    try:
        df = pd.read_csv(uploaded_file, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(uploaded_file, encoding='latin1')
        except:
            df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')

        # =========================
        # DATA PREVIEW
        # =========================

        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        st.subheader("Dataset Information")

        col1, col2 = st.columns(2)

        col1.metric("Rows", df.shape[0])
        col2.metric("Columns", df.shape[1])

        # =========================
        # COLUMN DETECTION
        # =========================

        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

        st.sidebar.title("Dashboard Controls")

        # =========================
        # KPI SECTION
        # =========================

        st.subheader("📈 KPI Analytics")

        if numeric_cols:

            selected_kpi = st.selectbox(
                "Select Numeric Column for KPI",
                numeric_cols
            )

            total_value = df[selected_kpi].sum()
            avg_value = df[selected_kpi].mean()
            max_value = df[selected_kpi].max()

            c1, c2, c3 = st.columns(3)

            c1.metric("Total", f"{total_value:,.2f}")
            c2.metric("Average", f"{avg_value:,.2f}")
            c3.metric("Maximum", f"{max_value:,.2f}")

        # =========================
        # BAR CHART
        # =========================

        st.subheader("📊 Interactive Bar Chart")

        if categorical_cols and numeric_cols:

            x_axis = st.selectbox("Select Category Column", categorical_cols)
            y_axis = st.selectbox("Select Numeric Column", numeric_cols)

            grouped_data = df.groupby(x_axis)[y_axis].sum().reset_index()

            fig = px.bar(
                grouped_data,
                x=x_axis,
                y=y_axis,
                color=x_axis,
                title=f"{y_axis} by {x_axis}"
            )

            st.plotly_chart(fig, use_container_width=True)

        # =========================
        # PIE CHART
        # =========================

        st.subheader("🥧 Pie Chart")

        if categorical_cols:

            pie_col = st.selectbox(
                "Select Column for Pie Chart",
                categorical_cols
            )

            pie_data = df[pie_col].value_counts().reset_index()
            pie_data.columns = [pie_col, "Count"]

            fig2 = px.pie(
                pie_data,
                names=pie_col,
                values="Count",
                title=f"Distribution of {pie_col}"
            )

            st.plotly_chart(fig2, use_container_width=True)

        # =========================
        # SCATTER PLOT
        # =========================

        st.subheader("🔍 Scatter Plot")

        if len(numeric_cols) >= 2:

            x_scatter = st.selectbox(
                "Select X-axis",
                numeric_cols,
                key="scatter_x"
            )

            y_scatter = st.selectbox(
                "Select Y-axis",
                numeric_cols,
                key="scatter_y"
            )

            fig3 = px.scatter(
                df,
                x=x_scatter,
                y=y_scatter,
                color=y_scatter,
                title=f"{y_scatter} vs {x_scatter}"
            )

            st.plotly_chart(fig3, use_container_width=True)

        # =========================
        # AI CUSTOMER SEGMENTATION
        # =========================

        st.subheader("🤖 AI Clustering")

        if len(numeric_cols) >= 2:

            cluster_features = st.multiselect(
                "Select Features for Clustering",
                numeric_cols,
                default=numeric_cols[:2]
            )

            if len(cluster_features) >= 2:

                X = df[cluster_features].dropna()

                kmeans = KMeans(n_clusters=3, random_state=42)

                clusters = kmeans.fit_predict(X)

                cluster_df = X.copy()
                cluster_df["Cluster"] = clusters

                fig4 = px.scatter(
                    cluster_df,
                    x=cluster_features[0],
                    y=cluster_features[1],
                    color=cluster_df["Cluster"].astype(str),
                    title="AI-Based Clustering"
                )

                st.plotly_chart(fig4, use_container_width=True)

        # =========================
        # AI PREDICTION
        # =========================

        st.subheader("📉 AI Prediction")

        if len(numeric_cols) >= 2:

            x_pred = st.selectbox(
                "Select Feature Column",
                numeric_cols,
                key="pred_x"
            )

            y_pred = st.selectbox(
                "Select Target Column",
                numeric_cols,
                key="pred_y"
            )

            X = df[[x_pred]].dropna()
            y = df[y_pred].dropna()

            if len(X) == len(y):

                model = LinearRegression()
                model.fit(X, y)

                future_value = st.number_input(
                    f"Enter Future {x_pred}",
                    value=10.0
                )

                future_df = pd.DataFrame(
                    [[future_value]],
                    columns=[x_pred]
                )

                prediction = model.predict(future_df)

                st.success(
                    f"Predicted {y_pred}: {prediction[0]:,.2f}"
                )

        # =========================
        # DOWNLOAD CLEANED DATA
        # =========================

        st.subheader("⬇ Download Processed Dataset")

        csv = df.to_csv(index=False)

        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="processed_data.csv",
            mime="text/csv"
        )

    except Exception as e:

        st.error(f"Error Processing File: {e}")

else:

    st.info("Please upload a CSV file to begin.")