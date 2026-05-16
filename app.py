import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Universal AI Analytics Dashboard",
    layout="wide"
)

st.title("📊 Universal AI-Powered Analytics Dashboard")

# =========================
# FILE UPLOAD
# =========================

uploaded_file = st.file_uploader(
    "Upload Any CSV File",
    type=["csv"]
)

# =========================
# MAIN APP
# =========================

if uploaded_file is not None:

    try:

        # =========================
        # SAFE CSV READING
        # =========================

        try:
            df = pd.read_csv(uploaded_file, encoding='utf-8')

        except UnicodeDecodeError:

            try:
                df = pd.read_csv(uploaded_file, encoding='latin1')

            except:
                df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')

        # =========================
        # SUCCESS MESSAGE
        # =========================

        st.success("CSV Loaded Successfully ✅")

        # =========================
        # DATA PREVIEW
        # =========================

        st.subheader("📄 Dataset Preview")
        st.dataframe(df.head())

        # =========================
        # DATASET INFO
        # =========================

        st.subheader("📌 Dataset Information")

        col1, col2 = st.columns(2)

        col1.metric("Rows", df.shape[0])
        col2.metric("Columns", df.shape[1])

        # =========================
        # COLUMN DETECTION
        # =========================

        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        categorical_cols = df.select_dtypes(
            exclude=np.number
        ).columns.tolist()

        # =========================
        # KPI SECTION
        # =========================

        st.subheader("📈 KPI Analytics")

        if len(numeric_cols) > 0:

            selected_kpi = st.selectbox(
                "Select Numeric Column",
                numeric_cols
            )

            total_value = df[selected_kpi].sum()
            avg_value = df[selected_kpi].mean()
            max_value = df[selected_kpi].max()

            k1, k2, k3 = st.columns(3)

            k1.metric("Total", f"{total_value:,.2f}")
            k2.metric("Average", f"{avg_value:,.2f}")
            k3.metric("Maximum", f"{max_value:,.2f}")

        # =========================
        # BAR CHART
        # =========================

        st.subheader("📊 Interactive Bar Chart")

        if len(categorical_cols) > 0 and len(numeric_cols) > 0:

            x_axis = st.selectbox(
                "Select Category Column",
                categorical_cols
            )

            y_axis = st.selectbox(
                "Select Numeric Column",
                numeric_cols,
                key="bar_y"
            )

            grouped_data = (
                df.groupby(x_axis)[y_axis]
                .sum()
                .reset_index()
            )

            fig_bar = px.bar(
                grouped_data,
                x=x_axis,
                y=y_axis,
                color=x_axis,
                title=f"{y_axis} by {x_axis}"
            )

            st.plotly_chart(
                fig_bar,
                use_container_width=True
            )

        # =========================
        # PIE CHART
        # =========================

        st.subheader("🥧 Pie Chart")

        if len(categorical_cols) > 0:

            pie_col = st.selectbox(
                "Select Column for Pie Chart",
                categorical_cols
            )

            pie_data = (
                df[pie_col]
                .value_counts()
                .reset_index()
            )

            pie_data.columns = [pie_col, "Count"]

            fig_pie = px.pie(
                pie_data,
                names=pie_col,
                values="Count",
                title=f"Distribution of {pie_col}"
            )

            st.plotly_chart(
                fig_pie,
                use_container_width=True
            )

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

            fig_scatter = px.scatter(
                df,
                x=x_scatter,
                y=y_scatter,
                color=y_scatter,
                title=f"{y_scatter} vs {x_scatter}"
            )

            st.plotly_chart(
                fig_scatter,
                use_container_width=True
            )

        # =========================
        # AI CLUSTERING
        # =========================

        st.subheader("🤖 AI Clustering")

        try:

            if len(numeric_cols) >= 2:

                cluster_features = st.multiselect(
                    "Select Features for Clustering",
                    numeric_cols,
                    default=numeric_cols[:2]
                )

                if len(cluster_features) >= 2:

                    X = df[cluster_features].dropna()

                    if len(X) > 2:

                        kmeans = KMeans(
                            n_clusters=3,
                            random_state=42
                        )

                        clusters = kmeans.fit_predict(X)

                        cluster_df = X.copy()
                        cluster_df["Cluster"] = clusters

                        fig_cluster = px.scatter(
                            cluster_df,
                            x=cluster_features[0],
                            y=cluster_features[1],
                            color=cluster_df["Cluster"].astype(str),
                            title="AI-Based Clustering"
                        )

                        st.plotly_chart(
                            fig_cluster,
                            use_container_width=True
                        )

        except Exception as e:

            st.error(f"Clustering Error: {e}")

        # =========================
        # AI PREDICTION
        # =========================

        st.subheader("📉 AI Prediction")

        try:

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

                prediction_df = df[
                    [x_pred, y_pred]
                ].dropna()

                X = prediction_df[[x_pred]]
                y = prediction_df[y_pred]

                if len(X) > 1:

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

                    prediction = model.predict(
                        future_df
                    )

                    st.success(
                        f"Predicted {y_pred}: "
                        f"{prediction[0]:,.2f}"
                    )

        except Exception as e:

            st.error(f"Prediction Error: {e}")

        # =========================
        # DOWNLOAD DATA
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

# =========================
# NO FILE UPLOADED
# =========================

else:

    st.info("Please upload a CSV file to begin.")