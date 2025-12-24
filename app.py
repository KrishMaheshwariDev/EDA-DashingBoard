import streamlit as st
import pandas as pd
import plotly.express as px

# SideBar Panel
st.sidebar.header("Dataset")
uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded is not None:
    df = pd.read_csv(uploaded)
else:
    st.warning("Upload a CSV file")
    st.stop()  

st.sidebar.header("EDA controls")
target = st.sidebar.selectbox("Select the Target Column", options=df.columns, index=None)

if target is None:
     st.warning("Select the Target Feature")
     st.stop()

is_target_numeric = pd.api.types.is_numeric_dtype(df[target])
is_target_categorical = not is_target_numeric

# Variables

all_num = df.select_dtypes(include=["number"]).columns
all_cat = df.select_dtypes(include=["object", "category"]).columns

num_features = all_num.drop(target) if target in all_num else all_num
cat_features = all_cat.drop(target) if target in all_cat else all_cat

# Memory optimization (little bit)

@st.cache_data
def get_correlation(df):
    return df.corr()


# Tabs Section

if is_target_numeric:

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Overview",
        "Distributions",
        "Relationships",
        "Correlation & Risk",
        "Categorical"
    ])

    # Tab1 : Overview
    with tab1:
        st.header("Dataset Overview")

        st.subheader("Quick Dataset Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", df.shape[0])
        col2.metric("Columns", df.shape[1])
        col3.metric("Missing Values", df.isna().sum().sum())

        st.dataframe(df.head(10))
        

        st.header(f"{target} Distribuition")
        col1, col2 = st.columns([3,1])

        with col1:
            fig = px.histogram(
                df,
                x=target,
                nbins=50,
                title=f"Distribution of {target}"
            )

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.box(
                df,
                y=target,
                title=f"Outliers of {target}"
            )

        st.plotly_chart(fig, use_container_width=True)

    # Tab2 : Distributions

    with tab2:
        feature = st.selectbox("Select the Numerical Feature", num_features, key=0)

        col1, col2 = st.columns([3,1])

        with col1:
                fig = px.histogram(
                    df,
                    x=feature,
                    nbins=50,
                    title=f"Distribution of {feature}"
                )

                st.plotly_chart(fig, use_container_width=True)

        with col2:
                fig = px.box(
                    df,
                    y=feature,
                    title=f"Outliers of {feature}"
                )

                st.plotly_chart(fig, use_container_width=True)

    # Tab 3: Relationships

    with tab3:
        feature = st.selectbox("Select the Numerical Feature", num_features, key=1)

        
        fig = px.scatter(
                df,
                x=feature,
                y=target,
                trendline="ols",
                opacity=0.5,
                title=f"{feature} vs {target}"
            )

        st.plotly_chart(fig, use_container_width=True)

    # Tab 4: correlations

    with tab4:
        st.header("Top Correlated Features")

        k = st.slider("Top K features", min_value=5, max_value=20, value=10)

        corr = (
                df.select_dtypes(include=["int64", "float64"])
                .corr()[target]
                .drop(target)
                .sort_values(ascending=False)
                .head(k)
            )

        fig = px.bar(
                corr,
                x=corr.values,
                y=corr.index,
                orientation="h",
                title=f"Top {k} Correlated features with {target}"
            )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Feature Redundancy Check")

        f1, f2 = st.selectbox(
                "Feature 1", num_features, key="f1"
            ), st.selectbox(
                "Feature 2", num_features, key="f2"
            )

        if f1 != f2:
                corr_val = df[[f1, f2]].corr().iloc[0, 1]

                fig = px.scatter(
                    df,
                    x=f1,
                    y=f2,
                    title=f"{f1} vs {f2} (corr = {corr_val:.2f})",
                    opacity=0.5
                )
                st.plotly_chart(fig, use_container_width=True)

        st.subheader("Potential Multicollinearity Risks")

        if len(num_features) < 2:
                st.info("Not enough numerical features for correlation analysis.")
        else:
                corr_matrix = get_correlation(df[num_features]).abs()

                high_corr_pairs = (
                    corr_matrix.where(
                        (corr_matrix > 0.8) & (corr_matrix < 1.0)
                    )
                    .stack()
                    .sort_values(ascending=False)
                )

                st.dataframe(high_corr_pairs.rename("Correlation"))

    # Tab 5: Categorical analysis

    with tab5:
        st.subheader(f"Categorical Feature Impact on {target}")

        cat_feature = st.selectbox("Select categorical feature", cat_features)


        # Compute median impact
        impact = df.groupby(cat_feature)[target].median().sort_values()

        fig = px.bar(
                impact,
                orientation="h",
                title=f"Median {target} by {cat_feature}",
            )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Category Variance (Stability Check)")

        stats = df.groupby(cat_feature)[target].agg(["median", "std", "count"])
        stats = stats.sort_values("median")

        fig = px.scatter(
                stats,
                x="median",
                y=stats.index,
                size="count",
                color="std",
                title=f"Stability of {cat_feature} categories"
            )

        st.plotly_chart(fig, use_container_width=True)

else:
     
    tab1, tab2 = st.tabs([
          "Overview",
          "Class Behaviour"
    ])

    # Tab 1: Overview

    with tab1:
        st.header("Dataset Overview")

        st.subheader("Quick Dataset Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", df.shape[0])
        col2.metric("Columns", df.shape[1])
        col3.metric("Missing Values", df.isna().sum().sum())

        st.dataframe(df.head(10))

        st.header(f"{target} Distribuition")

        fig = px.histogram(
                df,
                x=target,
                nbins=50,
                title=f"Distribution of {target}"
            )

        st.plotly_chart(fig, use_container_width=True)
    
    # Tab 2: Categorical

    with tab2:
        st.subheader("Class Distribution")

        class_counts = df[target].value_counts()

        fig = px.bar(
                class_counts,
                x=class_counts.index,
                y=class_counts.values,
                title=f"Distribution of target class: {target}"
            )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Numerical Feature Separation Across Classes")

        num_for_class = df.select_dtypes(include="number").columns

        if target in num_for_class:
             num_for_class = num_for_class.drop(target)

        num_feature = st.selectbox("Select numerical feature", num_for_class)

        fig = px.box(
                df,
                x=target,
                y=num_feature,
                title=f"{num_feature} distribution across {target} classes"
            )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Categorical Feature vs Target Relationship")

        cat_for_class = df.select_dtypes(include=["object", "category"]).columns.drop(target)

        cat_feature = st.selectbox("Select categorical feature", cat_for_class, key="cat_vs_target")

        crosstab = pd.crosstab(df[cat_feature], df[target], normalize="index").sort_index()

        fig = px.imshow(
                crosstab,
                aspect="auto",
                title=f"Distribution of {target} within each {cat_feature} category",
                color_continuous_scale="Blues"
            )
        st.plotly_chart(fig, use_container_width=True)
