import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

st.set_page_config(layout="wide", page_title="Dutch Real Estate")

def load_data():
    df = pd.read_csv("../data/raw_dutch_real_estate.csv")
    df = df.dropna(
        subset=[
            "Price",
            "Living space size (m2)",
            "Rooms",
            "Build year",
            "Lot size (m2)",
            "Estimated neighbourhood price per m2"
        ]
    )
    df["Living space size (m2)"] = (
        df["Living space size (m2)"]
        .str.replace(" m²", "", regex=False)
        .astype(float)
    )

    df["Lot size (m2)"] = (
        df["Lot size (m2)"]
        .str.replace(" m²", "", regex=False)
        .astype(float)
    )

    df["Price"] = (
        df["Price"]
        .str.replace("€ ", "", regex=False)
        .str.replace(".", "", regex=False)
    )

    df["Price"] = pd.to_numeric(
        df["Price"],
        errors="coerce"
    )

    df = df.dropna(subset=["Price"])

    df["Rooms"] = (
        df["Rooms"]
        .str.extract(r"(\d+)")
        .astype(int)
    )

    df["Build year"] = (
        df["Build year"]
        .replace("Voor 1906", "1905")
    )

    df["Build year"] = pd.to_numeric(
        df["Build year"],
        errors="coerce"
    )

    df = df.dropna(subset=["Build year"])

    df["Estimated neighbourhood price per m2"] = (
        df["Estimated neighbourhood price per m2"]
        .str.replace("€ ", "", regex=False)
        .str.replace(".", "", regex=False)
    )

    df["Estimated neighbourhood price per m2"] = pd.to_numeric(
        df["Estimated neighbourhood price per m2"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["Estimated neighbourhood price per m2"]
    )

    return df

df = load_data()

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Dashboard Section:",
        [
            "Market Overview",
            "Property Exploration",
            "Price Prediction"
        ]
)

if page == "Market Overview":
    st.title("Dutch Housing Market Analytics Dashboard")
    st.info(
        "Dataset Source: Dutch House Prices Dataset (Kaggle)"
    )
    st.markdown(
        """
        ## Market Overview

        The Dutch housing market has experienced significant changes in property values, housing demand, and regional price disparities.

        This dashboard was designed to support exploratory analysis of residential real estate listings across the Netherlands.

        Key questions addressed in this report:

        - Which cities have the highest property prices?
        - How does living space influence property value?
        - What is the relationship between energy labels and housing prices?
        - How do property characteristics vary across regions?
        - What patterns can be identified within the Dutch housing market?

        Additionally, the dashboard integrates a regression-based machine learning model for estimating residential property values using structural property characteristics and neighborhood market indicators.

        Navigate through the sections using the sidebar to explore the data and uncover market insights.
        """
    )
    st.markdown(
        "[View Dataset on Kaggle](https://www.kaggle.com/datasets/bryan2k19/dutch-house-prices-dataset)"
    )

# ------------------
# 2. Property Exploration
# ------------------

elif page == "Property Exploration":

    st.title("Property Exploration")

    st.sidebar.header("Filters")

    selected_city = st.sidebar.selectbox(
        "City",
        sorted(df["City"].dropna().unique())
    )

    living_space = st.sidebar.slider(
        "Living Space (m²)",
        int(df["Living space size (m2)"].min()),
        int(df["Living space size (m2)"].max()),
        (
            int(df["Living space size (m2)"].min()),
            int(df["Living space size (m2)"].max())
        )
    )

    rooms_filter = st.sidebar.selectbox(
        "Number of Rooms",
        sorted(df["Rooms"].unique())
    )

    filtered_df = df[
        (df["City"] == selected_city) &
        (df["Living space size (m2)"] >= living_space[0]) &
        (df["Living space size (m2)"] <= living_space[1]) &
        (df["Rooms"].astype(int) == rooms_filter)
    ]

    st.markdown(
        f"**Properties matching selected filters: {len(filtered_df)}**"
    )

    # ==================================================
    # PROPERTY PRICE DISTRIBUTION
    # ==================================================

    st.plotly_chart(
        px.histogram(
            filtered_df,
            x="Price",
            nbins=40,
            title="Property Price Distribution"
        ),
        use_container_width=True
    )

    # ==================================================
    # PROPERTY PRICE VS LIVING SPACE
    # ==================================================

    st.subheader("Property Price vs Living Space")

    st.plotly_chart(
        px.scatter(
            filtered_df,
            x="Living space size (m2)",
            y="Price",
            trendline="ols",
            title="Relationship Between Living Space and Property Price"
        ),
        use_container_width=True
    )

    # ==================================================
    # TOP 10 CITIES BY AVERAGE PROPERTY PRICE
    # ==================================================

    st.subheader("Top 10 Cities by Average Property Price")

    top_cities = (
        df.groupby("City")["Price"]
        .mean()
        .reset_index()
        .sort_values("Price", ascending=False)
        .head(10)
    )

    st.plotly_chart(
        px.bar(
            top_cities,
            x="City",
            y="Price",
            title="Top 10 Cities by Average Property Price"
        ),
        use_container_width=True
    )

# ==================================================
# PRICE PREDICTION
# ==================================================

elif page == "Price Prediction":

    st.title("Property Price Prediction")

    st.markdown("""
    This section demonstrates a supervised machine learning approach for residential property valuation.

    The model estimates property prices using a combination of structural housing features and local market indicators:

    - Living Space Size (m²)
    - Lot Size (m²)
    - Number of Rooms
    - Build Year
    - Neighbourhood Market Price per m²

    These variables capture both property-specific characteristics and local market conditions that influence residential property values.

    Users can provide property attributes and obtain an estimated property price based on historical Dutch housing market data.
    """)

    # ==================================================
    # MODEL TRAINING
    # ==================================================

    X = df[
        [
            "Living space size (m2)",
            "Lot size (m2)",
            "Rooms",
            "Build year",
            "Estimated neighbourhood price per m2"
        ]
    ]

    y = df["Price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = LinearRegression()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # ==================================================
    # MODEL PERFORMANCE
    # ==================================================

    st.markdown(
        f"""
        ### Model Performance

        - R² Score: {r2_score(y_test, y_pred):.3f}
        - Mean Absolute Error (MAE): €{mean_absolute_error(y_test, y_pred):,.0f}
        """
    )

    # ==================================================
    # PREDICTED VS ACTUAL PRICES
    # ==================================================

    st.subheader("Predicted vs Actual Property Prices")

    pred_df = pd.DataFrame({
        "Actual Price": y_test,
        "Predicted Price": y_pred
    })

    fig = px.scatter(
        pred_df,
        x="Actual Price",
        y="Predicted Price",
        trendline="ols",
        labels={
            "Actual Price": "Actual Property Price (€)",
            "Predicted Price": "Predicted Property Price (€)"
        },
        title="Predicted vs Actual Property Prices"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ==================================================
    # INTERACTIVE PRICE PREDICTION
    # ==================================================

    st.subheader("Enter Property Details and Generate Prediction")

    living_space_input = st.number_input(
        "Living Space Size (m²)",
        min_value=20,
        max_value=1000,
        value=120
    )

    lot_size_input = st.number_input(
        "Lot Size (m²)",
        min_value=20,
        max_value=5000,
        value=300
    )

    rooms_input = st.selectbox(
        "Number of Rooms",
        sorted(df["Rooms"].astype(int).unique())
    )

    build_year_input = st.number_input(
        "Build Year",
        min_value=int(df["Build year"].min()),
        max_value=int(df["Build year"].max()),
        value=2000
    )

    neighbourhood_price_input = st.number_input(
        "Neighbourhood Market Price per m²",
        min_value=int(df["Estimated neighbourhood price per m2"].min()),
        max_value=int(df["Estimated neighbourhood price per m2"].max()),
        value=int(df["Estimated neighbourhood price per m2"].median())
    )

    prediction_df = pd.DataFrame({
        "Living space size (m2)": [living_space_input],
        "Lot size (m2)": [lot_size_input],
        "Rooms": [rooms_input],
        "Build year": [build_year_input],
        "Estimated neighbourhood price per m2": [neighbourhood_price_input]
    })

    pred_price = model.predict(prediction_df)[0]

    st.success(
        f"Estimated Property Price: €{pred_price:,.0f}"
    )

