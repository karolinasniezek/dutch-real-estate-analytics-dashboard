# Dutch Real Estate Analytics Dashboard

## Project Overview

Dutch Real Estate Analytics Dashboard is an interactive data analytics application built with Streamlit for exploring residential property listings in the Netherlands.

The project combines exploratory data analysis, interactive visualizations, and machine learning techniques to help users analyze housing market trends and estimate residential property values based on property characteristics and local market indicators.

---

## Objectives

The main objectives of this project are:

* Explore residential real estate data from the Dutch housing market
* Identify factors influencing property prices
* Provide interactive filtering and visualization capabilities
* Develop a predictive model for residential property valuation
* Demonstrate practical applications of data analytics and machine learning

---

## Dataset

Source:

Dutch House Prices Dataset (Kaggle)

https://www.kaggle.com/datasets/bryan2k19/dutch-house-prices-dataset

The dataset contains information about residential properties listed for sale in the Netherlands, including:

* Property price
* Living space size
* Lot size
* Number of rooms
* Construction year
* Energy label
* Property type
* City
* Neighbourhood market price indicators

---

## Dashboard Sections

### 1. Market Overview

Provides an introduction to the Dutch housing market and presents the objectives of the analysis.

Key questions addressed:

* Which cities have the highest property prices?
* How does living space influence housing value?
* What patterns exist within the Dutch housing market?
* Which property characteristics are most associated with higher prices?

---

### 2. Property Exploration

Interactive exploratory analysis section allowing users to:

* Filter properties by city
* Filter properties by living space size
* Filter properties by number of rooms

Visualizations include:

* Property Price Distribution (Histogram)
* Property Price vs Living Space (Scatter Plot)
* Top 10 Cities by Average Property Price (Bar Chart)

---

### 3. Property Price Prediction

A machine learning module that estimates residential property values using Linear Regression.

Model features:

* Living Space Size (m²)
* Lot Size (m²)
* Number of Rooms
* Build Year
* Neighbourhood Market Price per m²

Model evaluation metrics:

* R² Score
* Mean Absolute Error (MAE)

Users can interactively enter property characteristics and generate estimated property prices.

---

## Data Cleaning & Preprocessing

Several preprocessing steps were applied before analysis:

* Missing value removal
* Currency formatting cleanup
* Area unit conversion
* Room count extraction from textual descriptions
* Build year normalization
* Conversion of numerical variables into machine-learning-ready formats

---

## Technology Stack

* Python
* Streamlit
* Pandas
* Plotly
* Scikit-learn

---

## Machine Learning

Algorithm:

* Linear Regression

Evaluation Metrics:

* R² Score
* Mean Absolute Error (MAE)

The model was trained on historical Dutch housing market data and serves as a proof-of-concept valuation tool.

---

## Project Structure

```text
Dutch-Real-Estate-Analytics/
│
├── data/
│   └── raw_dutch_real_estate.csv
│
├── src/
│   └── app.py
│
├── requirements.txt
│
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/karolinasniezek/dutch-real-estate-analytics-dashboard.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run src/app.py
```

---

## Future Improvements

Potential enhancements include:

* Advanced regression models (Random Forest, XGBoost)
* Geographical visualizations using maps
* Property type segmentation
* Time-series market analysis
* Model deployment and cloud hosting
