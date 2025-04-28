import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import shap
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.pyplot as plt

# Make all figures and axes transparent by default
plt.rcParams['figure.facecolor']  = 'none'   # figure background
plt.rcParams['axes.facecolor']    = 'none'   # plotting area background
plt.rcParams['savefig.facecolor'] = 'none'   # saved files background
plt.rcParams['savefig.transparent'] = True   # for plt.savefig(...)


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
local_css("style.css")

# List of the 10 selected predictor columns
selected_features = [
    'SP_Ajclose',
    'DJ_Ajclose',
    'OF_Price',
    'SF_Price',
    'PLT_Price',
    'PLD_Price',
    'USB_Price',
    'USDI_Price',
    'EU_Price',
    'GDX_Adj Close'
]
# 2️⃣ Cached loader that also subsets
@st.cache_data
def load_data(path, selected_cols):
    df = pd.read_csv(path, parse_dates=['Date'])
    df.sort_values('Date', inplace=True)
    return df[selected_cols]

# 3️⃣ Call it immediately so df is your slimmed‐down DataFrame
selected_cols = selected_features + ['Adj Close', 'Date']
df = load_data('FINAL_USO.csv', selected_cols)

# 2. Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Business Case", "Data Visualization", "Predictions", "Explainability", "Hyperparameter Tuning"])

# --- Page 1: Business Case & Data Presentation ---
if page == "Business Case":
    st.title("Business Case & Data Overview")
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    # Objective
    st.markdown("""
    <p style="font-size:25px; font-weight:bold;">Objective:</p >
    <p style="font-size:18px;">
    Build a forecasting dashboard to predict the Adjusted Close price of a Gold ETF using key economic and financial indicators.
    </p >
    """, unsafe_allow_html=True)

    # Purpose
    st.markdown("""
    <p style="font-size:25px; font-weight:bold;">Purpose:</p >
    <p style="font-size:18px;">
    Empower investors and financial analysts to make informed buy/sell decisions for gold. Provide insights into what drives gold price movements.
    </p >
    """, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Machine Learning Models
    st.markdown("""
    <p style="font-size:25px; font-weight:bold;">Our machine learning models:</p >
    <ul style="font-size:18px;">
        <li>Linear Regression (baseline, interpretable).</li>
        <li>Decision Tree Regressor (captures non-linear patterns).</li>
    </ul>
    """, unsafe_allow_html=True)

    # Key Features
    st.markdown("""
    <p style="font-size:25px; font-weight:bold;">Key Features (x variables):</p >
    <ul style="font-size:18px;">
        <li><b>SP_Ajclose & DJ_Ajclose:</b> Reflect market sentiment (S&P 500 & Dow Jones).</li>
        <li><b>OF_Price:</b> Tracks oil prices, a signal of inflationary pressure.</li>
        <li><b>SF_Price, PLT_Price, PLD_Price:</b> Prices of silver, platinum, palladium—precious metals that often move with gold.</li>
        <li><b>USB_Price:</b> 10-Year US Bond Price, a proxy for real interest rates.</li>
        <li><b>USDI_Price & EU_Price:</b> Reflect currency strength (US Dollar Index, Euro/USD exchange rate).</li>
        <li><b>GDX_Adj Close:</b> Gold Miners ETF, a leveraged signal for gold market trends.</li>
    </ul>
    """, unsafe_allow_html=True)


    st.image("stock.png")

    st.subheader("Dataset Preview")
    st.write(df.head())
    st.write("Shape:", df.shape)

    # 1️⃣ Prepare a single-row table of feature names
    features = selected_features

    # 2️⃣ Build a DataFrame with one row labeled "Feature Name"
    df_features = pd.DataFrame(
        [features],
        index=["Feature Name"],
        columns=[f"Col {i+1}" for i in range(len(features))]
    )

    # 3️⃣ Display as a neatly formatted table
    st.subheader("Selected Features")
    st.table(df_features)


# --- Page 2: Data Visualization ---
elif page == "Data Visualization":
    st.title("Data Visualization")

    # 4️⃣ Annual Bar Chart: Average Gold ETF Price per Year
    st.subheader("Annual Average Gold ETF Price")
    df_year = df.copy()
    df_year["Year"] = df_year["Date"].dt.year
    annual_avg = df_year.groupby("Year")["Adj Close"].mean()
    fig_bar, ax_bar = plt.subplots(figsize=(10, 5))
    ax_bar.bar(
        annual_avg.index.astype(str),
        annual_avg.values,
        color="#fddc5c",
        edgecolor="#d4af37"
    )
    ax_bar.set_title("Average Adjusted Close Price by Year")
    ax_bar.set_xlabel("Year")
    ax_bar.set_ylabel("Average Adj Close")
    for idx, val in enumerate(annual_avg.values):
        ax_bar.text(idx, val * 1.01, f"{val:.0f}", ha="center")
    st.pyplot(fig_bar)
    st.markdown("")
    st.markdown("")
    st.markdown("")

    # 3️⃣ Seaborn Regression Plot: Gold vs. Oil
    st.subheader("Gold vs. Oil Price Relationship")
    fig_reg, ax_reg = plt.subplots(figsize=(10, 5))
    sns.regplot(
        x="OF_Price",
        y="Adj Close",
        data=df,
        ax=ax_reg,
        line_kws={"color": "#d4af37"},
        scatter_kws={"alpha": 0.6}
    )
    ax_reg.set_title("Gold ETF Adj Close vs. Oil Futures Price")
    ax_reg.set_xlabel("Oil Futures Price")
    ax_reg.set_ylabel("Gold ETF Adj Close")
    st.pyplot(fig_reg)
    st.markdown("")
    st.markdown("")
    st.markdown("")

    # 1️⃣ Time Series Plots
    st.subheader("Time Series Plots")
    cols_to_plot = ['Adj Close', 'SP_Ajclose', 'OF_Price']
    fig_ts, ax_ts = plt.subplots(figsize=(10, 5))
    for col in cols_to_plot:
        ax_ts.plot(df['Date'], df[col], label=col)
    ax_ts.legend()
    ax_ts.set_xlabel("Date")
    ax_ts.set_ylabel("Price")
    st.pyplot(fig_ts)
    st.markdown("")
    st.markdown("")
    st.markdown("")

    # 2️⃣ Correlation Heatmap with Annotations
    st.subheader("Correlation Heatmap with Annotations")
    corr = df.select_dtypes(include='number').corr()
    fig_heat, ax_heat = plt.subplots(figsize=(12, 8))
    sns.heatmap(
        corr,
        cmap='coolwarm',
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 8},
        linewidths=0.5,
        linecolor='white',
        ax=ax_heat
    )
    st.pyplot(fig_heat)
    

# --- Page 3: Predictions ---
elif page == "Predictions":
    st.title("Model Predictions")
    # Prepare features and target
    features = st.multiselect("Select predictor features", options=[c for c in df.columns if c not in ['Date', 'Adj Close']], default=['SP_Ajclose', 'OF_Price', 'SF_Price', 'PLT_Price'])
    X = df[features]
    y = df['Adj Close']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model_choice = st.radio("Choose model", ["Linear Regression", "Decision Tree"])
    if model_choice == "Linear Regression":
        model = LinearRegression()
    else:
        max_depth = st.slider("Max depth", 2, 20, 5)
        model = DecisionTreeRegressor(max_depth=max_depth)

    if st.button("Train & Evaluate"):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        st.write("**Metrics**")
        st.write("MSE:", mean_squared_error(y_test, y_pred))
        st.write("MAE:", mean_absolute_error(y_test, y_pred))
        st.write("R²:", r2_score(y_test, y_pred))

        st.subheader("Actual vs Predicted")
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        ax3.plot(df['Date'].iloc[len(X_train):], y_test, label="Actual")
        ax3.plot(df['Date'].iloc[len(X_train):], y_pred, label="Predicted")
        ax3.legend(); ax3.set_xlabel("Date"); ax3.set_ylabel("Adj Close")
        st.pyplot(fig3)

# --- Page 4: Explainability ---
elif page == "Explainability":
    st.title("Explainable AI")
    # Train decision tree for SHAP
    features = ['SP_Ajclose', 'OF_Price', 'SF_Price', 'PLT_Price']
    X = df[features]
    y = df['Adj Close']
    model = DecisionTreeRegressor(max_depth=5)
    model.fit(X, y)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    st.subheader("SHAP Summary Plot")
    fig4 = plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, show=False)          # draw on current figure
    st.pyplot(fig4)

    st.subheader("SHAP Dependence Plot")
    feature = st.selectbox("Feature for dependence plot", features)

    fig5 = plt.figure(figsize=(10, 6))
    shap.dependence_plot(feature, shap_values, X, show=False)  # render on fig5
    st.pyplot(fig5)

# --- Page 5: Hyperparameter Tuning ---
elif page == "Hyperparameter Tuning":
    st.title("Hyperparameter Tuning with MLflow")

    mlflow.set_experiment("gold_price_forecast")
    features = ['SP_Ajclose', 'OF_Price', 'SF_Price', 'PLT_Price']
    X = df[features]
    y = df['Adj Close']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    max_depths = st.multiselect("Max depths to try", options=[3, 5, 7, 10, 15], default=[5, 10])
    min_samples_splits = st.multiselect("Min samples splits", options=[2, 5, 10], default=[2, 5])

    if st.button("Run Tuning"):
        best_rmse = float('inf')
        best_params = {}
        for md in max_depths:
            for mss in min_samples_splits:
                with mlflow.start_run():
                    mlflow.log_param("max_depth", md)
                    mlflow.log_param("min_samples_split", mss)
                    model = DecisionTreeRegressor(max_depth=md, min_samples_split=mss)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    # compute MSE
                    mse = mean_squared_error(y_test, y_pred)
                    # take square root to get RMSE
                    rmse = np.sqrt(mse)
                    mlflow.log_metric("rmse", rmse)
                    mlflow.sklearn.log_model(model, "model")
                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_params = {"max_depth": md, "min_samples_split": mss}

        st.write("Best RMSE:", best_rmse)
        st.write("Best Params:", best_params)
        st.write("View more runs in your MLflow UI.")

    st.markdown("MLflow automatically logs parameters, metrics, and models during tuning.")

