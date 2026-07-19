"""
Stock Market Prediction - End to End ML App (Streamlit)
---------------------------------------------------------
Steps implemented (each is a sidebar button/page):
1. Choose a Dataset (upload CSV)
2. Understand the Dataset
3. Data Cleaning
4. Exploratory Data Analysis (EDA)
5. Feature Engineering
6. Train/Test Split
7. Train & Evaluate Multiple ML Models
8. Save the Best Model
9. Give Input & Get Prediction

Run with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import io
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ----------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------
st.set_page_config(page_title="Stock Market Prediction ML App", layout="wide")

# ----------------------------------------------------------------------
# SESSION STATE INITIALIZATION
# (keeps data alive as you click between steps)
# ----------------------------------------------------------------------
defaults = {
    "df_raw": None,
    "df_clean": None,
    "df_features": None,
    "encoders": {},
    "target_col": None,
    "feature_cols": [],
    "X_train": None, "X_test": None, "y_train": None, "y_test": None,
    "scaler": None,
    "results": None,
    "best_model_name": None,
    "best_model": None,
    "models_trained": {},
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

MODEL_PATH = "best_stock_model.joblib"

# ----------------------------------------------------------------------
# SIDEBAR : FILE UPLOAD + STEP NAVIGATION BUTTONS
# ----------------------------------------------------------------------
st.sidebar.title("📈 Stock Prediction Pipeline")

uploaded_file = st.sidebar.file_uploader("Upload Stock CSV Dataset", type=["csv"])
if uploaded_file is not None:
    if st.session_state.df_raw is None or st.sidebar.button("🔄 Reload uploaded file"):
        st.session_state.df_raw = pd.read_csv(uploaded_file)
        st.sidebar.success(f"Loaded: {uploaded_file.name}")

st.sidebar.markdown("---")
st.sidebar.subheader("Pipeline Steps")

steps = [
    "1. Choose Dataset",
    "2. Understand Dataset",
    "3. Data Cleaning",
    "4. EDA",
    "5. Feature Engineering",
    "6. Train/Test Split",
    "7. Train & Evaluate Models",
    "8. Save Best Model",
    "9. Predict (Input -> Output)",
]

if "current_step" not in st.session_state:
    st.session_state.current_step = steps[0]

for s in steps:
    if st.sidebar.button(s, use_container_width=True):
        st.session_state.current_step = s

st.sidebar.markdown("---")
st.sidebar.caption("Upload a file, then click through steps 1-9 in order.")

step = st.session_state.current_step
st.title("Stock Market Prediction — ML Pipeline")
st.subheader(step)

# ----------------------------------------------------------------------
# STEP 1 : CHOOSE DATASET
# ----------------------------------------------------------------------
if step == steps[0]:
    st.write("Use the **file uploader in the sidebar** to upload your stock market CSV "
             "(e.g. columns like Date, Open, High, Low, Close, Volume).")
    if st.session_state.df_raw is not None:
        st.success("Dataset loaded successfully!")
        st.dataframe(st.session_state.df_raw.head())
    else:
        st.info("No dataset uploaded yet.")

# ----------------------------------------------------------------------
# STEP 2 : UNDERSTAND DATASET
# ----------------------------------------------------------------------
elif step == steps[1]:
    df = st.session_state.df_raw
    if df is None:
        st.warning("Please upload a dataset first (Step 1).")
    else:
        st.markdown("### Dataset Shape")
        st.write(f"Rows: **{df.shape[0]}**, Columns: **{df.shape[1]}**")

        st.markdown("### Column Names")
        st.write(list(df.columns))

        st.markdown("### Dataset Info")
        buffer = io.StringIO()
        df.info(buf=buffer)
        st.text(buffer.getvalue())

        st.markdown("### Summary Statistics (describe())")
        st.dataframe(df.describe(include="all"))

        st.markdown("### Preview (head())")
        st.dataframe(df.head())

# ----------------------------------------------------------------------
# STEP 3 : DATA CLEANING
# ----------------------------------------------------------------------
elif step == steps[2]:
    df = st.session_state.df_raw
    if df is None:
        st.warning("Please upload a dataset first (Step 1).")
    else:
        st.write("Configure cleaning options, then click **Run Data Cleaning**.")

        missing_strategy = st.selectbox(
            "Handle missing values",
            ["Drop rows with missing values", "Fill numeric with mean", "Fill numeric with median", "Forward fill (ffill)"]
        )
        drop_dupes = st.checkbox("Remove duplicate rows", value=True)
        fix_date = st.checkbox("Auto-detect & convert a 'Date' column to datetime", value=True)

        if st.button("▶ Run Data Cleaning"):
            data = df.copy()

            # Correct dtypes: try to convert obvious date column
            if fix_date:
                for col in data.columns:
                    if "date" in col.lower():
                        try:
                            data[col] = pd.to_datetime(data[col])
                        except Exception:
                            pass

            # Handle missing values
            num_cols = data.select_dtypes(include=np.number).columns
            if missing_strategy == "Drop rows with missing values":
                data = data.dropna()
            elif missing_strategy == "Fill numeric with mean":
                data[num_cols] = data[num_cols].fillna(data[num_cols].mean())
            elif missing_strategy == "Fill numeric with median":
                data[num_cols] = data[num_cols].fillna(data[num_cols].median())
            elif missing_strategy == "Forward fill (ffill)":
                data = data.fillna(method="ffill").fillna(method="bfill")

            # Remove duplicates
            if drop_dupes:
                before = len(data)
                data = data.drop_duplicates()
                st.write(f"Removed {before - len(data)} duplicate rows.")

            st.session_state.df_clean = data
            st.success("Data cleaning complete!")

        if st.session_state.df_clean is not None:
            st.markdown("### Cleaned Data Preview")
            st.dataframe(st.session_state.df_clean.head())
            st.write("Remaining missing values per column:")
            st.write(st.session_state.df_clean.isnull().sum())

# ----------------------------------------------------------------------
# STEP 4 : EDA
# ----------------------------------------------------------------------
elif step == steps[3]:
    df = st.session_state.df_clean if st.session_state.df_clean is not None else st.session_state.df_raw
    if df is None:
        st.warning("Please upload a dataset first (Step 1).")
    else:
        num_df = df.select_dtypes(include=np.number)

        st.markdown("### Correlation Heatmap")
        if num_df.shape[1] >= 2:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.heatmap(num_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
            st.pyplot(fig)
        else:
            st.info("Not enough numeric columns for a correlation heatmap.")

        st.markdown("### Line Plot of a Numeric Column Over Row Index (e.g. Close Price Trend)")
        col_choice = st.selectbox("Choose a numeric column to plot", num_df.columns)
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.plot(df[col_choice].values, color="royalblue")
        ax2.set_xlabel("Time / Row Index")
        ax2.set_ylabel(col_choice)
        st.pyplot(fig2)

        st.markdown("### Distribution Plot")
        col_choice2 = st.selectbox("Choose a numeric column for distribution", num_df.columns, key="dist_col")
        fig3, ax3 = plt.subplots(figsize=(8, 4))
        sns.histplot(df[col_choice2], kde=True, ax=ax3, color="seagreen")
        st.pyplot(fig3)

        st.markdown("### Boxplot (Outlier Check)")
        col_choice3 = st.selectbox("Choose a numeric column for boxplot", num_df.columns, key="box_col")
        fig4, ax4 = plt.subplots(figsize=(8, 3))
        sns.boxplot(x=df[col_choice3], ax=ax4, color="orange")
        st.pyplot(fig4)

# ----------------------------------------------------------------------
# STEP 5 : FEATURE ENGINEERING
# ----------------------------------------------------------------------
elif step == steps[4]:
    df = st.session_state.df_clean
    if df is None:
        st.warning("Please complete Data Cleaning first (Step 3).")
    else:
        st.write("Select the **target column** you want to predict (e.g. `Close`) "
                 "and the **feature columns** to use as predictors.")

        all_cols = list(df.columns)
        datetime_cols = [c for c in all_cols if pd.api.types.is_datetime64_any_dtype(df[c])]
        target_col = st.selectbox(
            "🎯 Target column (what to predict)",
            [c for c in all_cols if c not in datetime_cols]
        )
        # Datetime columns (e.g. 'Date') are excluded from the pickable feature
        # list entirely, since raw datetimes can't be fed into most models/scalers.
        feature_options = [c for c in all_cols if c != target_col and c not in datetime_cols]
        feature_cols = st.multiselect(
            "🧩 Feature columns (predictors)",
            feature_options,
            default=feature_options[:5]
        )
        if datetime_cols:
            st.caption(f"Note: datetime column(s) {datetime_cols} were excluded from feature selection "
                       f"automatically (they can't be scaled directly).")

        add_ma = st.checkbox("Add Moving Average features (5-day & 10-day) on target column", value=True)
        do_label_encode = st.checkbox("Apply Label Encoding to categorical (object) columns", value=True)

        if st.button("▶ Apply Feature Engineering"):
            data = df.copy()

            # Moving average features based on target column
            if add_ma:
                data[f"{target_col}_MA5"] = data[target_col].rolling(window=5).mean()
                data[f"{target_col}_MA10"] = data[target_col].rolling(window=10).mean()
                if f"{target_col}_MA5" not in feature_cols:
                    feature_cols.append(f"{target_col}_MA5")
                if f"{target_col}_MA10" not in feature_cols:
                    feature_cols.append(f"{target_col}_MA10")
                data = data.dropna()

            # Label Encoding for categorical/object columns among selected features
            encoders = {}
            if do_label_encode:
                for col in feature_cols:
                    if data[col].dtype == "object":
                        le = LabelEncoder()
                        data[col] = le.fit_transform(data[col].astype(str))
                        encoders[col] = le

            st.session_state.df_features = data
            st.session_state.target_col = target_col
            st.session_state.feature_cols = feature_cols
            st.session_state.encoders = encoders

            st.success("Feature engineering complete!")
            st.write("Target column:", target_col)
            st.write("Feature columns:", feature_cols)
            st.dataframe(data[feature_cols + [target_col]].head())

# ----------------------------------------------------------------------
# STEP 6 : TRAIN/TEST SPLIT
# ----------------------------------------------------------------------
elif step == steps[5]:
    data = st.session_state.df_features
    if data is None:
        st.warning("Please complete Feature Engineering first (Step 5).")
    else:
        test_size = st.slider("Test set size (%)", 10, 40, 20) / 100.0
        shuffle_data = st.checkbox("Shuffle data before splitting", value=False,
                                    help="For time-series stock data, usually keep this OFF to preserve chronological order.")
        scale_features = st.checkbox("Standardize features (StandardScaler)", value=True)

        if st.button("▶ Split Dataset"):
            safe_feature_cols = [
                c for c in st.session_state.feature_cols
                if not pd.api.types.is_datetime64_any_dtype(data[c])
            ]
            dropped = set(st.session_state.feature_cols) - set(safe_feature_cols)
            if dropped:
                st.warning(f"Dropped non-numeric/datetime column(s) from features: {dropped}")
            st.session_state.feature_cols = safe_feature_cols

            X = data[safe_feature_cols]
            y = data[st.session_state.target_col]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, shuffle=shuffle_data, random_state=42
            )

            scaler = None
            if scale_features:
                scaler = StandardScaler()
                X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns, index=X_train.index)
                X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns, index=X_test.index)

            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            st.session_state.scaler = scaler

            st.success("Dataset split complete!")
            st.write(f"Training samples: {X_train.shape[0]} | Testing samples: {X_test.shape[0]}")

        if st.session_state.X_train is not None:
            st.markdown("### Training Features Preview")
            st.dataframe(st.session_state.X_train.head())

# ----------------------------------------------------------------------
# STEP 7 : TRAIN & EVALUATE MULTIPLE MODELS
# ----------------------------------------------------------------------
elif step == steps[6]:
    if st.session_state.X_train is None:
        st.warning("Please complete the Train/Test Split first (Step 6).")
    else:
        st.write("Select which models to train and compare:")
        c1, c2 = st.columns(2)
        with c1:
            use_lr = st.checkbox("Linear Regression", value=True)
            use_dt = st.checkbox("Decision Tree Regressor", value=True)
        with c2:
            use_rf = st.checkbox("Random Forest Regressor", value=True)
            use_gb = st.checkbox("Gradient Boosting Regressor", value=True)
        use_svr = st.checkbox("Support Vector Regressor (SVR)", value=False)

        if st.button("▶ Train & Evaluate Models"):
            X_train, X_test = st.session_state.X_train, st.session_state.X_test
            y_train, y_test = st.session_state.y_train, st.session_state.y_test

            candidates = {}
            if use_lr: candidates["Linear Regression"] = LinearRegression()
            if use_dt: candidates["Decision Tree"] = DecisionTreeRegressor(random_state=42)
            if use_rf: candidates["Random Forest"] = RandomForestRegressor(n_estimators=200, random_state=42)
            if use_gb: candidates["Gradient Boosting"] = GradientBoostingRegressor(random_state=42)
            if use_svr: candidates["SVR"] = SVR()

            results = []
            trained = {}
            progress = st.progress(0)
            for i, (name, model) in enumerate(candidates.items()):
                model.fit(X_train, y_train)
                preds = model.predict(X_test)

                r2 = r2_score(y_test, preds)
                mae = mean_absolute_error(y_test, preds)
                rmse = np.sqrt(mean_squared_error(y_test, preds))

                results.append({"Model": name, "R2 Score": r2, "MAE": mae, "RMSE": rmse})
                trained[name] = model
                progress.progress((i + 1) / len(candidates))

            results_df = pd.DataFrame(results).sort_values("R2 Score", ascending=False).reset_index(drop=True)
            st.session_state.results = results_df
            st.session_state.models_trained = trained

            best_name = results_df.iloc[0]["Model"]
            st.session_state.best_model_name = best_name
            st.session_state.best_model = trained[best_name]

            st.success(f"Training complete! Best model: **{best_name}**")

        if st.session_state.results is not None:
            st.markdown("### Model Comparison")
            st.dataframe(st.session_state.results.style.highlight_max(subset=["R2 Score"], color="lightgreen"))

            fig, ax = plt.subplots(figsize=(8, 4))
            sns.barplot(data=st.session_state.results, x="Model", y="R2 Score", ax=ax, palette="viridis")
            ax.set_title("R2 Score Comparison")
            plt.xticks(rotation=20)
            st.pyplot(fig)

            st.info(f"🏆 Best performing model: **{st.session_state.best_model_name}**")

# ----------------------------------------------------------------------
# STEP 8 : SAVE BEST MODEL
# ----------------------------------------------------------------------
elif step == steps[7]:
    if st.session_state.best_model is None:
        st.warning("Please train and evaluate models first (Step 7).")
    else:
        st.write(f"Best model selected: **{st.session_state.best_model_name}**")
        if st.button("💾 Save Best Model"):
            bundle = {
                "model": st.session_state.best_model,
                "scaler": st.session_state.scaler,
                "encoders": st.session_state.encoders,
                "feature_cols": st.session_state.feature_cols,
                "target_col": st.session_state.target_col,
            }
            joblib.dump(bundle, MODEL_PATH)
            st.success(f"Model saved to `{MODEL_PATH}`")

        if os.path.exists(MODEL_PATH):
            with open(MODEL_PATH, "rb") as f:
                st.download_button("⬇ Download Saved Model (.joblib)", f, file_name=MODEL_PATH)

# ----------------------------------------------------------------------
# STEP 9 : PREDICT (INPUT -> OUTPUT)
# ----------------------------------------------------------------------
elif step == steps[8]:
    if not os.path.exists(MODEL_PATH) and st.session_state.best_model is None:
        st.warning("Please train and save a model first (Steps 7 & 8).")
    else:
        # Load bundle either from session or disk
        if os.path.exists(MODEL_PATH):
            bundle = joblib.load(MODEL_PATH)
        else:
            bundle = {
                "model": st.session_state.best_model,
                "scaler": st.session_state.scaler,
                "encoders": st.session_state.encoders,
                "feature_cols": st.session_state.feature_cols,
                "target_col": st.session_state.target_col,
            }

        model = bundle["model"]
        scaler = bundle["scaler"]
        encoders = bundle["encoders"]
        feature_cols = bundle["feature_cols"]
        target_col = bundle["target_col"]

        st.write(f"Enter values for the features below to predict **{target_col}**:")

        input_vals = {}
        cols = st.columns(2)
        for i, col in enumerate(feature_cols):
            with cols[i % 2]:
                if col in encoders:
                    options = list(encoders[col].classes_)
                    val = st.selectbox(col, options)
                    input_vals[col] = encoders[col].transform([val])[0]
                else:
                    val = st.number_input(col, value=0.0, format="%.4f")
                    input_vals[col] = val

        if st.button("🔮 Predict"):
            input_df = pd.DataFrame([input_vals])[feature_cols]
            if scaler is not None:
                input_df = pd.DataFrame(scaler.transform(input_df), columns=feature_cols)
            prediction = model.predict(input_df)[0]
            st.success(f"### Predicted {target_col}: **{prediction:.4f}**")