import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from statsmodels.tsa.api import SimpleExpSmoothing, Holt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import plotly.express as px
import plotly.graph_objects as go
from xgboost import XGBClassifier, plot_importance  # Import necessary XGBoost components
import xgboost as xgb  # For xgb_model.get_booster()

import warnings
import os

warnings.filterwarnings("ignore")

# === Sidebar Navigation ===
st.set_page_config(page_title="Bogotá School Analytics", layout="wide")
section = st.sidebar.selectbox(
    "Select Section",
    ["1. School Classification", "2. Time Series Prediction", "3. School Ranking Prediction"]
)

# ---
# ## Section 1: School Classification
# ---
if section == "1. School Classification":
    st.title("🎯 School Ranking Classifier")
    st.info("This section classifies schools into rankings (A+, A, B, C, D) using a Random Forest model.")

    # === Load and clean data ===
    file_path = r"C:\Study document\6611 Dataset\Final\MASTER_CLEANED1.xlsx"
    df = pd.read_excel(file_path)
    # Apply standardization immediately after loading
    df.columns = df.columns.str.strip().str.title()

    # Filter and clean
    df = df[df['Ranking'].isin(['A+', 'A', 'B', 'C', 'D'])].dropna(subset=['Ranking'])
    df['Sector'] = df['Sector'].str.upper().str.strip()
    df['Sector'] = df['Sector'].apply(lambda x: 'PRIVATE' if 'PRIVATE' in x else 'PUBLIC')
    # Use standardized column name
    if 'Location Name' in df.columns:
        df['Location Name'] = df['Location Name'].astype(str).str.upper().str.strip()
    else:
        st.error("Missing 'Location Name' column in Section 1. Please check your Excel file.")
        st.stop()

    # Define features
    categorical_cols = ['Concession', 'Sector', 'Location Name']
    numerical_cols = [
        "Enrolled Students (Last 3 Years)",
        "Mathematics Index", "Natural Sciences Index", "Social And Citizenship Index",
        "Critical Reading Index", "English Index", "Total Index"
    ]
    df = df.dropna(subset=numerical_cols)

    # Encode target label
    label_encoder = LabelEncoder()
    df['Ranking_Label'] = label_encoder.fit_transform(df['Ranking'])

    # Preprocessing
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer([
        ("num", num_pipeline, numerical_cols),
        ("cat", cat_pipeline, categorical_cols)
    ])

    # Train/Test split
    X = df[categorical_cols + numerical_cols]
    y = df["Ranking_Label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    # Build model pipeline
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    pipeline.fit(X_train, y_train)

    # === Sidebar Input Form ===
    st.sidebar.header("📥 Enter New School Data")
    with st.sidebar.form("input_form"):
        user_input = {
            "Concession": st.selectbox("Concession", df["Concession"].dropna().unique()),
            "Sector": st.selectbox("Sector", df["Sector"].dropna().unique()),
            "Location Name": st.selectbox("Location", df["Location Name"].dropna().unique()),  # Use standardized name
            "Enrolled Students (Last 3 Years)": st.number_input("Enrolled students", value=100),
            "Mathematics Index": st.slider("Math Index", 0.0, 1.0, 0.5),
            "Natural Sciences Index": st.slider("Science Index", 0.0, 1.0, 0.5),
            "Social And Citizenship Index": st.slider("Social Index", 0.0, 1.0, 0.5),
            "Critical Reading Index": st.slider("Reading Index", 0.0, 1.0, 0.5),
            "English Index": st.slider("English Index", 0.0, 1.0, 0.5),
            "Total Index": st.slider("Total Index", 0.0, 1.0, 0.5)
        }
        submitted = st.form_submit_button("🔮 Predict Ranking")

    # === Output Prediction and Feature Importance ===
    if submitted:
        user_df = pd.DataFrame([user_input])
        pred = pipeline.predict(user_df)[0]
        pred_label = label_encoder.inverse_transform([pred])[0]
        st.success(f"🎯 Predicted School Ranking: **{pred_label}**")

        raw_feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()


        def clean_feature_name(raw):
            if raw.startswith("num__"):
                return raw.replace("num__", "").replace("_", " ").title()
            elif raw.startswith("cat__"):
                if "Sector_" in raw:
                    return "Sector"
                elif "Location Name_" in raw:  # Use standardized name
                    return "Location"
                elif "Concession_" in raw:
                    return "Concession"
                else:
                    return "Category"
            return raw.title()


        cleaned_features = [clean_feature_name(f) for f in raw_feature_names]

        model = pipeline.named_steps['classifier']
        importances = model.feature_importances_
        importance_df = pd.DataFrame({
            "Feature": cleaned_features,
            "Importance": importances
        })
        importance_df = importance_df.groupby("Feature").sum().reset_index()
        importance_df = importance_df.sort_values("Importance", ascending=False).head(10)

        fig = px.bar(importance_df, x="Importance", y="Feature", orientation="h",
                     title="🔍 Top 10 Feature Importances", color="Importance",
                     color_continuous_scale="Cividis")
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

# ---
# ## Section 2: Time Series Prediction
# ---
elif section == "2. Time Series Prediction":
    st.title("📈 School Indicator Time Series Predictor")

    DATA_FILE = r"C:\Users\linht\PycharmProjects\pythonProject3\MASTER_CLEANED1.xlsx"  # Adjusted path based on typical project structure
    PREDICTION_YEARS = [2025, 2026, 2027]
    INDICATORS = [
        'Total Index', 'Pass Rate', 'Dropout Rate',  # Use standardized names
        'Failure Rate', 'Enrollment', 'Absorption Rate'  # Use standardized names
    ]


    @st.cache_data
    def load_data():
        if not os.path.exists(DATA_FILE):
            st.error(f"File not found: {DATA_FILE}")
            st.stop()
        df = pd.read_excel(DATA_FILE, dtype={'Dane Code': str})

        # Standardize column names immediately after loading
        df.columns = df.columns.str.strip().str.title()

        # Now, rename using the *standardized* source column names
        df.rename(columns={
            'Enrolled Students (Last 3 Years)': 'Enrollment',  # This should now exist after .title()
            'Absorption Rate In Higher Education': 'Absorption Rate',  # This should now exist after .title()
        }, inplace=True)

        if 'Year' in df.columns:
            df['Year'] = pd.to_numeric(df['Year'], errors='coerce').fillna(-1).astype(int)
            df = df[df['Year'] != -1]
        else:
            st.error("Missing 'Year' column in the dataset. Cannot proceed.")
            st.stop()

        for col in INDICATORS:
            if col in df.columns:  # Check if column exists after standardization and renames
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace(',', '.', regex=False).str.replace('%', '', regex=False),
                    errors='coerce'
                )
                if col in ['Pass Rate', 'Dropout Rate', 'Failure Rate', 'Absorption Rate']:
                    df[col] /= 100
            else:
                st.warning(
                    f"Indicator column '{col}' not found in the dataset. It will not be available for prediction.")

        df['Dane Code'] = df['Dane Code'].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
        df.dropna(subset=['Dane Code'], inplace=True)
        return df


    def predict_time_series(series, future_years, indicator, dane):
        s = series.dropna()
        n = len(s)
        if n == 0:
            return pd.Series(np.nan, index=future_years)
        if n < 3:  # Need at least 3 points for meaningful forecasting, otherwise just extend last value
            return pd.Series(s.iloc[-1], index=future_years)

        try:
            model = ARIMA(s, order=(1, 1, 1))
            fit = model.fit()
            fc = fit.get_forecast(steps=len(future_years)).predicted_mean
            if fc.isnull().any() or np.isinf(fc).any():
                raise ValueError("NaN or Inf in ARIMA forecast.")
            return pd.Series(fc.values, index=future_years)
        except Exception as e:
            # st.warning(f"ARIMA failed for {dane} {indicator}: {e}") # Optional: uncomment for detailed debug
            pass

        try:
            fit = Holt(s, initialization_method="estimated").fit()
            fc = fit.forecast(len(future_years))
            if fc.isnull().any() or np.isinf(fc).any():
                raise ValueError("NaN or Inf in Holt forecast.")
            return pd.Series(fc.values, index=future_years)
        except Exception as e:
            # st.warning(f"Holt failed for {dane} {indicator}: {e}") # Optional: uncomment for detailed debug
            pass

        try:
            fit = SimpleExpSmoothing(s, initialization_method="estimated").fit()
            fc = fit.forecast(len(future_years))
            if fc.isnull().any() or np.isinf(fc).any():
                raise ValueError("NaN or Inf in SES forecast.")
            return pd.Series(fc.values, index=future_years)
        except Exception as e:
            # st.warning(f"SES failed for {dane} {indicator}: {e}") # Optional: uncomment for detailed debug
            pass

        # Fallback: if all models fail or data is insufficient, predict the last value
        return pd.Series(s.iloc[-1], index=future_years)


    df = load_data()  # Load the data with the corrected column titles

    dane_input = st.text_input("Enter DANE Code:").strip()

    if dane_input:
        school = df[df['Dane Code'] == dane_input].sort_values('Year')
        if school.empty:
            st.warning("No data for that DANE Code.")
        else:
            # Use the standardized column name "Name Of Establishment"
            if "Name Of Establishment" in school.columns:
                name = school["Name Of Establishment"].iloc[0]
            else:
                name = f"School with DANE Code: {dane_input}"  # Fallback name
                st.warning("'Name Of Establishment' column not found, using DANE Code as name.")

            st.subheader(f"{name} (DANE Code: {dane_input})")

            min_y, max_y = school['Year'].min(), school['Year'].max()
            years = list(range(min_y, max_y + 1)) + PREDICTION_YEARS
            results = pd.DataFrame(index=years)

            for ind in INDICATORS:
                if ind in school.columns:  # Ensure indicator column exists after load and renames
                    hist = school.set_index('Year')[ind].reindex(range(min_y, max_y + 1))
                    fc = predict_time_series(hist, PREDICTION_YEARS, ind, dane_input)
                    combined = pd.concat([hist, fc])
                    combined.index = combined.index.astype(int)
                    results[ind] = combined
                else:
                    st.warning(
                        f"Indicator '{ind}' not found in data for DANE Code {dane_input}. Skipping prediction for this indicator.")

            # Filter out indicators not found in the DataFrame from the dropdown list
            available_indicators = [ind for ind in INDICATORS if ind in results.columns]

            if available_indicators:
                choose = st.selectbox("Choose an indicator:", available_indicators)
                plot_df = results[choose].reset_index().rename(columns={'index': 'Year', choose: 'Value'})
                plot_df['Type'] = plot_df['Year'].apply(
                    lambda y: 'Prediction' if y in PREDICTION_YEARS else 'Historical')
                plot_df.dropna(subset=['Value'], inplace=True)

                if plot_df.empty:
                    st.warning("No data to plot for the selected indicator.")
                else:
                    fig = px.line(plot_df, x='Year', y='Value', line_dash='Type',
                                  title=f"{choose} trend for {name}")
                    fig.update_xaxes(dtick=1)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No valid indicators found for plotting.")

# ---
# ## Section 3: School Ranking Prediction
# ---
elif section == "3. School Ranking Prediction":
    st.title("📊 School Ranking Analysis & Projection")

    # --- File uploader and default example ---
    st.sidebar.header("📂 Upload Dataset")
    user_file = st.sidebar.file_uploader("Upload Excel File", type=["xlsx"])


    # --- Load default example ---
    @st.cache_data  # Added cache for this function
    def load_data_ranking():
        path = r"C:\Study document\6611 Dataset\Final\MASTER_CLEANED1.xlsx" # Adjusted path
        df_ranking = pd.read_excel(path)
        # Standardize column names here too
        df_ranking.columns = df_ranking.columns.str.strip().str.title()
        return df_ranking


    if user_file:
        df = pd.read_excel(user_file)
        df.columns = df.columns.str.strip().str.title()  # Apply here if user uploads
    else:
        df = load_data_ranking()  # Use the consistent loader
        st.info("Using example dataset. Upload your own file to analyze custom data.")

    # --- Preprocessing ---
    # Ensure 'Total Index' exists before dropping NaNs
    if 'Total Index' in df.columns:
        df = df.dropna(subset=['Total Index'])
    else:
        st.error("Missing 'Total Index' column, cannot proceed with ranking prediction.")
        st.stop()

    # Handle 'Year' column consistently
    # Check for 'Evaluated (Last 3 Years)' (standardized)
    if 'Evaluated (Last 3 Years)' in df.columns:
        if 'Year' not in df.columns:
            df['Year'] = pd.to_datetime(df['Evaluated (Last 3 Years)'], errors='coerce').dt.year.fillna(2024).astype(
                int)
        else:
            df['Year'] = df['Year'].astype(int)
    elif 'Year' in df.columns:
        df['Year'] = df['Year'].astype(int)
    else:
        st.warning("Missing 'Year' or 'Evaluated (Last 3 Years)' column. Defaulting Year to 2024.")
        df['Year'] = 2024  # Default year if no date column found

    # Columns to standardize and fillna (ensure these are the standardized names)
    cols_to_process = ['Concession', 'Ranking', 'Location Name']
    for col in cols_to_process:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].fillna("Unknown")
        else:
            st.warning(f"Column '{col}' not found for processing. Some features might be missing.")
            if col == 'Ranking':  # Ranking is critical for this section
                st.error("Missing 'Ranking' column. Cannot proceed with ranking prediction.")
                st.stop()

    label_enc_ranking = LabelEncoder()
    label_enc_concession = LabelEncoder()
    label_enc_location = LabelEncoder()

    if 'Ranking' in df.columns:
        df['Ranking_encoded'] = label_enc_ranking.fit_transform(df['Ranking'])
    else:
        st.error("Missing 'Ranking' column for encoding. Cannot proceed.")
        st.stop()

    if 'Concession' in df.columns:
        df['Concession_encoded'] = label_enc_concession.fit_transform(df['Concession'])
    else:
        st.warning("Missing 'Concession' column for encoding. Setting default to 0.")
        df['Concession_encoded'] = 0

    if 'Location Name' in df.columns:  # Use standardized name
        df['Location_encoded'] = label_enc_location.fit_transform(df['Location Name'])
    else:
        st.warning("Missing 'Location Name' column for encoding. Setting default to 0.")
        df['Location_encoded'] = 0

    # Features must use the standardized column names
    features = ['Year', 'Concession_encoded', 'Location_encoded', 'Mathematics Index',
                'Natural Sciences Index', 'Social And Citizenship Index',  # Standardized name
                'Critical Reading Index', 'English Index', 'Total Index']

    # Filter features that are actually in the DataFrame
    available_features = [f for f in features if f in df.columns]

    X = df[available_features].dropna()
    y = df.loc[X.index, 'Ranking_encoded']

    if X.empty or y.empty or len(y.unique()) < 2:  # Need at least 2 classes for classification
        st.warning("Not enough data or unique rankings to train the model after dropping NaNs and filtering features.")
        st.stop()

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss',
                              n_estimators=100, max_depth=5, random_state=42)
    xgb_model.fit(X_train, y_train)

    # --- Predict future years ---
    latest = df[df['Year'] == df['Year'].max()].copy()
    projected_years = [2025, 2026, 2027]
    projection_data = []

    for year in projected_years:
        future = latest.copy()
        future['Year'] = year

        if 'Concession' in future.columns:
            future['Concession_encoded'] = label_enc_concession.transform(future['Concession'])
        else:
            future['Concession_encoded'] = 0  # Default if column missing

        if 'Location Name' in future.columns:  # Use standardized name
            future['Location_encoded'] = label_enc_location.transform(future['Location Name'])
        else:
            future['Location_encoded'] = 0  # Default if column missing

        X_future = future[available_features].dropna()  # Use available_features for consistency
        future = future.loc[X_future.index]  # Realign 'future' with 'X_future' indices after dropna

        if not X_future.empty:
            predicted = xgb_model.predict(X_future)
            future['Ranking_encoded'] = predicted
            future['Ranking'] = label_enc_ranking.inverse_transform(predicted)
        else:
            future['Ranking_encoded'] = np.nan
            future['Ranking'] = "N/A"

        future['Prediction'] = True
        projection_data.append(future)

    projected_df = pd.concat(projection_data)
    df['Prediction'] = False
    combined_df = pd.concat([df, projected_df])

    # --- Filter by type ---
    type_filter = st.sidebar.radio("🎓 Filter School Type", options=["All", "PUBLIC", "PRIVATE", "Concession"])


    def filter_data(df_to_filter):  # Renamed parameter to avoid conflict
        if type_filter == "All":
            return df_to_filter
        elif type_filter == "Concession":
            if 'Concession' in df_to_filter.columns:
                return df_to_filter[df_to_filter['Concession'].astype(str).str.upper() == 'YES']
            return pd.DataFrame(columns=df_to_filter.columns)  # Return empty df if column missing
        else:
            if 'Sector' in df_to_filter.columns:
                return df_to_filter[df_to_filter['Sector'].astype(str).str.upper() == type_filter.upper()]
            return pd.DataFrame(columns=df_to_filter.columns)  # Return empty df if column missing


    filtered_df = filter_data(combined_df)

    # --- Timeline Plot ---
    st.subheader("📊 Timeline of Rankings")
    if 'Year' in filtered_df.columns:
        filtered_df['Year'] = filtered_df['Year'].astype(int)

    real_data = filtered_df[filtered_df['Prediction'] == False]
    predict_data = filtered_df[filtered_df['Prediction'] == True]

    if 'Ranking' in real_data.columns and not real_data.empty:
        timeline_real = real_data.groupby(['Year', 'Ranking']).size().unstack(fill_value=0)
    else:
        timeline_real = pd.DataFrame()  # Ensure it's an empty DataFrame

    if 'Ranking' in predict_data.columns and not predict_data.empty:
        timeline_predict = predict_data.groupby(['Year', 'Ranking']).size().unstack(fill_value=0)
    else:
        timeline_predict = pd.DataFrame()  # Ensure it's an empty DataFrame

    fig = go.Figure()
    if not timeline_real.empty:
        for col in timeline_real.columns:
            fig.add_trace(go.Scatter(x=timeline_real.index, y=timeline_real[col],
                                     mode='lines+markers', name=f"{col} (Real)"))
    if not timeline_predict.empty:
        for col in timeline_predict.columns:
            fig.add_trace(go.Scatter(x=timeline_predict.index, y=timeline_predict[col],
                                     mode='lines+markers', name=f"{col} (Predicted)", line=dict(dash='dash')))

    fig.update_layout(title=f"Ranking Evolution ({type_filter} Schools)",
                      xaxis_title='Year', yaxis_title='Number of Schools', height=500)
    st.plotly_chart(fig, use_container_width=True)

    # --- 2027 Predicted Ranking ---
    st.subheader("🔮 Predicted Rankings for 2027")
    pred_2027 = projected_df[projected_df['Year'] == 2027]
    pred_2027_filtered = filter_data(pred_2027)

    if not pred_2027_filtered.empty and 'Ranking' in pred_2027_filtered.columns:
        count_df = pred_2027_filtered['Ranking'].value_counts().reset_index()
        count_df.columns = ['Ranking', 'Count']
        fig_bar = px.bar(count_df, x='Ranking', y='Count', color='Ranking', text='Count',
                         title=f"Predicted Rankings for 2027 ({type_filter} Schools)")
        fig_bar.update_traces(textposition='outside')
        fig_bar.update_layout(height=400)
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("No predicted ranking data for 2027 based on current filters.")

    # --- Details Table ---
    st.markdown("### 🏢 School List")
    # Ensure columns exist before displaying
    # Use "Name Of Establishment" (standardized)
    display_cols = ["Name Of Establishment", 'City', 'Sector', 'Concession', 'Ranking']
    display_cols_present = [col for col in display_cols if col in pred_2027_filtered.columns]

    if not pred_2027_filtered.empty and 'Ranking' in pred_2027_filtered.columns:  # Check if not empty and 'Ranking' exists
        if "Name Of Establishment" not in pred_2027_filtered.columns:
            st.warning("The column 'Name Of Establishment' is missing in the filtered data for display.")
        st.dataframe(pred_2027_filtered[display_cols_present].sort_values('Ranking'))
    else:
        st.info("No schools to display in the list.")

    # --- Highlight C and D ---
    st.subheader("🚨 Schools Predicted in C or D")
    if 'Ranking' in pred_2027_filtered.columns:
        c_or_d = pred_2027_filtered[
            pred_2027_filtered['Ranking'].isin(['C', 'D'])].copy()  # Add .copy() to avoid SettingWithCopyWarning
    else:
        c_or_d = pd.DataFrame(columns=pred_2027_filtered.columns)  # Create an empty DataFrame with same columns

    if not c_or_d.empty and 'Ranking' in c_or_d.columns:
        c_or_d_count = c_or_d['Ranking'].value_counts().reset_index()
        c_or_d_count.columns = ['Ranking', 'Count']
        fig_cd = px.bar(c_or_d_count, x='Ranking', y='Count', color='Ranking', text='Count',
                        title="Schools Predicted in C or D")
        fig_cd.update_traces(textposition='outside')
        st.plotly_chart(fig_cd, use_container_width=True)
        # Ensure display columns are present in c_or_d
        display_cols_cd = [col for col in display_cols if col in c_or_d.columns]
        st.dataframe(c_or_d[display_cols_cd])
    else:
        st.info("No schools predicted in C or D for 2027 based on current filters.")

    # --- Location Breakdown of Public C/D ---
    st.subheader("📍 Public Schools in C or D (by Location)")
    # Check for all necessary columns before filtering
    if all(col in pred_2027.columns for col in ['Concession', 'Ranking', 'Location Name', 'Sector']):
        public_cd = pred_2027[
            (pred_2027['Sector'].astype(str).str.upper() == 'PUBLIC') &  # Changed 'Concession' to 'Sector' for public
            (pred_2027['Ranking'].isin(['C', 'D']))].copy()  # Add .copy()
        if not public_cd.empty:
            loc_count = public_cd['Location Name'].value_counts().reset_index()  # Use standardized 'Location Name'
            loc_count.columns = ['Location Name', 'Count']  # Use standardized 'Location Name'
            fig_loc = px.bar(loc_count, x='Count', y='Location Name', orientation='h',
                             # Use standardized 'Location Name'
                             title="📍 Public Schools in C or D by Location",
                             color='Count', color_continuous_scale='YlOrRd')
            fig_loc.update_layout(height=500)
            st.plotly_chart(fig_loc, use_container_width=True)
        else:
            st.info("No public schools predicted in C or D for 2027.")
    else:
        st.info("Required columns ('Sector', 'Ranking', or 'Location Name') are missing for this analysis.")

    # --- Top Feature Importance for Public C/D ---
    st.subheader("🔍 Top Predictors for C/D Ranked Public Schools")
    # Ensure public_cd is not empty and has enough unique ranking values for a meaningful model
    if 'Ranking_encoded' in public_cd.columns and not public_cd.empty:  # Removed len(unique()) > 1 check here, moved below
        # Filter for available_features here too for X_public_cd
        X_public_cd = public_cd[[f for f in features if f in public_cd.columns]].dropna()  # Use a list comprehension

        # Check if X_public_cd is not empty after dropna
        if not X_public_cd.empty:
            y_public_cd_original = public_cd.loc[X_public_cd.index, 'Ranking_encoded']  # Keep original encoded values

            # Check for at least two unique classes AFTER filtering for C/D and dropping NaNs
            if len(y_public_cd_original.unique()) < 2:
                st.info("Not enough variety in C/D rankings to calculate feature importance for public C/D schools.")
            else:
                # FIX: Re-encode y_public_cd to be binary (0 or 1)
                # Map the original encoded values (e.g., 3 and 4) to 0 and 1
                # This ensures the target is binary for temp_xgb_model

                # First, find the mapping of 'C' and 'D' to their original encoded values
                # You might have A+=0, A=1, B=2, C=3, D=4
                # Or A+=4, A=3, B=2, C=1, D=0

                # A robust way is to use a new LabelEncoder specifically for 'C' and 'D'
                binary_label_encoder = LabelEncoder()
                # Use the *original* 'Ranking' column for fitting the new encoder
                # It's safer to work with actual labels here, then map.

                # Get the actual string labels for C and D from the original label_encoder
                c_label_val = \
                [k for k, v in zip(label_enc_ranking.classes_, label_enc_ranking.transform(label_enc_ranking.classes_))
                 if k == 'C'][0]
                d_label_val = \
                [k for k, v in zip(label_enc_ranking.classes_, label_enc_ranking.transform(label_enc_ranking.classes_))
                 if k == 'D'][0]

                # Create a mapping from original encoded values (3, 4) to new binary (0, 1)
                # Assign the lower encoded value to 0 and the higher to 1.
                # Or, if you want 'C' to always be 0 and 'D' to always be 1, regardless of their original numeric order:

                # Option 1: Consistent mapping for 'C' and 'D'
                mapping_dict = {
                    label_enc_ranking.transform(['C'])[0]: 0,  # Map original encoded 'C' to 0
                    label_enc_ranking.transform(['D'])[0]: 1  # Map original encoded 'D' to 1
                }

                y_public_cd = y_public_cd_original.map(mapping_dict)

                # Now, fit the temp_xgb_model
                temp_xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss',
                                               # Use 'logloss' for binary classification
                                               n_estimators=100, random_state=42)
                temp_xgb_model.fit(X_public_cd, y_public_cd)

                booster = temp_xgb_model.get_booster()
                importance_dict = booster.get_score(importance_type='gain')

                if importance_dict:  # Check if importance_dict is not empty
                    importance_df = pd.DataFrame({
                        'Feature': list(importance_dict.keys()),
                        'Importance': list(importance_dict.values())
                    }).sort_values('Importance', ascending=True)

                    # Normalize importance scores if needed (optional, for better visualization)
                    importance_df['Importance'] /= importance_df['Importance'].sum()

                    # Map original feature names to user-friendly names
                    feature_mapping = {
                        'Concession_encoded': 'Concession',
                        'Location_encoded': 'Location',
                        'Mathematics Index': 'Mathematics Index',
                        'Natural Sciences Index': 'Natural Sciences Index',
                        'Social And Citizenship Index': 'Social And Citizenship Index',
                        'Critical Reading Index': 'Critical Reading Index',
                        'English Index': 'English Index',
                        'Total Index': 'Total Index',
                        'Year': 'Year'
                    }
                    importance_df['Feature_Display'] = importance_df['Feature'].replace(feature_mapping)

                    fig_feat = px.bar(importance_df, x='Importance', y='Feature_Display', orientation='h',
                                      title="🔍 Gain-Based Feature Importance (C/D Public Schools)",
                                      color='Importance', color_continuous_scale='Blues')
                    fig_feat.update_layout(height=500)
                    st.plotly_chart(fig_feat, use_container_width=True)
                else:
                    st.info("No feature importance data to display for public C/D schools (all importances are zero).")
        else:
            st.info("Not enough data to calculate feature importance for public C/D schools after filtering.")
    else:
        st.info("No data available or insufficient ranking variety to show feature importance for public C/D schools.")