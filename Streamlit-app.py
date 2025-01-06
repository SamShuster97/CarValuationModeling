import shap
import pandas as pd
import numpy as np
import streamlit as st
from matplotlib import pyplot as plt
from pyarrow import parquet as pq
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Path of the trained model and data
MODEL_PATH = "../model/xgboost_model.cbm" 
DATA_PATH = "../model/valuation_model_data.parquet"

st.set_page_config(page_title="Car Valuation Project")

@st.cache_resource
def load_data():
    data = pd.read_parquet(DATA_PATH)
    return data

def load_x_y(file_path):
    data = joblib.load(file_path)  # Load the pickle file

    if isinstance(data, pd.DataFrame):  # If it's a DataFrame, reset index
        data.reset_index(drop=True, inplace=True)
    return data

def load_model():
    model = xgb.XGBRegressor()
    model.load_model(MODEL_PATH)
    return model

def calculate_shap(model, X_train, X_test):
    # Calculate SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values_xgb_train = explainer.shap_values(X_train)
    shap_values_xgb_test = explainer.shap_values(X_test)
    return explainer, shap_values_xgb_train, shap_values_xgb_test

def plot_shap_values(model, explainer, shap_values_xgb_train, shap_values_xgb_test, vin, X_test, X_train):
    # Visualize SHAP values for a specific vehicle
    vehicle_index = X_test[X_test['vin'] == vin].index[0]
    fig, ax_2 = plt.subplots(figsize=(6,6), dpi=200)
    shap.decision_plot(explainer.expected_value, shap_values_xgb_test[vehicle_index], X_test[X_test['vin'] == vin], link="logit")
    st.pyplot(fig)
    plt.close()

def display_shap_summary(shap_values_xgb_train, X_train):
    # Create the plot summarizing the SHAP values
    shap.summary_plot(shap_values_xgb_train, X_train, plot_type="bar", plot_size=(12,12))
    summary_fig, _ = plt.gcf(), plt.gca()
    st.pyplot(summary_fig)
    plt.close()

def display_shap_waterfall_plot(explainer, expected_value, shap_values, feature_names, max_display=20):
    # Create SHAP waterfall drawing
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    expected_value = explainer.expected_value[0] if isinstance(explainer.expected_value, list) else explainer.expected_value
    shap.plots._waterfall.waterfall_legacy(expected_value, shap_values, feature_names=feature_names, max_display=max_display, show=False)
    st.pyplot(fig)
    plt.close()

def summary(model, data, X_train, X_test):
    # Calculate SHAP values
    explainer, shap_values_xgb_train, shap_values_xgb_test = calculate_shap(model, X_train, X_test)

    # Summarize and visualize SHAP values
    display_shap_summary(shap_values_xgb_train, X_train)

def plot_shap(model, data, vin, X_train, X_test):
    # Calculate SHAP values
    explainer, shap_values_xgb_train, shap_values_xgb_test = calculate_shap(model, X_train, X_test)
    
    # Visualize SHAP values
    plot_shap_values(model, explainer, shap_values_xgb_train, shap_values_xgb_test, vin, X_test, X_train)

    # Waterfall
    vehicle_index = X_test[X_test['vin'] == vin].index[0]
    display_shap_waterfall_plot(explainer, explainer.expected_value, shap_values_xgb_test[vehicle_index], feature_names=X_test.columns, max_display=20)

st.title("Used Vehicle Valuation")

def main():
    model = load_model()
    data = load_data()

    # Load data using corrected load_x_y function
    X_train = load_x_y("../Car_valuation_model/model/X_train.pkl")
    X_test = load_x_y("../Car_valuation_model/model/X_test.pkl")
    y_train = load_x_y("../Car_valuation_model/model/y_train.pkl")
    y_test = load_x_y("../Car_valuation_model/model/y_test.pkl")

    max_tenure = data['tenure'].max()
    max_monthly_charges = data['MonthlyCharges'].max()
    max_total_charges = data['TotalCharges'].max()

    # Radio buttons for options
    election = st.radio("Make Your Choice:", ("Feature Importance", "User-based SHAP", "Calculate the approximate price:"))
    available_vins = X_test['vin'].tolist()
    
    # If User-based SHAP option is selected
    if election == "User-based SHAP":
        # vehicle ID text input
        vin = st.selectbox("Choose the vehicle", available_vins)
        vehicle_index = X_test[X_test['vin'] == vin].index[0]
        st.write(f'vehicle {vin}: Actual value for the vehicle price : {y_test.iloc[vehicle_index]}')
        y_pred = model.predict(X_test)
        st.write(f"vehicle {vin}: XGBoost Model's prediction for the vehicle price : {y_pred[vehicle_index]}")
        plot_shap(model, data, vin, X_train=X_train, X_test=X_test)
    
    # If Feature Importance is selected
    elif election == "Feature Importance":
        summary(model, data, X_train=X_train, X_test=X_test)

    # If Calculate approximate price option is selected
    elif election == "Calculate the approximate price":
        # Retrieving data from the user
        vin = "jtezu11f88k007763"
        brand = st.selectbox("Brand:", ("acura", "audi", "bmw", "buick", "cadillac", "chevrolet", "chrysler", "dodge", "ford", "gmc", "harley-davidson", "heartland", "honda", "hyundai", "infiniti", "jeep", "kia", "landrover", "lexus", "lincoln", "maserati", "mazda", "mercedes-benz", "nissan", "peterbilt", "ram", "toyota"))
        model = st.number_input("model:" ("300", "1500", "2500", "a5", "acadia", "altima", "armada", "cab", "boxster", "camaro", "caravan", "cargo", "challenger", "charger", "chassis", "cherokee", "colorado", "compass", "corvette", "country", "cruze", "cutaway", "cx-3", "dart", "discovery", "door", "doors", "dr", "drw", "durango", "e-class", "ecosport", "edge", "el", "elantra", "enclave", "encore", "equinox", "escape", "esv", "expedition", "explorer", "f-150", "fiesta", "flex", "focus", "forte", "frontier", "fusion", "glc", "gle", "gx", "hybrid", "impala", "journey", "juke", "kicks", "ld", "limited", "m3", "malibu", "max", "maxima", "mdx", "mpv", "murano", "mustang", "nautilus", "note", "pacifica", "passenger", "pathfinder", "pioneer", "pk", "q5", "q50", "q70", "ranger", "rogue", "se", "sedan", "s-class", "sentra", "sonic", "soul", "spark", "srw", "srx", "sport", "sportage", "sonata", "sorento", "suburban", "tacoma", "tahoe", "titan", "trail", "transit", "transverse", "trax", "truck", "van", "vans", "versa", "volt", "wagon", "x3", "x5", "xt5", "xterra"))
        year = st.selectbox("Year:", ("1973", "1984", "1993", "1994", "1995", "1996", "1997", "1998", "1999", "2000", "2001", "2002", "2003", "2004", "2005", "2006", "2007", "2008", "2009", "2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020"))
        title_status = st.selectbox("Title Status:", ("clean vehicle", "salvage insurance"))
        mileage = st.number_input("Tenure:", min_value=0, max_value=999999)
        color = st.selectbox("Color:", ("beige", "black", "blue", "brown", "gold", "gray", "green", "orange", "purple", "red", "silver", "white", "yellow"))
        state = st.selectbox("State:", ("alabama", "arizona", "arkansas", "california", "colorado", "connecticut", "delaware", "florida", "georgia", "idaho", "illinois", "indiana", "iowa", "kansas", "kentucky", "louisiana", "maine", "maryland", "massachusetts", "michigan", "minnesota", "mississippi", "missouri", "montana", "nebraska", "nevada", "new hampshire", "new jersey", "new mexico", "new york", "north carolina", "north dakota", "ohio", "oklahoma", "oregon", "pennsylvania", "rhode island", "south carolina", "south dakota", "tennessee", "texas", "utah", "vermont", "virginia", "washington", "west virginia", "wisconsin", "wyoming"))
        
        # Confirmation button
        confirmation_button = st.button("Confirm")

        # When the confirmation button is clicked
        if confirmation_button:
            # Convert user-entered data into a data frame
            new_vehicle_data = pd.DataFrame({
                "vin": [vin],
                "brand": [brand],
                "model": [model],
                "year": [year],
                "title_status": [title_status],
                "mileage": [mileage],
                "color": [color],
                "state": [state],
            })

            # Predict approximate price using the model
            approximate_price = model.predict(new_vehicle_data)

            big_text = f"<h1>Approximate Price: {approximate_price}</h1>"
            st.markdown(big_text, unsafe_allow_html=True)
            st.write(new_vehicle_data.to_dict())

if __name__ == "__main__":
    main()
