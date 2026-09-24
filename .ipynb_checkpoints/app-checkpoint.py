import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# ==========================================
# Page Configuration
# ==========================================

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="centered"
)


# ==========================================
# Load Model
# ==========================================

MODEL_PATH = Path(__file__).parent / "model_SVR.pkl"

try:
    model = joblib.load(MODEL_PATH)

except FileNotFoundError:
    st.error(
        f"Model file not found:\n{MODEL_PATH}"
    )
    st.stop()


# ==========================================
# Title
# ==========================================

st.title("🏠 House Price Prediction")

st.write(
    "Enter the house details below to predict the house price."
)


# ==========================================
# House Information
# ==========================================

st.subheader("House Information")


MSSubClass = st.number_input(
    "MSSubClass",
    min_value=0,
    value=60,
    step=1
)


LotArea = st.number_input(
    "Lot Area",
    min_value=0,
    value=8000,
    step=100
)


OverallCond = st.number_input(
    "Overall Condition",
    min_value=1,
    max_value=10,
    value=5,
    step=1
)


YearBuilt = st.number_input(
    "Year Built",
    min_value=1800,
    max_value=2026,
    value=2000,
    step=1
)


YearRemodAdd = st.number_input(
    "Year Remodeled",
    min_value=1800,
    max_value=2026,
    value=2005,
    step=1
)


BsmtFinSF2 = st.number_input(
    "Basement Finished Area 2",
    min_value=0,
    value=0,
    step=10
)


TotalBsmtSF = st.number_input(
    "Total Basement Area",
    min_value=0,
    value=1000,
    step=10
)


# ==========================================
# Categorical Features
# ==========================================

st.subheader("Categorical Information")


MSZoning = st.selectbox(
    "MSZoning",
    [
        "C (all)",
        "FV",
        "RH",
        "RL",
        "RM"
    ]
)


LotConfig = st.selectbox(
    "Lot Configuration",
    [
        "Corner",
        "CulDSac",
        "FR2",
        "FR3",
        "Inside"
    ]
)


BldgType = st.selectbox(
    "Building Type",
    [
        "1Fam",
        "2fmCon",
        "Duplex",
        "Twnhs",
        "TwnhsE"
    ]
)


Exterior1st = st.selectbox(
    "Exterior Material",
    [
        "AsbShng",
        "AsphShn",
        "BrkComm",
        "BrkFace",
        "CemntBd",
        "HdBoard",
        "MetalSd",
        "Plywood",
        "Stone",
        "Stucco",
        "VinylSd",
        "Wd Sdng",
        "WdShing"
    ]
)


# ==========================================
# Prediction
# ==========================================

if st.button("🔮 Predict House Price"):

    # --------------------------------------
    # Create input dataframe
    # --------------------------------------

    input_data = pd.DataFrame({
        "MSSubClass": [MSSubClass],
        "LotArea": [LotArea],
        "OverallCond": [OverallCond],
        "YearBuilt": [YearBuilt],
        "YearRemodAdd": [YearRemodAdd],
        "BsmtFinSF2": [BsmtFinSF2],
        "TotalBsmtSF": [TotalBsmtSF],
        "MSZoning": [MSZoning],
        "LotConfig": [LotConfig],
        "BldgType": [BldgType],
        "Exterior1st": [Exterior1st]
    })


    # --------------------------------------
    # Display input
    # --------------------------------------

    st.subheader("House Information")

    st.dataframe(
        input_data,
        use_container_width=True
    )


    # --------------------------------------
    # Prediction
    # --------------------------------------

    try:

        prediction = model.predict(
            input_data
        )

        predicted_price = prediction[0]


        # ----------------------------------
        # Display result
        # ----------------------------------

        st.subheader("Prediction")

        st.success(
            f"🏠 Predicted House Price: "
            f"${predicted_price:,.2f}"
        )


    except Exception as e:

        st.error(
            "Prediction failed."
        )

        st.exception(e)


# ==========================================
# Model Information
# ==========================================

with st.expander("Model Information"):

    st.write("Model: SVR")

    st.write("Features used by the model:")

    st.write([
        "MSSubClass",
        "LotArea",
        "OverallCond",
        "YearBuilt",
        "YearRemodAdd",
        "BsmtFinSF2",
        "TotalBsmtSF",
        "MSZoning",
        "LotConfig",
        "BldgType",
        "Exterior1st"
    ])

    st.write(
        "Categorical features are automatically "
        "One-Hot Encoded by the saved pipeline."
    )