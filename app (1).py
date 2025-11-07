import json, joblib, pandas as pd, numpy as np
import streamlit as st

st.set_page_config(page_title="Travel Insurance Predictor", page_icon="✈️", layout="wide")

@st.cache_resource
def load_models():
    dt = joblib.load("models/dt_model.joblib")
    xgb = joblib.load("models/xgb_model.joblib")
    with open("models/feature_names.json") as f:
        feat = json.load(f)
    return dt, xgb, feat

dt_model, xgb_model, feat = load_models()
CAT_COLS, NUM_COLS, ALL_COLS = feat["cat_cols"], feat["num_cols"], feat["all_cols"]

st.title("✈️ Travel Insurance Purchase Prediction")

# Sidebar controls
st.sidebar.header("Model & Settings")
model_name = st.sidebar.selectbox("Select Model", ["Decision Tree (Balanced)", "XGBoost"])
threshold = st.sidebar.slider("Decision Threshold", 0.10, 0.90, 0.50, 0.01)
model = dt_model if "Decision Tree" in model_name else xgb_model

tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction (CSV)"])

with tab1:
    st.subheader("Customer Inputs")
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=80, value=30)
        family = st.number_input("Family Members", min_value=0, max_value=10, value=2)
        chronic = st.selectbox("Chronic Diseases", [0, 1])
    with col2:
        income = st.number_input("Annual Income", min_value=1000, max_value=500000, value=120000, step=1000)
        emp = st.selectbox("Employment Type", ["Government Sector", "Private Sector/Self Employed"])
        grad = st.selectbox("GraduateOrNot", ["No", "Yes"])
    with col3:
        ff = st.selectbox("FrequentFlyer", ["No", "Yes"])
        abroad = st.selectbox("EverTravelledAbroad", ["No", "Yes"])

    input_df = pd.DataFrame([{
        "Age": age,
        "AnnualIncome": income,
        "FamilyMembers": family,
        "ChronicDiseases": chronic,
        "Employment Type": emp,
        "GraduateOrNot": grad,
        "FrequentFlyer": ff,
        "EverTravelledAbroad": abroad
    }])[ALL_COLS]

    if st.button("Predict"):
        proba = model.predict_proba(input_df)[0,1]
        pred = int(proba >= threshold)
        st.metric("Purchase Probability", f"{proba:.2%}")
        st.write(f"**Decision (threshold {threshold:.2f})**: {'Buy (1)' if pred==1 else 'Not Buy (0)'}")

        # Feature importance for tree-based model
        try:
            ohe = model.named_steps["pre"].named_transformers_["cat"]
            ohe_names = list(ohe.get_feature_names_out(CAT_COLS))
            final_cols = ohe_names + NUM_COLS
            importances = model.named_steps["clf"].feature_importances_
            top = pd.Series(importances, index=final_cols).sort_values(ascending=False).head(10)
            st.subheader("Top Feature Importance")
            st.bar_chart(top)
        except Exception:
            st.info("Feature importance not available for this configuration.")

with tab2:
    st.subheader("Upload CSV for Bulk Scoring")
    st.caption(f"Columns required: {', '.join(ALL_COLS)}")
    file = st.file_uploader("Upload CSV", type=["csv"])
    if file:
        df = pd.read_csv(file)
        missing = [c for c in ALL_COLS if c not in df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            proba = model.predict_proba(df[ALL_COLS])[:,1]
            pred = (proba >= threshold).astype(int)
            out = df.copy()
            out["prob_buy"] = proba
            out["pred_buy"] = pred
            st.success(f"Scored {len(out)} rows.")
            st.dataframe(out.head(20))
            st.download_button(
                "Download Results",
                out.to_csv(index=False).encode("utf-8"),
                file_name="predictions.csv",
                mime="text/csv"
            )

st.markdown("---")
st.caption("Models: Decision Tree (class_weight='balanced') & XGBoost. Adjust the decision threshold to fit business goals.")
