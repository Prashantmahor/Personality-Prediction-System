import streamlit as st
import pickle
import pandas as pd
import warnings

# =========================
# Suppress sklearn warnings
# =========================
warnings.filterwarnings("ignore")

# =========================
# Load model + scaler
# =========================
@st.cache_resource
def load_artifacts():
    model = pickle.load(open("personality_model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    return model, scaler

model, scaler = load_artifacts()

# =========================
# Feature columns (same as training)
# =========================
feature_columns = [
    'social_energy',
    'alone_time_preference',
    'talkativeness',
    'deep_reflection',
    'group_comfort',
    'party_liking',
    'listening_skill',
    'empathy',
    'organization',
    'leadership',
    'risk_taking',
    'public_speaking_comfort',
    'curiosity',
    'routine_preference',
    'excitement_seeking',
    'friendliness',
    'planning',
    'spontaneity',
    'adventurousness',
    'reading_habit',
    'sports_interest',
    'online_social_usage',
    'travel_desire',
    'gadget_usage',
    'work_style_collaborative',
    'decision_speed'
]

# =========================
# Label mapping
# =========================
label_map = {
    0: "Introvert",
    1: "Ambivert",
    2: "Extrovert"
}

# =========================
# UI
# =========================
st.set_page_config(page_title="Personality Predictor", page_icon="🧠", layout="wide")

st.title("🧠 Personality Type Predictor")
st.caption("Adjust sliders to describe personality traits")

user_inputs = {}

cols = st.columns(3)

for i, feature in enumerate(feature_columns):
    with cols[i % 3]:
        user_inputs[feature] = st.slider(
            feature.replace("_", " ").title(),
            min_value=0,
            max_value=10,
            value=5
        )

# =========================
# Prediction
# =========================
if st.button("Predict Personality", use_container_width=True):

    input_df = pd.DataFrame([user_inputs])[feature_columns]

    scaled_input = scaler.transform(input_df)

    prediction = model.predict(scaled_input)[0]

    # Convert numeric → text
    if prediction in label_map:
        prediction_text = label_map[prediction]
    else:
        prediction_text = str(prediction)

    st.success(f"Predicted Personality Type: {prediction_text}")

st.divider()
st.info("Run this app using: python -m streamlit run app.py")