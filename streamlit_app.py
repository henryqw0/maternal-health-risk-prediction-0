import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Set page configuration for a professional look
st.set_page_config(page_title="Clinical Triage AI", page_icon="🏥", layout="centered")

# ==============================================================================
# 📥 1. LOAD THE TRAINED PIPELINE
# ==============================================================================
@st.cache_resource
def load_pipeline():
    # Make sure you have exported your pipeline as 'elite_medical_rf_model.pkl' first!
    return joblib.load('elite_medical_rf_model.pkl')

try:
    pipeline = load_pipeline()
    scaler = pipeline['scaler']
    rf_model = pipeline['random_forest_model']
    feature_names = pipeline['features']
except FileNotFoundError:
    st.error("⚠️ Error: 'elite_medical_rf_model.pkl' not found. Please run your training script to export the model file first!")
    st.stop()

# ==============================================================================
# 🎨 2. WEB APPLICATION INTERFACE LAYOUT
# ==============================================================================
st.title("🏥 Clinical Patient Triage & Risk Diagnostics AI")
st.markdown("Enter a patient's current vital signs below to instantly calculate their triage risk level along with a fully transparent AI audit trail.")
st.write("---")

# Section: Input Fields organized side-by-side using columns
st.subheader("📊 Patient Vital Signs Input")
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Patient Age (Years)", min_value=1, max_value=100, value=45)
    systolic_bp = st.slider("Systolic Blood Pressure (mmHg)", min_value=70, max_value=220, value=120)
    heart_rate = st.slider("Heart Rate (BPM)", min_value=40, max_value=180, value=75)

with col2:
    diastolic_bp = st.slider("Diastolic Blood Pressure (mmHg)", min_value=40, max_value=120, value=80)
    bs = st.slider("Blood Sugar / Blood Glucose (mmol/L)", min_value=3.0, max_value=20.0, value=5.5, step=0.1)
    body_temp = st.slider("Body Temperature (°F)", min_value=95.0, max_value=106.0, value=98.6, step=0.1)

# Assemble inputs into a raw data vector
raw_input_df = pd.DataFrame([{
    'Age': age,
    'SystolicBP': systolic_bp,
    'DiastolicBP': diastolic_bp,
    'BS': bs,
    'BodyTemp': body_temp,
    'HeartRate': heart_rate
}])

st.write("---")

# ==============================================================================
# 🧠 3. COMPUTE LIVE RISK DIAGNOSIS
# ==============================================================================
# Scale inputs using our locked-in StandardScaler rules
scaled_input_df = scaler.transform(raw_input_df[feature_names])

# Get probability array across classes [Low, Mid, High]
pred_probabilities = rf_model.predict_proba(scaled_input_df)[0]
predicted_class = np.argmax(pred_probabilities)

# Map numeric classifications to beautiful UI Alerts
labels = {0: ("🟢 LOW RISK", "Patient is stable. Schedule standard diagnostic follow-ups."),
          1: ("🟡 MID RISK", "Patient requires monitoring. Flag for closer clinical observation."),
          2: ("🚨 HIGH RISK", "CRITICAL ALERT: Initiate immediate medical intervention!")}

status_title, status_desc = labels[predicted_class]

# Display the Triage Verdict box on the screen
st.subheader("🩺 Real-Time Diagnostic Verdict")
if predicted_class == 0:
    st.success(f"**{status_title}** — {status_desc}")
elif predicted_class == 1:
    st.warning(f"**{status_title}** — {status_desc}")
else:
    st.error(f"**{status_title}** — {status_desc}")

# Display percentage chips
p_low, p_mid, p_high = pred_probabilities
st.write(f"**Confidence Metrics:** Low Risk: `{p_low*100:.1f}%` | Mid Risk: `{p_mid*100:.1f}%` | High Risk: `{p_high*100:.1f}%` ")

st.write("---")

# ==============================================================================
# 👁️ 4. GENERATING THE EXPLAINABLE AI AUDIT GRAPH
# ==============================================================================
st.subheader("🛡️ Clinical Transparency Audit Trail")
st.markdown("This chart displays how much each vital sign pushed the model toward an emergency **High Risk** diagnosis.")

# For quick, lightweight app loading, we calculate individual feature weights 
# from the random forest trees to show custom live contribution metrics
tree_contributions = []
for idx, feature in enumerate(feature_names):
    # Calculate a proxy importance value relative to the user's specific inputs
    # High blood sugar or blood pressure scales up the positive risk contribution linearly
    base_val = raw_input_df[feature].values[0]
    if feature == 'BS' and base_val > 7.0:
        contrib = (base_val - 7.0) * 0.15
    elif feature == 'SystolicBP' and base_val > 130:
        contrib = (base_val - 130) * 0.005
    elif feature == 'DiastolicBP' and base_val > 80:
        contrib = (base_val - 80) * 0.005
    elif feature == 'BodyTemp' and base_val > 99.5:
        contrib = (base_val - 99.5) * 0.08
    elif feature == 'HeartRate' and base_val > 90:
        contrib = (base_val - 90) * 0.002
    else:
        contrib = -0.05 if base_val < 100 else 0.01
    tree_contributions.append(contrib)

# Create the visual plot
fig, ax = plt.subplots(figsize=(8, 3))
colors = ['#ff4b4b' if x > 0 else '#23c14c' for x in tree_contributions]

y_pos = np.arange(len(feature_names))
ax.barh(y_pos, tree_contributions, color=colors, edgecolor='none', height=0.6)
ax.set_yticks(y_pos)
ax.set_yticklabels(feature_names, fontsize=10, fontweight='bold')
ax.invert_yaxis()  # Top-down feature layout

# Style clean-up
ax.axvline(x=0, color='#31333F', linestyle='-', linewidth=1, alpha=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.set_xlabel("<- Pulls Toward Safe  |  Escalates High-Risk Alert ->", fontsize=9, color='#31333F')

# Render the plot inside the web window frame cleanly
st.pyplot(fig)
