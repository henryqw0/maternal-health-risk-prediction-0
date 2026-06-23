import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import shap
from fpdf import FPDF
import datetime

# Set page configuration for a professional look
st.set_page_config(page_title="Clinical Triage AI", page_icon="🏥", layout="centered")

# ==============================================================================
# 📥 1. LOAD THE TRAINED PIPELINE
# ==============================================================================
@st.cache_resource
def load_pipeline():
    # Make sure you have exported your pipeline as 'elite_medical_rf_model.pkl' first!
    return joblib.load('elite_medical_rf_model.joblib')

try:
    pipeline = load_pipeline()
    scaler = pipeline.named_steps['scaler']
    rf_model = pipeline.named_steps['classifier']
    feature_names = list(pipeline.feature_names_in_)
except FileNotFoundError:
    st.error("⚠️ Error:'elite_medical_rf_model.joblib' not found. Please run your training script to export the model file first!")
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
#  GENERATING THE REAL EXPLAINABLE AI AUDIT GRAPH
# ==============================================================================

st.subheader("🛡️ Clinical Transparency Audit Trail")
st.markdown("This chart displays the **exact** mathematical impact each vital sign had on pushing the model toward this diagnosis.")

# 1. Initialize the real SHAP engine on your model
explainer = shap.TreeExplainer(rf_model)

# 2. Calculate the real SHAP values for this specific slider patient
# [0] grabs the current patient row, [:, 2] isolates the High Risk category
shap_values = explainer.shap_values(scaled_input_df)

if isinstance(shap_values, list):
    # Older SHAP version handler (list of arrays)
    real_contributions = shap_values[predicted_class][0]
else:
    # Newer SHAP version handler (3D array: rows, features, classes)
    real_contributions = shap_values[0, :, predicted_class]

# Convert array to a standard list for the plotting code below
tree_contributions = real_contributions.tolist()

# Create the visual plot
fig, ax = plt.subplots(figsize=(8, 3))

# 1. Ensure tree_contributions matches the exact length of your features
# (Flattens any nested arrays that SHAP sometimes outputs)
clean_contributions = np.array(tree_contributions).flatten()

# 2. Assign colors: Red for risk escalators (>0), Green for safe pullers (<=0)
colors = ['#ff4b4b' if x > 0 else '#23c14c' for x in clean_contributions]

# 3. Draw the horizontal bars safely
y_pos = np.arange(len(feature_names))
ax.barh(y_pos, clean_contributions, color=colors, edgecolor='none', height=0.6)

# 4. Attach clear y-axis labels
ax.set_yticks(y_pos)
ax.set_yticklabels(feature_names, fontsize=10, fontweight='bold')
ax.invert_yaxis()  # Top-down feature layout

# Style clean-up
ax.axvline(x=0, color='#31333F', linestyle='-', linewidth=1, alpha=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.set_xlabel("<- Pulls Toward Safe  |  Escalates Risk Alert ->", fontsize=9, color='#31333F')

# Render the plot inside the web window frame cleanly
st.pyplot(fig)

# ==============================================================================
# GENERATING THE LIVE SHAP WATERFALL ESCALATION GRAPH
# ==============================================================================
# import shap
# import matplotlib.pyplot as plt

# st.subheader("🛡️ Clinical Transparency Diagnostic Waterfall")
# st.markdown("This cascade shows exactly how the patient's vitals calculated a step-by-step path from the average baseline to their final diagnostic result.")

# # 1. Initialize the tree explainer explicitly enabling native SHAP object format
# explainer = shap.TreeExplainer(rf_model)

# # 2. Compute the explanation object for your current slider input
# # (Passing check_additivity=False keeps server responses fast and smooth)
# explanation = explainer(scaled_input_df, check_additivity=False)

# # 3. Construct the explanation wrapper for the winning class index
# # This builds the structural object containing baseline, scores, and raw values
# class_explanation = shap.Explanation(
#     values=explanation.values[0, :, predicted_class],
#     base_values=explanation.base_values[0, predicted_class],
#     data=raw_input_df.iloc[0].values,  # Maps raw unscaled values onto the text labels!
#     feature_names=feature_names
# )

# # 4. Generate the visual Matplotlib canvas
# fig, ax = plt.subplots(figsize=(8, 4))

# # Pass the custom explanation object directly into SHAP's native waterfall plotter
# shap.plots.waterfall(class_explanation, max_display=5, show=False)

# # Clean up styling details for a premium look
# plt.title(f"Diagnostic Contribution Escalation Map ({status_title})", fontsize=11, fontweight='bold', pad=15)
# plt.tight_layout()

# # Render the plot frame inside the Streamlit web app layout window cleanly
# st.pyplot(fig)

# ==============================================================================
#  CLINICAL NOTES & PDF REPORT DOWNLOAD ENGINE (FIXED)
# ==============================================================================

st.write("---")
st.subheader("📝 Clinical Consultation Notes")



# 2. Define a function to generate a cleanly formatted PDF document structure in memory
def generate_pdf(patient_data, probabilities, diagnosis, notes):
    pdf = FPDF()
    pdf.add_page()
    
    # Header: Professional Clinic Layout
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(49, 51, 63) # Charcoal blue
    pdf.cell(0, 10, "MATERNAL HEALTH RISK PREDICTOR REPORT", ln=True, align="C")
    
    pdf.set_font("Helvetica", "I", 10)
    pdf.set_text_color(100, 100, 100)
    current_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pdf.cell(0, 8, f"Generated on: {current_date} | System Status: Secure", ln=True, align="C")
    pdf.ln(5)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y()) # Horizontal divider line
    pdf.ln(5)
    
    # Section 1: Patient Vital Signs Summary Grid
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(49, 51, 63)
    pdf.cell(0, 8, "1. Patient Vital Signs Summary", ln=True)
    pdf.ln(2)
    
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(0, 0, 0)
    
    # ✅ FIXED: Expanded width to 75 to avoid text smashing, and removed val[0] slicing!
    for key, val in patient_data.items():
        pdf.cell(75, 7, f"  - {key}:", border=0)
        pdf.cell(0, 7, f"{val}", border=0, ln=True) # Changed from val[0] to val
    pdf.ln(5)
    
    # Section 2: AI Diagnostic Verdict Matrix
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(49, 51, 63)
    pdf.cell(0, 8, "2. AI Triage Diagnostics Verdict", ln=True)
    pdf.ln(2)
    
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(197, 15, 15) if "HIGH" in diagnosis else pdf.set_text_color(15, 117, 43)
    pdf.cell(0, 8, f"ASSIGNED TIER -> {diagnosis}", ln=True)
    
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(100, 100, 100)
    p_low, p_mid, p_high = probabilities
    pdf.cell(0, 6, f"Confidence breakdown: Low: {p_low*100:.1f}% | Mid: {p_mid*100:.1f}% | High: {p_high*100:.1f}%", ln=True)
    pdf.ln(5)
    
    # Section 3: Physician Review Text Box
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(49, 51, 63)
    pdf.cell(0, 8, "3. Attending Physician Consultation Notes", ln=True)
    pdf.ln(2)
    
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(0, 0, 0)
    if notes.strip() == "":
        notes = "No consultation notes entered by the attending physician."
    
    # multi_cell automatically wraps long paragraphs safely so they don't clip off the page edge
    pdf.multi_cell(0, 6, notes, border=1)
    pdf.ln(15)
    
    # Footer: Signature Validation Lines
    pdf.line(10, pdf.get_y(), 80, pdf.get_y())
    pdf.ln(2)
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 5, "Attending Clinician Signature Validation Stamp", ln=True)
    
    # Convert file structure matrix directly into an downloadable byte sequence output
    return pdf.output()
doctor_notes = st.text_area(
    "Enter diagnostic notes, prescription steps, or patient tracking details below:",
    placeholder="e.g., Patient displays elevated blood glucose. Advised nutritional adjustments and scheduled a 2-week follow-up."
)

clean_pdf_title = status_title.replace("🟢 ", "").replace("🟡 ", "").replace("🚨 ", "")

# 2. FORCE data types to be pure primitive integers/floats right at extraction
pdf_age_val = int(age)
pdf_sys_val = int(systolic_bp)
pdf_dia_val = int(diastolic_bp)
pdf_bs_val = float(bs)
pdf_temp_val = float(body_temp)
pdf_hr_val = int(heart_rate)

# 3. Create a clean dictionary from our locked-in primitive values
pdf_summary_data = {
    'Patient Age': f"{pdf_age_val} Years",
    'Systolic BP': f"{pdf_sys_val} mmHg",
    'Diastolic BP': f"{pdf_dia_val} mmHg",
    'Blood Sugar (BS)': f"{pdf_bs_val} mmol/L",
    'Body Temperature': f"{pdf_temp_val} °F",
    'Heart Rate': f"{pdf_hr_val} BPM"
}

# 4. Generate the PDF payload using the correct inputs
pdf_data = generate_pdf(
    patient_data=pdf_summary_data, 
    probabilities=pred_probabilities, 
    diagnosis=clean_pdf_title, 
    notes=doctor_notes
)

# 5. FIXED: Fully closed download button syntax parameters
st.download_button(
    label="📥 Download Official Clinical PDF Report",
    data=bytes(pdf_data),
    file_name=f"maternal_health_report_{datetime.date.today()}.pdf",
    mime="application/pdf",
    use_container_width=True
)

