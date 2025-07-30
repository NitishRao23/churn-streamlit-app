cat > app/app.py << 'EOF'
import streamlit as st
import pickle
import numpy as np

st.title("📉 Telecom Customer Churn Prediction")

# Load the model
with open("mlmodel/churn_model.pkl", "rb") as f:
    model = pickle.load(f)


# Sample input fields
gender = st.selectbox("Gender", ["Male", "Female"])
senior = st.selectbox("Senior Citizen", ["Yes", "No"])
tenure = st.slider("Tenure (Months)", 0, 72)
monthly_charges = st.number_input("Monthly Charges")
total_charges = st.number_input("Total Charges")

# Convert input into model format
if st.button("Predict Churn"):
    input_data = np.array([
        1 if gender == "Male" else 0,
        1 if senior == "Yes" else 0,
        tenure,
        monthly_charges,
        total_charges
    ]).reshape(1, -1)

    prediction = model.predict(input_data)
    st.success("❌ Not Churning" if prediction[0] == 0 else "⚠️ Likely to Churn")
EOF
