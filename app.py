import streamlit as st
import pickle
import numpy as np

# 1. Jo model humne abhi train kiya hai, use load karna
with open('fraud_model.pkl', 'rb') as f:
    model = pickle.load(f)

# 2. Webpage ka Title aur Heading banana
st.set_page_config(page_title="Fraud Detection Analytics", page_icon="🛡️")
st.title("🛡️ Financial Fraud Detection System")
st.write("Welcome, Akanksha! Enter transaction details below to check for fraud risk.")

st.markdown("---")

# 3. User se input lene ke liye boxes banana
st.subheader("💵 Transaction Details")
amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=100.0, step=1.0)

st.subheader("📊 Anonymized PCA Features (V1 & V2)")
st.write("Real-world banking data me features secure hote hain, testing ke liye values enter karein:")
v1 = st.number_input("Feature V1", value=0.0)
v2 = st.number_input("Feature V2", value=0.0)

# 4. Model me 29 features chahiye, baki 26 ko hum default 0.0 set kar dete hain
dummy_features = [0.0] * 26
# Saare inputs ko ek sath jod kar array banana
input_data = np.array([[v1, v2] + dummy_features + [amount]])

st.markdown("---")

# 5. Predict Button dabane par kya hoga
if st.button("🔍 Analyze Transaction", type="primary"):
    # Machine se prediction lena
    prediction = model.predict(input_data)
    prediction_prob = model.predict_proba(input_data)[0][1] # Fraud hone ki percentage
    
    if prediction[0] == 1:
        st.error(f"🚨 High Risk! Potential Fraud Detected (Risk Score: {prediction_prob*100:.2f}%)")
    else:
        st.success(f"✅ Safe Transaction! (Fraud Probability: {prediction_prob*100:.2f}%)")