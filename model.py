# =====================================================================
# 1. Sabse Pehle: Zaroori Softwares (Libraries) Ko Bulana
# =====================================================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import pickle

# =====================================================================
# 2. Jo csv file download ki thi, use padhna aur clean karna
# =====================================================================
print("📦 Loading dataset...")
df = pd.read_csv("creditcard.csv")

print("🧹 Preprocessing and Scaling...")
# Amount column ko scale karna (bade numbers ko normal range me lana)
scaler = StandardScaler()
df['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))

# Purane bina scale kiye hue columns ko hata dena
df.drop(['Time', 'Amount'], axis=1, inplace=True)

# Features (X) aur Target (y) ko alag karna
X = df.drop('Class', axis=1) # Class column ko chhod kar baaki sab X hain
y = df['Class']              # Class column (0 = Sahi, 1 = Fraud) y hai

# =====================================================================
# 3. Data Ko Baantna Aur Balance (SMOTE) Karna
# =====================================================================
# Data ko 80% Train (seekhne) aur 20% Test me divide karna
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# SMOTE lagakar fraud data ko badhana (Data Balancing)
print("⚖️ Balancing data using SMOTE... Isme 1 minute lag sakta hai...")
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# =====================================================================
# 4. Machine Ko Sikhana Aur Save Karna
# =====================================================================
# XGBoost Classifier ko initialize karna
print("🚀 Machine learning model train ho raha hai...")
model = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)

# Machine ko sikhana (Training)
model.fit(X_train_res, y_train_res)

# Seekhe hue brain ko 'fraud_model.pkl' naam ki file me save karna
with open('fraud_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("💾 Model successfully save ho gaya hai 'fraud_model.pkl' ke naam se!")