import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sqlalchemy import create_engine
# PostgreSQL connection details
user = 'postgres'
password = 'Nitish@23'
host = '127.0.0.1'
port = '5432'
database = 'churn_project'

# Create engine
engine = create_engine('postgresql+psycopg2://postgres:Nitish%4023@127.0.0.1:5432/churn_project')

print("🚀 Starting training...")

df = pd.read_sql('SELECT * FROM public.telco_churn_data', engine)

# Check what values are in the target column 'Churn'
print("Original 'Churn' values:", df['Churn'].unique())

# Clean the 'Churn' column: remove spaces and convert to lowercase
df['Churn'] = df['Churn'].astype(str).str.strip().str.lower()

print("Cleaned 'Churn' values:", df['Churn'].unique())
print("Counts of each 'Churn' value:")
print(df['Churn'].value_counts(dropna=False))

# Map target to 0 and 1 (yes=1, no=0)
y = df['Churn'].astype(int)

print(f"Number of valid target samples: {y.notna().sum()}")

# Keep only rows where target is valid
mask = y.notna()
X = df.loc[mask].drop(columns=['Churn', 'customerID'])
y = y[mask]

# Convert categorical features to numeric
non_numeric = X.select_dtypes(include=['object']).columns
if len(non_numeric) > 0:
    print(f"Converting these columns to numeric via one-hot encoding: {list(non_numeric)}")
    X = pd.get_dummies(X, columns=non_numeric, drop_first=True)

print("✅ All features are numeric now.")

# Split the data into train and test sets, stratify if more than one class
stratify = y if len(y.unique()) > 1 else None
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=stratify
)

print("🏋️ Training models...")

# Define models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

best_model = None
best_accuracy = 0

# Train and evaluate models
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    print(f"{name} results:")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1 Score:  {f1:.4f}")

    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model

# Save the best model
os.makedirs('ml_model', exist_ok=True)
joblib.dump(best_model, 'ml_model/churn_model.pkl')
print(f"\n✅ Best model saved as 'ml_model/churn_model.pkl'")
