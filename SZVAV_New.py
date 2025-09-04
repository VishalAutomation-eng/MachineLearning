
# Fault Detection & Control


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve

# 1. Load & Preprocess Data

df = pd.read_csv('/home/vkpande/Documents/machine learning/SZVAV/SZVAV.csv')
# Clean invalid Excel-like strings
import numpy as np
df = df.replace(["#VALUE!", "#DIV/0!", "N/A", "NaN"], np.nan)
df = df.dropna().reset_index(drop=True)

# Convert to datetime & sort
df['Datetime'] = pd.to_datetime(df['Datetime'])
df = df.sort_values('Datetime').reset_index(drop=True)

# Time-based features
df['hour'] = df['Datetime'].dt.hour
df['dow'] = df['Datetime'].dt.dayofweek

# Select numeric columns only
num_cols = [
    c for c in df.columns
    if pd.api.types.is_numeric_dtype(df[c]) and c not in ['Fault Detection Ground Truth']
]

# Create lag and rolling mean features
for c in num_cols:
    df[f'{c}_lag1'] = df[c].shift(1)
    df[f'{c}_roll5'] = df[c].rolling(5, min_periods=1).mean()


# Drop missing values after lag/rolling
df = df.dropna().reset_index(drop=True)

# 2. Features & Target

X = df.drop(columns=['Datetime', 'Fault Detection Ground Truth'])
y = df['Fault Detection Ground Truth']


# 3. Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# 4. Train Random Forest

clf = RandomForestClassifier(
    n_estimators=100,       # fewer trees
    max_depth=6,            # limit tree depth to reduce overfitting
    min_samples_split=10,   # require more samples before splitting
    min_samples_leaf=5,     # each leaf must have at least 5 samples
    random_state=42
)
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]


# 5. Evaluation

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_proba))

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.plot(fpr, tpr, label=f"ROC AUC={roc_auc_score(y_test, y_proba):.2f}")
plt.plot([0, 1], [0, 1], '--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# Feature Importance
importances = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
importances.head(20).plot(kind='barh')
plt.title("Top Feature Importances")
plt.show()


# 6. Optimization Function

try:
    from pulp import LpProblem, LpMinimize, LpVariable, LpStatus, value
    pulp_ok = True
except ImportError:
    pulp_ok = False

def recommend(zone_temp, supply_temp, setpoint=72):
    """
    Recommend fan speed & valve position using optimization (if PuLP available).
    Falls back to heuristic if PuLP not installed.
    """
    if not pulp_ok:
        return {"fan_speed": 0.5, "valve_pos": 0.5, "status": "heuristic"}

    # Define optimization problem
    prob = LpProblem("opt", LpMinimize)
    fan = LpVariable("fan", 0, 1)
    valve = LpVariable("valve", 0, 1)

    # Objective: minimize energy cost (weighted fan + valve)
    prob += fan + 0.2 * valve

    # Comfort constraints
    prob += zone_temp + 0.04 * (supply_temp - zone_temp) * fan + 0.05 * valve <= setpoint + 1
    prob += zone_temp + 0.04 * (supply_temp - zone_temp) * fan + 0.05 * valve >= setpoint - 1

    # Solve
    prob.solve()

    return {
        "fan_speed": float(value(fan)),
        "valve_pos": float(value(valve)),
        "status": LpStatus[prob.status]
    }


# 7. Example Recommendation

latest = df.iloc[-1]
rec = recommend(
    latest['AHU: Return Air Temperature'],
    latest['AHU: Supply Air Temperature']
)
print("\n=== Recommended Control Action ===")
print(rec)
