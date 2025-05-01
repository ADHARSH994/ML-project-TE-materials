import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


# Step 1: Load Dataset
file_path = "D:\\download 2\\tm2.csv"  # Change this to your actual file path
df = pd.read_csv(file_path)

# Step 2: Separate Features (X) and Target (y)
X = df.drop(columns=["zT"])  # Features: Material properties
y = df["zT"]  # Target: Figure of Merit (ZT)

# Step 3: Standardize Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Fit and transform data

# Step 4: Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 5: Train Random Forest Model
rf_model = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
rf_model.fit(X_train, y_train)

# Step 6: Predict on Test Data
y_pred = rf_model.predict(X_test)

# Step 7: Evaluate Model
rmse = mean_squared_error(y_test, y_pred) ** 0.5  # Manually take the square root
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"Model Performance:")
print(f"✅ RMSE: {rmse:.4f}")
print(f"✅ R² Score: {r2:.4f}")
print(f"📊 mae Score: {mae:.4f}")

# Step 8: Save the Trained Model and Scaler for Future Use
joblib.dump(rf_model, "D:\\Download\\zt_model.pkl")  # Save model
joblib.dump(scaler, "D:\\Download\\scaler.pkl")  # Save scaler

print("✅ Model and scaler saved successfully!")

# Step 9: Visualize Actual vs. Predicted ZT
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, color="blue", alpha=0.6, label="Out of Samples")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle="--", color="red", label="Base Line")  # y=x line
plt.xlabel("Actual ZT")
plt.ylabel("Predicted ZT")
plt.title("Actual vs. Predicted ZT (XGBoost)")
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.minorticks_on()
plt.tick_params(axis='both', which='both', direction='in', length=6)
plt.text(min(y_test) + 0.2, max(y_pred) - 0.2, f"R²: {r2:.4f}\nRMSE: {rmse:.4f}\nMAE: {mae:.4f}", fontsize=12, bbox=dict(facecolor='white', alpha=0.6))
plt.legend()
plt.show()
