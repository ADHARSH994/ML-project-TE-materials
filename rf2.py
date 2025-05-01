import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


file_path = "D:\\download\\featureset.csv"
df = pd.read_csv(file_path)


base_features = [
    "Mendeleev_M_A", "covalent_radius_bragg_M_A", "covalent_radius_bragg_M_S", 
    "fusion_heat_M_S", "thermal_conductivity_M_A", "thermal_conductivity_M_S", "electron_affinity_M_A", 
    "specific_heat_capacity_M_A", "density_M_A", "atomic_volume_M_S", "vdw_radius_M_S", 
    "lattice_constant_M_S", "specific_heat_capacity_M_S", "electron_affinity_M_S", 
    "density_M_E", "lattice_constant_M_E", "specific_heat_capacity_M_E", 
    "dipole_polarizability_M_E", "vdw_radius_M_G"
]

T_features_list = [
    "Mendeleev_M_A", "Mendeleev_Min", "fusion_heat_M_S", 
    "thermal_conductivity_M_A", "specific_heat_capacity_M_A", "atomic_volume_M_S", "density_M_S", 
    "lattice_constant_M_S", "specific_heat_capacity_M_S", "electron_affinity_M_S", "dipole_polarizability_M_S", 
    "specific_heat_capacity_M_E", "dipole_polarizability_M_A","atomic_weight_Max","atomic_weight_M_A"
]


if "ZT" not in df.columns or "T" not in df.columns:
    raise ValueError(" not found ")

# Create new features by multiplying relevant features with Temperature
T_features = {f"T*{feature}": df["T"] * df[feature] for feature in T_features_list if feature in df.columns}
df_T = pd.DataFrame(T_features)

# Select input features (both T* features and base features)
X = pd.concat([df_T, df[base_features]], axis=1)
y = df["ZT"]

# Handle missing values (Fill with column mean)
X.fillna(X.mean(), inplace=True)

# Standardize the input features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Random Forest Model
rf_model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)

# Make Predictions
y_pred = rf_model.predict(X_test)

# Evaluate Model Performance
rmse = mean_squared_error(y_test, y_pred) ** 0.5  # Manually take the square root
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f" Model Trained Successfully!")
print(f" RMSE: {rmse:.4f}")
print(f" R² Score: {r2:.4f}")
print(f" mae Score: {mae:.4f}")

# Save the trained model and scaler
joblib.dump(rf_model, "D:\\download\\Rf\\randomforest_zt_model3.pkl")
joblib.dump(scaler, "D:\\download\\Rf\\scaler4.pkl")

# Plot Actual vs Predicted ZT
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, color="blue", alpha=0.6, label="Out of Samples")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle="--", color="red", label="Base Line")  # y=x line
plt.xlabel("Actual ZT")
plt.ylabel("Predicted ZT")
plt.title("Actual vs. Predicted ZT (Random Forest)")
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.minorticks_on()
plt.tick_params(axis='both', which='both', direction='in', length=6)
plt.text(min(y_test) + 0.2, max(y_pred) - 0.2, f"R²: {r2:.4f}\nRMSE: {rmse:.4f}\nMAE: {mae:.4f}", fontsize=12, bbox=dict(facecolor='white', alpha=0.6))
plt.legend()
plt.show()

print(" Random Forest model and scaler saved for future use!")
