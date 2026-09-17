# %%
import pandas as pd

# Load FreeSolv dataset
df = pd.read_csv("freesolv.csv")

print("Dataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())


# %%
print("\nHydration free energy statistics:")
print(df["y"].describe())


# %%
from rdkit import Chem

# Convert SMILES strings into RDKit molecular objects
df["Molecule"] = df["smiles"].apply(Chem.MolFromSmiles)

# Check for invalid SMILES
invalid_smiles = df["Molecule"].isnull().sum()

print("Invalid SMILES:", invalid_smiles)
print("Valid molecules:", len(df) - invalid_smiles)



# %%
from rdkit.Chem import Descriptors, Lipinski

# Calculate basic molecular descriptors
df["MolWt"] = df["Molecule"].apply(Descriptors.MolWt)
df["LogP"] = df["Molecule"].apply(Descriptors.MolLogP)
df["HBD"] = df["Molecule"].apply(Lipinski.NumHDonors)
df["HBA"] = df["Molecule"].apply(Lipinski.NumHAcceptors)

print("\nMolecular descriptors:")
print(df[["smiles", "MolWt", "LogP", "HBD", "HBA"]].head())

# %%
# Define input features (X) and target (y)

X = df[["MolWt", "LogP", "HBD", "HBA"]]
y = df["y"]

print("X shape:", X.shape)
print("y shape:", y.shape)

print("\nX columns:")
print(X.columns.tolist())

print("\nFirst 5 target values:")
print(y.head())


# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)



# %%
from sklearn.linear_model import LinearRegression

# Create Linear Regression model
linear_model = LinearRegression()

# Train the model
linear_model.fit(X_train, y_train)

# Generate predictions
y_pred_linear = linear_model.predict(X_test)

print("Linear Regression completed.")
print("First 5 predictions:")
print(y_pred_linear[:5])

# %%
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

mae_linear = mean_absolute_error(y_test, y_pred_linear)
rmse_linear = np.sqrt(mean_squared_error(y_test, y_pred_linear))
r2_linear = r2_score(y_test, y_pred_linear)

print("Linear Regression Performance:")
print("MAE :", mae_linear)
print("RMSE:", rmse_linear)
print("R²  :", r2_linear)



from sklearn.svm import SVR

svr_model = SVR(kernel="rbf")

svr_model.fit(X_train, y_train)

y_pred_svr = svr_model.predict(X_test)

print("SVR completed.")
print("First 5 predictions:")
print(y_pred_svr[:5])

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

mae_svr = mean_absolute_error(y_test, y_pred_svr)
rmse_svr = np.sqrt(mean_squared_error(y_test, y_pred_svr))
r2_svr = r2_score(y_test, y_pred_svr)

print("SVR Performance:")
print("MAE :", mae_svr)
print("RMSE:", rmse_svr)
print("R²  :", r2_svr)


from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svr_scaled = SVR(kernel="rbf")

svr_scaled.fit(X_train_scaled, y_train)

y_pred_svr_scaled = svr_scaled.predict(X_test_scaled)

print("Scaled SVR completed.")
print("First 5 predictions:")
print(y_pred_svr_scaled[:5])

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

mae_svr_scaled = mean_absolute_error(y_test, y_pred_svr_scaled)
rmse_svr_scaled = np.sqrt(mean_squared_error(y_test, y_pred_svr_scaled))
r2_svr_scaled = r2_score(y_test, y_pred_svr_scaled)

print("Scaled SVR Performance:")
print("MAE :", mae_svr_scaled)
print("RMSE:", rmse_svr_scaled)
print("R²  :", r2_svr_scaled)



from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

print("Random Forest completed.")
print("First 5 predictions:")
print(y_pred_rf[:5])

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)

print("Random Forest Performance:")
print("MAE :", mae_rf)
print("RMSE:", rmse_rf)
print("R²  :", r2_rf)

import xgboost

print("XGBoost version:", xgboost.__version__)


from xgboost import XGBRegressor

xgb_model = XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    random_state=42
)

xgb_model.fit(X_train, y_train)

y_pred_xgb = xgb_model.predict(X_test)

print("XGBoost completed.")
print("First 5 predictions:")
print(y_pred_xgb[:5])

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
r2_xgb = r2_score(y_test, y_pred_xgb)

print("XGBoost Performance:")
print("MAE :", mae_xgb)
print("RMSE:", rmse_xgb)
print("R²  :", r2_xgb)





import tensorflow as tf

print("TensorFlow version:", tf.__version__)

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

nn_model = Sequential([
    Dense(64, activation="relu", input_shape=(4,)),
    Dense(32, activation="relu"),
    Dense(1)
])

nn_model.compile(
    optimizer="adam",
    loss="mse"
)

print("Neural Network architecture created.")
nn_model.summary()

# ============================================================
# TRAIN NEURAL NETWORK
# ============================================================

history = nn_model.fit(
    X_train_scaled,
    y_train,
    validation_split=0.20,
    epochs=200,
    batch_size=32,
    verbose=1
)

print("Neural Network training completed.")
y_pred_nn = nn_model.predict(X_test_scaled).flatten()

print("Neural Network predictions generated.")
print("First 5 predictions:")
print(y_pred_nn[:5])

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae_nn = mean_absolute_error(y_test, y_pred_nn)
rmse_nn = mean_squared_error(y_test, y_pred_nn) ** 0.5
r2_nn = r2_score(y_test, y_pred_nn)

print("\nNeural Network Performance:")
print("MAE :", mae_nn)
print("RMSE:", rmse_nn)
print("R²  :", r2_nn)




from sklearn.linear_model import Ridge, ElasticNet

# Ridge Regression
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train_scaled, y_train)

y_pred_ridge = ridge_model.predict(X_test_scaled)

mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
rmse_ridge = mean_squared_error(y_test, y_pred_ridge) ** 0.5
r2_ridge = r2_score(y_test, y_pred_ridge)

print("\nRidge Regression Performance:")
print("MAE :", mae_ridge)
print("RMSE:", rmse_ridge)
print("R²  :", r2_ridge)


# ElasticNet Regression
elastic_model = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
elastic_model.fit(X_train_scaled, y_train)

y_pred_elastic = elastic_model.predict(X_test_scaled)

mae_elastic = mean_absolute_error(y_test, y_pred_elastic)
rmse_elastic = mean_squared_error(y_test, y_pred_elastic) ** 0.5
r2_elastic = r2_score(y_test, y_pred_elastic)

print("\nElasticNet Regression Performance:")
print("MAE :", mae_elastic)
print("RMSE:", rmse_elastic)
print("R²  :", r2_elastic)


# ============================================================
# MODEL COMPARISON
# ============================================================

results = {
    "Linear Regression": {
        "MAE": mae_linear,
        "RMSE": rmse_linear,
        "R2": r2_linear
    },

    "Ridge": {
        "MAE": mae_ridge,
        "RMSE": rmse_ridge,
        "R2": r2_ridge
    },

    "ElasticNet": {
        "MAE": mae_elastic,
        "RMSE": rmse_elastic,
        "R2": r2_elastic
    },

    "SVR (Scaled)": {
        "MAE": mae_svr_scaled,
        "RMSE": rmse_svr_scaled,
        "R2": r2_svr_scaled
    },

    "Random Forest": {
        "MAE": mae_rf,
        "RMSE": rmse_rf,
        "R2": r2_rf
    },

    "XGBoost": {
        "MAE": mae_xgb,
        "RMSE": rmse_xgb,
        "R2": r2_xgb
    },

    "Neural Network": {
        "MAE": mae_nn,
        "RMSE": rmse_nn,
        "R2": r2_nn
    }
}

results_df = pd.DataFrame(results).T

print("\n================ MODEL COMPARISON ================")
print(results_df.round(4))


# ============================================================
# R² COMPARISON PLOT
# ============================================================

import matplotlib.pyplot as plt

models = results_df.index
r2_values = results_df["R2"]

plt.figure(figsize=(10, 6))
plt.bar(models, r2_values)

plt.xlabel("Model")
plt.ylabel("R² Score")
plt.title("Model Performance Comparison — R²")

plt.ylim(0, 1)
plt.xticks(rotation=30)
plt.tight_layout()

plt.savefig("model_comparison_r2.png", dpi=300)
plt.show()


# ============================================================
# ACTUAL vs PREDICTED — ALL MODELS
# ============================================================

plt.figure(figsize=(9, 8))

plt.scatter(y_test, y_pred_linear, alpha=0.5, label="Linear Regression")
plt.scatter(y_test, y_pred_ridge, alpha=0.5, label="Ridge")
plt.scatter(y_test, y_pred_elastic, alpha=0.5, label="ElasticNet")
plt.scatter(y_test, y_pred_svr_scaled, alpha=0.5, label="SVR")
plt.scatter(y_test, y_pred_rf, alpha=0.5, label="Random Forest")
plt.scatter(y_test, y_pred_xgb, alpha=0.5, label="XGBoost")
plt.scatter(y_test, y_pred_nn, alpha=0.5, label="Neural Network")

min_val = min(
    y_test.min(),
    y_pred_linear.min(),
    y_pred_ridge.min(),
    y_pred_elastic.min(),
    y_pred_svr_scaled.min(),
    y_pred_rf.min(),
    y_pred_xgb.min(),
    y_pred_nn.min()
)

max_val = max(
    y_test.max(),
    y_pred_linear.max(),
    y_pred_ridge.max(),
    y_pred_elastic.max(),
    y_pred_svr_scaled.max(),
    y_pred_rf.max(),
    y_pred_xgb.max(),
    y_pred_nn.max()
)

plt.plot(
    [min_val, max_val],
    [min_val, max_val],
    linestyle="--",
    label="Perfect Prediction"
)

plt.xlabel("Actual Target")
plt.ylabel("Predicted Target")
plt.title("Actual vs Predicted — Model Comparison")
plt.legend()
plt.tight_layout()

plt.savefig("actual_vs_predicted_models.png", dpi=300)
plt.show()

# ============================================================
# NEURAL NETWORK TRAINING CURVE
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Neural Network Training and Validation Loss")

plt.legend()
plt.tight_layout()

plt.savefig("neural_network_training_curve.png", dpi=300)
plt.show()