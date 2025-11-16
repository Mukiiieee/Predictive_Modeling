# ==========================================
# MODEL TRAINING SCRIPT (ALL 14 CELLS COMBINED)
# ==========================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, RocCurveDisplay
)
import tensorflow as tf
from tensorflow import keras

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Load dataset
df = pd.read_csv("Dataset_ATS_cleaned.csv")

# Encode target
df["churn"] = df["churn"].map({"No": 0, "Yes": 1}).astype(int)
X = df.drop(columns=["churn"])
y = df["churn"]

# Identify columns
num_cols = ["seniorcitizen", "tenure", "monthlycharges"]
cat_cols = [c for c in X.columns if c not in num_cols]

# Split data
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=SEED)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=SEED)

# Preprocessing
numeric_tf = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_tf = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_tf, num_cols),
        ("cat", categorical_tf, cat_cols)
    ]
)

preprocessor.fit(X_train)
X_train_t = preprocessor.transform(X_train)
X_val_t   = preprocessor.transform(X_val)
X_test_t  = preprocessor.transform(X_test)

input_dim = X_train_t.shape[1]

# Build ANN
def build_ann(input_dim, width_1=128, width_2=64, width_3=32,
              dropout_rate=0.3, l2_reg=1e-4, lr=1e-3):

    reg = keras.regularizers.l2(l2_reg)
    inputs = keras.Input(shape=(input_dim,))

    x = keras.layers.Dense(width_1, activation="relu", kernel_regularizer=reg)(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(dropout_rate)(x)

    x = keras.layers.Dense(width_2, activation="relu", kernel_regularizer=reg)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(dropout_rate)(x)

    x = keras.layers.Dense(width_3, activation="relu", kernel_regularizer=reg)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(dropout_rate)(x)

    outputs = keras.layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs)
    optimizer = keras.optimizers.Adam(learning_rate=lr)
    
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="roc_auc"),
                 keras.metrics.AUC(name="pr_auc", curve="PR"),
                 keras.metrics.Precision(name="precision"),
                 keras.metrics.Recall(name="recall")]
    )
    return model

model = build_ann(input_dim=input_dim)

# Class weights
pos = y_train.sum()
neg = len(y_train) - pos
pos_weight = neg / pos
class_weight = {0: 1.0, 1: float(pos_weight)}

# Training
callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_pr_auc", patience=10,
                                  mode="max", restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor="val_pr_auc", factor=0.5,
                                      patience=5, mode="max", min_lr=1e-5)
]

history = model.fit(
    X_train_t, y_train,
    validation_data=(X_val_t, y_val),
    epochs=100,
    batch_size=512,
    class_weight=class_weight,
    callbacks=callbacks,
    verbose=1
)

# Test evaluation
y_proba_test = model.predict(X_test_t).ravel()
y_pred_test = (y_proba_test >= 0.5).astype(int)

acc  = accuracy_score(y_test, y_pred_test)
prec = precision_score(y_test, y_pred_test)
rec  = recall_score(y_test, y_pred_test)
f1   = f1_score(y_test, y_pred_test)
roc  = roc_auc_score(y_test, y_proba_test)

print("Accuracy:", acc)
print("Precision:", prec)
print("Recall:", rec)
print("F1:", f1)
print("ROC-AUC:", roc)

# Save outputs
model.save("churn_ann_model.keras")
pd.DataFrame({
    "y_true": y_test,
    "y_pred": y_pred_test,
    "y_proba": y_proba_test
}).to_csv("churn_ann_test_predictions.csv", index=False)
