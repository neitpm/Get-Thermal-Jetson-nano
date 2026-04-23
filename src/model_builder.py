import json
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout
from tkan import TKAN


def build_mtkan_model(model_config_path="model/model_config_train.json"):
    with open(model_config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    window_size = cfg["window_size"]
    n_features = cfg["n_features"]
    n_targets = cfg["n_targets"]
    tkan_units_1 = cfg["tkan_units_1"]
    tkan_units_2 = cfg["tkan_units_2"]
    dense_units = cfg["dense_units"]
    dropout_rate = cfg["dropout_rate"]

    model = keras.Sequential([
        keras.layers.Input(shape=(window_size, n_features)),
        TKAN(tkan_units_1, return_sequences=True),
        Dropout(dropout_rate),
        TKAN(tkan_units_2, return_sequences=False),
        Dense(dense_units, activation="relu"),
        Dropout(dropout_rate),
        Dense(n_targets)
    ])

    return model
