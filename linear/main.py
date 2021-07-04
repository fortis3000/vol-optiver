import os
import shutil
import glob

import pandas as pd
import tensorflow as tf

from features import get_book_features
from helpers import seed_everything, load_config, get_dt_str
from logger import logger

from linear.model import linear_model, r2_score

CONFIG_FILE = "config.yml"
CONFIG_FILE_LR = r"linear/config_lr.yml"

config = load_config(CONFIG_FILE)
config.update(load_config(CONFIG_FILE_LR))

model_folder = os.path.join(
    config["MODEL_PATH"], "linear" + "_" + get_dt_str()
)

os.mkdir(model_folder)
shutil.copy2(r"linear/main.py", model_folder)
shutil.copy2("config.yml", model_folder)
shutil.copy2(r"linear/config_lr.yml", model_folder)


LOG_FILE = os.path.join(model_folder, "logs.txt")


seed_everything(config["SEED"])


if __name__ == "__main__":

    dataset = pd.merge(
        pd.read_parquet(
            os.path.join(config["DATAPATH"], "trade_train.parquet")
        ),
        pd.read_csv(os.path.join(config["DATAPATH"], "train.csv")),
        how="inner",
        on=["stock_id", "time_id"],
    )

    files = glob.glob("data/book_train.parquet/*")

    dataset_new = get_book_features(files)

    dataset_new = pd.merge(
        dataset_new,
        dataset[["time_id", "stock_id", "target"]],
        how="inner",
        on=["time_id", "stock_id"],
    )

    logger.info("Dataset was built")

    linear_model.compile(
        optimizer=tf.optimizers.Adamax(learning_rate=0.001),
        loss="mean_squared_error",
        metrics=[
            "mean_absolute_error",
            "mean_absolute_percentage_error",
            r2_score,
        ],
    )

    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.1,
            patience=20,
            verbose=1,
            min_delta=1e-4,
            min_lr=1e-8,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            min_delta=0,
            patience=20,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.TensorBoard(log_dir=model_folder, profile_batch=0),
    ]

    features = ["vol", "vol2"]

    history = linear_model.fit(
        dataset_new[features].values,
        dataset_new["target"].values,
        epochs=config["EPOCHS"],
        verbose=1,
        validation_split=config["VALIDATION_SPLIT"],
        batch_size=config["BATCH_SIZE"],
        callbacks=callbacks,
        shuffle=True,
    )

    logger.info("Model was trained successfully")

    linear_model.save(model_folder)
