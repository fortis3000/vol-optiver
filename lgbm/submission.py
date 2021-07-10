import glob
import lightgbm as lgb
from features import get_book_features


features = ["vol", "vol2"]

if __name__ == "__main__":

    files_test = glob.glob(r"data/book_test.parquet/*")

    dataset_test = get_book_features(files_test)

    model = lgb.Booster(
        model_file=r"models/lgbm_2021-07-05_16:26:21/model.txt"
    )

    dataset_test["target"] = model.predict(dataset_test[features].values)

    dataset_test[["row_id", "target"]].to_csv("submission.csv", index=False)
