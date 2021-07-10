import lightgbm as lgb
from helpers import rmspe_np
from logger import logger


def train_lgbm(
    params: dict,
    train_ds: lgb.Dataset,
    val_ds: lgb.Dataset,
    fit_params: dict = {},
):

    params.update(fit_params)

    model = lgb.train(
        params=params,
        train_set=train_ds,
        valid_sets=[val_ds],
        valid_names=["val"],
        categorical_feature="auto",
        early_stopping_rounds=20,
    )
    return model


def custom_cv(params: dict, X, Y, validator, fit_params: dict = {}):

    metrics = []

    for train_index, test_index in validator.split(X):
        ds = lgb.Dataset(X.iloc[train_index], label=Y.iloc[train_index])
        val_ds = lgb.Dataset(X.iloc[test_index], label=Y.iloc[test_index])

        model = train_lgbm(
            params=params, train_ds=ds, val_ds=val_ds, fit_params=fit_params
        )
        preds = model.predict(X.iloc[test_index])
        metrics.append(rmspe_np(Y.iloc[test_index], preds))

    logger.info(f"CV metrics: {metrics}")
    return metrics


def kfold(params: dict, X, Y, kfolder, fit_params: dict = {}):

    artifacts = []

    for train_index, test_index in kfolder.split(X):
        ds = lgb.Dataset(X.iloc[train_index], label=Y.iloc[train_index])
        val_ds = lgb.Dataset(X.iloc[test_index], label=Y.iloc[test_index])

        model = train_lgbm(
            params=params, train_ds=ds, val_ds=val_ds, fit_params=fit_params
        )
        preds = model.predict(X.iloc[test_index])
        artifacts.append(
            {
                "model": model,
                "preds": preds,
                "metrics": rmspe_np(Y.iloc[test_index], preds),
            }
        )

    logger.info(f"CV metrics: {[x['metrics'] for x in artifacts]}")
    return artifacts
