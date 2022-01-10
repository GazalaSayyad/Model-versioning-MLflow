import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import mlflow.lightgbm
from sklearn.metrics import accuracy_score, log_loss

mlflow.set_tracking_uri("http://localhost:5000")


with mlflow.start_run():
    mlflow.set_experiment("LightGBM_model")
    iris = datasets.load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    train_set = lgb.Dataset(X_train, label=y_train)
    val_set = lgb.Dataset(X_val, label=y_val)

    bst_params = {
        'objective': 'multiclass',
        'num_class': 3,
        'metric': 'multi_logloss',
        'colsample_bytree': 0.8,
        'subsample': 0.8,
        'seed': 42,
    }

    train_params = {
        'num_boost_round': 30,
        'verbose_eval': 5,
        'early_stopping_rounds': 5,
    }

    model = lgb.train(
            bst_params,
            train_set,
            valid_sets=[train_set, val_set],
            valid_names=['train', 'valid'],
            **train_params,
        )
    y_proba = model.predict(X_val)
    y_pred = y_proba.argmax(axis=1)
    loss = log_loss(y_val, y_proba)
    acc = accuracy_score(y_val, y_pred)

        # log metrics
    mlflow.log_metrics({"log_loss": loss, "accuracy": acc})
    mlflow.lightgbm.autolog()  # Enable auto logging


