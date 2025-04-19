from metaflow import (
    FlowSpec,
    step,
    Parameter,
    resources,
    retry,
    catch,
    timeout,
    conda_base,
)


@conda_base(
    libraries={
        "xgboost": "1.7.6",
        "pandas": "1.5.3",
        "scikit-learn": "1.2.2",
        "mlflow": "2.11.1",
        "hyperopt": "0.2.7",
    },
    python="3.9.16",
)
class ClassifierTrainFlow(FlowSpec):

    num_features = Parameter("num_features", type=int)

    @retry(times=2)
    @timeout(seconds=300)
    @catch(var="error")
    @step
    def start(self):
        import pandas as pd

        df_train = pd.read_csv(
            "https://storage.googleapis.com/mlflow-temp-data/train.csv"
        )
        df_test = pd.read_csv(
            "https://storage.googleapis.com/mlflow-temp-data/test.csv"
        )

        self.target_col = "price_range"
        self.feature_cols = [col for col in df_train.columns if col != self.target_col]
        self.train_data = df_train
        self.test_data = df_test

        self.next(self.train_initial_model)

    @retry(times=2)
    @timeout(seconds=600)
    @catch(var="error")
    @step
    def train_initial_model(self):
        from xgboost import XGBClassifier

        X = self.train_data[self.feature_cols]
        y = self.train_data[self.target_col]

        self.model = XGBClassifier()
        self.model.fit(X, y)

        self.next(self.feature_importances)

    @retry(times=2)
    @catch(var="error")
    @step
    def feature_importances(self):
        import pandas as pd

        importances = self.model.feature_importances_
        importances_df = pd.DataFrame(
            {"feature": self.feature_cols, "importance": importances}
        )

        self.selected_features = (
            importances_df.sort_values(by="importance", ascending=False)
            .head(self.num_features)["feature"]
            .values
        )
        self.next(self.hyperparameter_tuning)

    @retry(times=1)
    @timeout(seconds=900)
    @catch(var="error")
    @step
    def hyperparameter_tuning(self):
        from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
        from sklearn.model_selection import cross_val_score
        from xgboost import XGBClassifier
        import mlflow
        import mlflow.xgboost

        X = self.train_data[list(self.selected_features)]
        y = self.train_data[self.target_col]

        def objective(params):
            mlflow.xgboost.autolog(disable=True)
            with mlflow.start_run(nested=True):
                clf = XGBClassifier(eval_metric="mlogloss", **params)
                accuracy = cross_val_score(clf, X, y, cv=3).mean()

                mlflow.set_tags(
                    {"model_type": "xgboost", "run-type": "hyperopt-tuning"}
                )
                mlflow.log_params(params)
                mlflow.log_metric("cv_accuracy", accuracy)

                return {"loss": -accuracy, "status": STATUS_OK}

        search_space = {
            "n_estimators": hp.randint("n_estimators", 20, 200),
            "max_depth": hp.randint("max_depth", 2, 10),
            "learning_rate": hp.uniform("learning_rate", 0.01, 0.3),
            "subsample": hp.uniform("subsample", 0.5, 1.0),
            "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1.0),
            "gamma": hp.uniform("gamma", 0, 5),
        }

        trials = Trials()
        best_params = fmin(
            fn=objective,
            space=search_space,
            algo=tpe.suggest,
            max_evals=25,
            trials=trials,
        )

        self.best_params = best_params
        self.selected_features = list(self.selected_features)
        self.next(self.train_and_register_final_model)

    @timeout(seconds=600)
    @catch(var="error")
    @step
    def train_and_register_final_model(self):
        from xgboost import XGBClassifier
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import train_test_split
        import mlflow
        import mlflow.xgboost

        mlflow.set_tracking_uri("https://mlflow-service-963097677027.us-west2.run.app")
        mlflow.set_experiment("metaflow-experiment")

        X = self.train_data[self.selected_features]
        y = self.train_data[self.target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        best_params = dict(self.best_params)
        best_params["eval_metric"] = "mlogloss"

        model = XGBClassifier(**best_params)
        model.fit(X_train, y_train)

        test_accuracy = accuracy_score(y_test, model.predict(X_test))

        with mlflow.start_run(run_name="final_xgb_model"):
            mlflow.set_tags({"model_type": "xgboost", "final_model": True})
            mlflow.log_params(best_params)
            mlflow.log_metric("test_accuracy", test_accuracy)
            mlflow.xgboost.log_model(
                model,
                artifact_path="xgb_final_model",
                registered_model_name="metaflow-mobile-model",
            )

        self.model = model
        self.test_accuracy = test_accuracy
        self.next(self.end)

    @step
    def end(self):
        print("Final XGBoost model test accuracy:", self.test_accuracy)


if __name__ == "__main__":
    ClassifierTrainFlow()
