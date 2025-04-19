from metaflow import (
    FlowSpec,
    step,
    Parameter,
    Flow,
    resources,
    retry,
    timeout,
    catch,
    conda_base,
)


@conda_base(
    libraries={"pandas": "1.5.3", "scikit-learn": "1.2.2", "xgboost": "1.7.6"},
    python="3.9.16",
)
class ClassifierPredictFlow(FlowSpec):
    data_path = Parameter("data_path", type=str, required=True)

    @step
    @retry(times=2)
    @timeout(seconds=300)
    @catch(var="error")
    def start(self):
        run = Flow("ClassifierTrainFlow").latest_run
        self.train_run_id = run.pathspec
        self.model = run["end"].task.data.model
        self.target_col = run["end"].task.data.target_col
        self.selected_features = run["end"].task.data.selected_features

        print("Input path", self.data_path)
        self.next(self.load_data)

    @step
    @resources(cpu=2, memory=4096)
    @retry(times=2)
    @timeout(seconds=300)
    @catch(var="error")
    def load_data(self):
        import pandas as pd

        self.df_test = pd.read_csv(self.data_path)
        self.next(self.end)

    @step
    def end(self):
        print("Model", self.model)
        print(
            "Predicted class",
            self.model.predict(self.df_test[self.selected_features])[0],
        )


if __name__ == "__main__":
    ClassifierPredictFlow()
