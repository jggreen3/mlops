from metaflow import FlowSpec, step, Parameter, Flow


class ClassifierPredictFlow(FlowSpec):
    data_path = Parameter("data_path", type=str, required=True)

    @step
    def start(self):
        run = Flow("ClassifierTrainFlow").latest_run
        self.train_run_id = run.pathspec
        self.model = run["end"].task.data.model
        self.target_col = run["end"].task.data.target_col
        self.selected_features = run["end"].task.data.selected_features

        print("Input path", self.data_path)
        self.next(self.load_data)

    @step
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
