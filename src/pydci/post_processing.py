import pandas as pd
import numpy as np


class PostProcessor:
    def __init__(self) -> None:
        pass

    def to_pandas_df(self, array_list, column_names, number_intervals):
        categories = [f"Time_Interval_{i}" for i in range(number_intervals)]

        if len(array_list) == 1:
            df = pd.DataFrame(columns=column_names + ["Interval"])
            for i, arr in enumerate(array_list[0]):
                category = categories[i]
                arr_df = pd.DataFrame(arr, columns=column_names)
                arr_df["Interval"] = category
                df = pd.concat([df, arr_df], axis=0, ignore_index=True)
        elif len(array_list) == 2:
            df = pd.DataFrame(columns=["Times"] + column_names + ["Interval"])
            for i, (arr_1, arr_2) in enumerate(zip(array_list[0], array_list[1])):
                category = categories[i]
                arr_1_df = pd.DataFrame(arr_1, columns=["Times"])
                arr_2_df = pd.DataFrame(arr_2, columns=column_names)
                arr_df = pd.concat([arr_1_df, arr_2_df], axis=1)
                arr_df["Interval"] = category
                df = pd.concat([df, arr_df], axis=0, ignore_index=True)

        df["Interval"] = pd.Categorical(df["Interval"], categories=categories)

        return df

    def post_process_model(self, res):
        # Get model parameter dimensions
        number_intervals = len(res["true_values"])
        number_states = res["true_values"][0].shape[1]
        number_parameters = len(res["spatio_temporal_probs"][0].lam_ref)

        # Generate DataFrame labels
        column_labels = [f"X_{i}" for i in range(number_states)]
        lambda_labels = [f"$\pi_{{\lambda_{x}}}$" for x in range(number_parameters)]

        # Post process spatio temporal and mud objects
        predicted_lambdas = [lambdas.lam for lambdas in res["spatio_temporal_probs"]]
        best_residuals = [
            mud_problem["best"]._r for mud_problem in res["mud_probs"]
        ]
        measurement_times = [
            stp.times for i, stp in enumerate(res["spatio_temporal_probs"])
        ]
        measurements = [
            stp.measurements.reshape(stp.n_ts, stp.n_sensors)
            for i, stp in enumerate(res["spatio_temporal_probs"])
        ]
        # best_er = [mud_problem['best']['prob'].expected_ratio() for mud_problem in lv_res['mud_probs']]

        stacked_pushforward_df = pd.concat(res["push_forwards"], axis=0)
        pushforward_df_df = stacked_pushforward_df.reset_index(drop=True)

        # Generate DataFrames
        state_df = self.to_pandas_df(
            [res["times"], res["true_values"]], column_labels, number_intervals
        )
        measure_df = self.to_pandas_df(
            [measurement_times, measurements], column_labels, number_intervals
        )
        predicted_lambdas_df = self.to_pandas_df(
            [predicted_lambdas], lambda_labels, number_intervals
        )
        residuals_df = self.to_pandas_df(
            [best_residuals], ["Residuals"], number_intervals
        )
        # expected_ratio_df = to_pandas_df1(best_er,['Expected_Ratio'])

        return (
            state_df,
            measure_df,
            residuals_df,
            predicted_lambdas_df,
            pushforward_df_df,
        )

    def stats_df(self, residuals_df):
        df_grouped = residuals_df.groupby("Interval")
        return pd.concat(
            [
                df_grouped.mean(),
                df_grouped.std(),
                df_grouped.skew(),
                df_grouped.apply(pd.DataFrame.kurt, numeric_only=True),
            ],
            axis=1,
            keys=["Mean", "Std", "Skew", "Kurt"],
        )
