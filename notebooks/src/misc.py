import pandas as pd

def results_to_df(results):
    reform = {(outerKey, innerKey): values for outerKey, innerDict in results.items() for innerKey, values in innerDict.items()}
    removed_metrics = ["score_time", "fit_time"]
    return (
        pd
        .DataFrame(reform)
        .agg(['mean', 'std'])
        .swapaxes(0, 1)
        .reset_index()
        .rename(columns = {"level_0": "model", "level_1": "metric"}, inplace = False)
        .query("metric not in @removed_metrics")
    )