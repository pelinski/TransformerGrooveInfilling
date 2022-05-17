import numpy as np
import pandas as pd


def get_stats_from_samples(feature_value_dict, csv_file=None):
    stats = []  # list of lists stats[i] corresponds to [mean, std, min, max, median, q1, q3]
    labels = []

    for key in feature_value_dict.keys():
        # Compile all genre data together
        data = []
        for key2 in feature_value_dict[key].keys():
            data.extend(feature_value_dict[key][key2])

        # Calc stats
        stats.append(
            [np.mean(data), np.std(data), np.min(data), np.max(data), np.percentile(data, 50), np.percentile(data, 25),
             np.percentile(data, 75)])
        labels.append(key)

    df2 = pd.DataFrame(np.array(stats).transpose(),
                       ["mean", "std", "min", "max", "median", "q1", "q3"],
                       labels)

    if csv_file is not None:
        df2.to_csv(csv_file)

    return df2


# del get_stats_from_evaluator


def get_stats_from_evaluator(evaluator_, calc_gt=True, calc_pred=True, csv_file=None):
    gt_df = get_stats_from_samples(
        evaluator_.gt_SubSet_Evaluator.feature_extractor.get_global_features_dicts(True), None
    ) if (calc_gt and evaluator_.gt_SubSet_Evaluator is not None) else None

    pd_df = get_stats_from_samples(
        evaluator_.prediction_SubSet_Evaluator.feature_extractor.get_global_features_dicts(True), None
    ) if (calc_gt and evaluator_.prediction_SubSet_Evaluator is not None) else None

    keys = []
    if gt_df is not None:
        keys.extend(gt_df.columns)
    if pd_df is not None:
        keys.extend(pd_df.columns)

    datas = []
    labels = []

    for key in keys:
        if gt_df is not None:
            data = gt_df.iloc[:][key].values if key in gt_df.columns else [None] * 7
        else:
            data = [None] * 7

        labels.append(key + "__Ground_Truth")
        datas.append(data)

        if pd_df is not None:
            data = pd_df.iloc[:][key].values if key in pd_df.columns else [None] * 7
        else:
            data = [None] * 7

        datas.append(data)
        labels.append(key + "__Prediction")

    df2 = pd.DataFrame(np.array(datas).transpose(),
                       ["mean", "std", "min", "max", "median", "q1", "q3"],
                       labels
                       )

    df2 = df2.loc[:, ~df2.columns.duplicated()]  # cols are duplicated

    if csv_file is not None:
        df2.to_csv(csv_file)

    return df2


if __name__ == '__main__':
    from GrooveEvaluator import evaluator  # import your version of evaluator!!
    import pickle

    run_names = [
        "hopeful-gorge-252_Epoch_90",
        "misunderstood-bush-246_Epoch_26",
        "rosy-durian-248_Epoch_26",
        "solar-shadow-247_Epoch_41",
        "groovae"]

    for run_name in run_names:
        test_set_evaluator = pickle.load(open(
            f"post_processing_scripts/evaluators_monotonic_groove_transformer_v1/validation_set_evaluator_run_{run_name}.Eval","rb"))

        df_compiled = get_stats_from_evaluator(test_set_evaluator, csv_file="misc/test_data.csv").transpose()
        df_compiled.to_csv(f"post_processing_scripts/evaluators_monotonic_groove_transformer_v1/{run_name}.csv")
