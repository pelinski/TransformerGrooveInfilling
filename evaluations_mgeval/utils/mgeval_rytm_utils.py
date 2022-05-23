import numpy as np
import pandas as pd
from GrooveEvaluator.GrooveEvaluator import evaluator  # import your version of evaluator!!
import pickle
import matplotlib.pyplot as plt
from scipy import stats, integrate
import matplotlib.pyplot as plt
import os
from itertools import cycle
import random
import textwrap
from tqdm import tqdm
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, matthews_corrcoef
from datetime import datetime

def flatten(t):
    return [item for sublist in t for item in sublist]

del boxplot_soundfonts
def boxplot_soundfonts(sets, fs = 30, legend_fs = 10, legend_ncols = 3, fig_path=None,
                              show=False, ncols=4, figsize=(20, 10), color_map="pastel1", filename=None,
                              force_ylim=None, shift_colors_by=0, max_label_len=100,
                              sharey=False, share_legend=False, show_legend=False, bbox_to_anchor=(1.2, 0.12)):
    # fontsize
    n_plots = len(sets[list(sets.keys())[0]].keys())
    nrows = int(np.ceil(n_plots / ncols))
    fig, axes = plt.subplots(nrows=int(np.ceil(n_plots / ncols)), ncols=ncols, figsize=figsize, sharey=sharey)
    cnt = 0

    cmap = get_cmap(len(sets.keys())+shift_colors_by, name=color_map)

    for feature in sets[list(sets.keys())[0]].keys():
        yrange = 0

        labels = []
        handles = []
        datas_for_range = []

        if nrows == 1 and ncols == 1:
            ax_ = axes
        elif nrows == 1:
            ax_ = axes[(cnt % ncols)]
        elif ncols == 1:
            ax_ = axes[(cnt // ncols)]
        else:
            ax_ = axes[(cnt // ncols)][(cnt % ncols)]


        for set_ix, set_name in enumerate(sets):
            print(set_name, feature)
            labels.append(f"{set_name[:max_label_len].replace('gmd', 'GMD')}")
            data = sets[set_name][feature]
            datas_for_range.append(min(data))
            datas_for_range.append(max(data))

            #yrange = max(yrange, max(data) - min(data))
            handle = ax_.boxplot(data, positions=[set_ix], labels=[set_name], notch=True, widths=.5,
                               patch_artist=True,
                               boxprops=dict(facecolor=cmap(set_ix+shift_colors_by), alpha = 1), whis=(0, 100))  # , boxprops=dict(facecolor=cmap(set_ix))
            handles.append(handle['boxes'][0])

            y = data
            x = np.random.normal(set_ix, 0.04, size=len(y))
            ax_.scatter(x, y, marker='o', color=cmap(set_ix+shift_colors_by), alpha=0.8, s=12)

            ax_.set_title(feature.split("::")[-1], fontsize=fs)
            # ax_.tick_params(axis='x', labelrotation=90, labelsize=fs * 1)
            ax_.get_xaxis().set_visible(False)

        if share_legend is not True:
            if show_legend is True:
                ax_.legend(handles, labels, loc='lower center', prop={'size':legend_fs}, ncol=legend_ncols)

        if force_ylim is not None:
            ax_.set_ylim(bottom=force_ylim[0], top=force_ylim[1])

        for label in ax_.get_yticklabels():
          label.set_fontsize(fs * .75)

        cnt += 1

        if force_ylim is None and share_legend is not True:
            if show_legend is True:
                y_min, y_max = ax_.get_ylim()
                ax_.set_ylim(y_min - 0.5 * (y_max - y_min), y_max)

    if share_legend is True:
        if show_legend is True:
            fig.legend(handles, labels, bbox_to_anchor=bbox_to_anchor, loc='lower right', prop={'size':legend_fs}, ncol=legend_ncols)
        #fig.subplots_adjust(bottom=-1)

    if fig_path is not None:
        if filename is None:
            filename = ""
            for set_name in sets.keys():
                filename = filename + f"{set_name}_"

        filename = os.path.join(fig_path, filename)
        fig.savefig(filename+".png")

    if show is True:
        fig.show()


# ---- boxplot raw data
def boxplot_absolute_measures(sets, fs = 30, legend_fs = 10, legend_ncols = 3, fig_path=None,
                              show=False, ncols=4, figsize=(20, 10), color_map="pastel1", filename=None,
                              force_ylim=None, shift_colors_by=0, max_label_len=100,
                              sharey=False, share_legend=False, show_legend=False):
    # fontsize
    n_plots = len(sets[list(sets.keys())[0]].keys())
    nrows = int(np.ceil(n_plots / ncols))
    fig, axes = plt.subplots(nrows=int(np.ceil(n_plots / ncols)), ncols=ncols, figsize=figsize, sharey=sharey)
    cnt = 0

    cmap = get_cmap(len(sets.keys())+shift_colors_by, name=color_map)

    for feature in sets[list(sets.keys())[0]].keys():
        yrange = 0

        labels = []
        handles = []
        datas_for_range = []

        if nrows == 1 and ncols == 1:
            ax_ = axes
        elif nrows == 1:
            ax_ = axes[(cnt % ncols)]
        elif ncols == 1:
            ax_ = axes[(cnt // ncols)]
        else:
            ax_ = axes[(cnt // ncols)][(cnt % ncols)]


        for set_ix, set_name in enumerate(sets):
            print(set_name, feature)
            labels.append(f"{set_name[:max_label_len].replace('gmd', 'GMD')}")
            data = sets[set_name][feature]
            datas_for_range.append(min(data))
            datas_for_range.append(max(data))

            #yrange = max(yrange, max(data) - min(data))
            handle = ax_.boxplot(data, positions=[set_ix], labels=[set_name], notch=True, widths=0.1,
                               patch_artist=True,
                               boxprops=dict(facecolor=cmap(set_ix+shift_colors_by)))  # , boxprops=dict(facecolor=cmap(set_ix))
            handles.append(handle['boxes'][0])



            violin_parts = ax_.violinplot(data, positions=[set_ix])
            for pc in violin_parts['bodies']:
              pc.set_facecolor(cmap(set_ix+shift_colors_by))
              pc.set_edgecolor('black')
              pc.set_linewidth(1)

            ax_.set_title(feature.split("::")[-1], fontsize=fs)
            # ax_.tick_params(axis='x', labelrotation=90, labelsize=fs * 1)
            ax_.get_xaxis().set_visible(False)

        if share_legend is not True:
            if show_legend is True:
                ax_.legend(handles, labels, loc='lower right', prop={'size':legend_fs}, ncol=legend_ncols)

        if force_ylim is not None:
            ax_.set_ylim(bottom=force_ylim[0], top=force_ylim[1])

        for label in ax_.get_yticklabels():
          label.set_fontsize(fs * .75)

        cnt += 1

        if force_ylim is None and share_legend is not True:
            if show_legend is True:
                y_min, y_max = ax_.get_ylim()
                ax_.set_ylim(y_min - 0.5 * (y_max - y_min), y_max)

    if share_legend is True:
        if show_legend is True:
            fig.legend(handles, labels, loc='lower right', prop={'size':legend_fs}, ncol=legend_ncols)
        #fig.subplots_adjust(bottom=-1)

    if fig_path is not None:
        if filename is None:
            filename = ""
            for set_name in sets.keys():
                filename = filename + f"{set_name}_"

        filename = os.path.join(fig_path, filename)
        fig.savefig(filename+".png")

    if show is True:
        fig.show()

def get_positive_negative_vel_stats(sets_evals, ground_truth_key = ["GMD"]):
    stats_sets = dict()
    for set_name, evaluator_ in sets_evals.items():
        vel_actual = np.array([])
        vel_all_Hits = np.array([])
        vel_TP = np.array([])
        vel_FP = np.array([])
        vel_actual_mean = np.array([])
        vel_actual_std = np.array([])
        vel_all_Hits_mean = np.array([])
        vel_all_Hits_std = np.array([])
        vel_TP_mean = np.array([])
        vel_TP_std = np.array([])
        vel_FP_mean = np.array([])
        vel_FP_std = np.array([])

        for (true_values, predictions) in zip(evaluator_._gt_hvos_array, evaluator_._prediction_hvos_array):
            true_vels = true_values[:, 9: 18][np.nonzero(true_values[:, 9: 18])]
            true_vels = np.where(true_vels>0.5, 0.5, true_vels)
            vel_actual=np.append(vel_actual, true_vels)
            vel_actual_mean=np.append(vel_actual_mean, np.nanmean(true_values[:, 9: 18][np.nonzero(true_values[:, :9])]))
            vel_actual_std=np.append(vel_actual_std, np.nanstd(true_values[:, 9: 18][np.nonzero(true_values[:, :9])]))
            vels_predicted = np.array(predictions[:, 9: 18]).flatten()
            actual_hits = np.array(true_values[:, :9]).flatten()
            predicted_hits = np.array(predictions[:, :9]).flatten()
            all_predicted_hit_indices, = (predicted_hits==1).nonzero()
            vel_all_Hits = np.append(vel_all_Hits, vels_predicted[all_predicted_hit_indices])
            vel_all_Hits_mean = np.append(vel_all_Hits_mean, np.nanmean(vels_predicted[all_predicted_hit_indices]))
            vel_all_Hits_std = np.append(vel_all_Hits_std, np.nanstd(vels_predicted[all_predicted_hit_indices]))
            true_hit_indices, = np.logical_and(actual_hits==1, predicted_hits==1).nonzero()
            vel_TP = np.append(vel_TP, vels_predicted[true_hit_indices])
            vel_TP_mean = np.append(vel_TP_mean, np.nanmean(vels_predicted[true_hit_indices]))
            vel_TP_std = np.append(vel_TP_std, np.nanstd(vels_predicted[true_hit_indices]))
            false_hit_indices, = np.logical_and(actual_hits==0, predicted_hits==1).nonzero()
            vel_FP = np.append(vel_FP, vels_predicted[false_hit_indices])
            vel_FP_mean = np.append(vel_FP_mean, np.nanmean(vels_predicted[false_hit_indices]))
            vel_FP_std = np.append(vel_FP_std, np.nanstd(vels_predicted[false_hit_indices]))

        stats_sets.update(
            {
                set_name:
                    {
                        "Average Velocity": np.nan_to_num(vel_all_Hits_mean),
                        "True Hits (mean per Loop)": np.nan_to_num(vel_TP_mean),
                        "False Hits (mean per Loop)": np.nan_to_num(vel_FP_mean),
                        "All Hits (std per Loop)": np.nan_to_num(vel_all_Hits_std),
                        "True Hits (std per Loop)": np.nan_to_num(vel_TP_std),
                        "False Hits (std per Loop)": np.nan_to_num(vel_FP_std),
                    } if set_name not in ground_truth_key else
                    {
                        "Average Velocity": np.nan_to_num(vel_all_Hits_mean),
                        "True Hits (mean per Loop)": np.nan_to_num(vel_all_Hits_mean),
                        "False Hits (mean per Loop)": np.nan_to_num(vel_all_Hits_mean),
                        "All Hits (std per Loop)": np.nan_to_num(vel_all_Hits_std),
                        "True Hits (std per Loop)": np.nan_to_num(vel_all_Hits_std),
                        "False Hits (std per Loop)": np.nan_to_num(vel_all_Hits_std),
                    }

             }
        )

    return stats_sets

def get_positive_negative_utiming_stats(sets_evals, ground_truth_key = ["GMD"]):
    stats_sets = dict()
    for set_name, evaluator_ in sets_evals.items():
        uTiming_actual = np.array([])
        uTiming_all_Hits = np.array([])
        uTiming_TP = np.array([])
        uTiming_FP = np.array([])
        uTiming_actual_mean = np.array([])
        uTiming_actual_std = np.array([])
        uTiming_all_Hits_mean = np.array([])
        uTiming_all_Hits_std = np.array([])
        uTiming_TP_mean = np.array([])
        uTiming_TP_std = np.array([])
        uTiming_FP_mean = np.array([])
        uTiming_FP_std = np.array([])

        for (true_values, predictions) in zip(evaluator_._gt_hvos_array, evaluator_._prediction_hvos_array):
            true_utimings = true_values[:, 18:][np.nonzero(true_values[:, 18:])]
            true_utimings = np.where(true_utimings>0.5, 0.5, true_utimings)
            uTiming_actual=np.append(uTiming_actual, true_utimings)
            uTiming_actual_mean=np.append(uTiming_actual_mean, np.nanmean(true_values[:, 18:][np.nonzero(true_values[:, :9])]))
            uTiming_actual_std=np.append(uTiming_actual_std, np.nanstd(true_values[:, 18:][np.nonzero(true_values[:, :9])]))
            uts_predicted = np.array(predictions[:, 18:]).flatten()
            actual_hits = np.array(true_values[:, :9]).flatten()
            predicted_hits = np.array(predictions[:, :9]).flatten()
            all_predicted_hit_indices, = (predicted_hits==1).nonzero()
            uTiming_all_Hits = np.append(uTiming_all_Hits, uts_predicted[all_predicted_hit_indices])
            uTiming_all_Hits_mean = np.append(uTiming_all_Hits_mean, np.nanmean(uts_predicted[all_predicted_hit_indices]))
            uTiming_all_Hits_std = np.append(uTiming_all_Hits_std, np.nanstd(uts_predicted[all_predicted_hit_indices]))
            true_hit_indices, = np.logical_and(actual_hits==1, predicted_hits==1).nonzero()
            uTiming_TP = np.append(uTiming_TP, uts_predicted[true_hit_indices])
            uTiming_TP_mean = np.append(uTiming_TP_mean, np.nanmean(uts_predicted[true_hit_indices]))
            uTiming_TP_std = np.append(uTiming_TP_std, np.nanstd(uts_predicted[true_hit_indices]))
            false_hit_indices, = np.logical_and(actual_hits==0, predicted_hits==1).nonzero()
            uTiming_FP = np.append(uTiming_FP, uts_predicted[false_hit_indices])
            uTiming_FP_mean = np.append(uTiming_FP_mean, np.nanmean(uts_predicted[false_hit_indices]))
            uTiming_FP_std = np.append(uTiming_FP_std, np.nanstd(uts_predicted[false_hit_indices]))

        stats_sets.update(
            {
                set_name:
                    {
                        "Average Offset": np.nan_to_num(uTiming_all_Hits_mean),
                        "True Hits (mean per Loop)": np.nan_to_num(uTiming_TP_mean),
                        "False Hits (mean per Loop)": np.nan_to_num(uTiming_FP_mean),
                        "All Hits (std per Loop)": np.nan_to_num(uTiming_all_Hits_std),
                        "True Hits (std per Loop)": np.nan_to_num(uTiming_TP_std),
                        "False Hits (std per Loop)": np.nan_to_num(uTiming_FP_std),
                    } if set_name not in ground_truth_key else
                    {
                        "Average Offset": np.nan_to_num(uTiming_all_Hits_mean),
                        "True Hits (mean per Loop)": np.nan_to_num(uTiming_all_Hits_mean),
                        "False Hits (mean per Loop)": np.nan_to_num(uTiming_all_Hits_mean),
                        "All Hits (std per Loop)": np.nan_to_num(uTiming_all_Hits_std),
                        "True Hits (std per Loop)": np.nan_to_num(uTiming_all_Hits_std),
                        "False Hits (std per Loop)": np.nan_to_num(uTiming_all_Hits_std),
                    }

            }
        )

    return stats_sets


def get_positive_negative_hit_stats(sets_evals, hit_weight=1):
    stats_sets = dict()
    for set_name, evaluator_ in sets_evals.items():
        stats_sets.update({set_name:
            {
                'Accuracy': [
                    accuracy_score(true_values[:, :9].flatten(), predictions[:, :9].flatten(),
                                   sample_weight=((hit_weight - 1)*predictions[:, :9].flatten()+1))
                    for (true_values, predictions) in zip(evaluator_._gt_hvos_array,
                                                          evaluator_._prediction_hvos_array)],
                'Precision': [
                    precision_score(true_values[:, :9].flatten(), predictions[:, :9].flatten(),
                                   sample_weight=((hit_weight - 1)*predictions[:, :9].flatten()+1))
                    for (true_values, predictions) in zip(evaluator_._gt_hvos_array,
                                                          evaluator_._prediction_hvos_array)],
                'Recall': [
                    recall_score(true_values[:, :9].flatten(), predictions[:, :9].flatten(),
                                   sample_weight=((hit_weight - 1)*predictions[:, :9].flatten()+1))
                    for (true_values, predictions) in zip(evaluator_._gt_hvos_array,
                                                          evaluator_._prediction_hvos_array)],
                'F1-Score': [
                    f1_score(true_values[:, :9].flatten(), predictions[:, :9].flatten(),
                                   sample_weight=((hit_weight - 1)*predictions[:, :9].flatten()+1))
                    for (true_values, predictions) in zip(evaluator_._gt_hvos_array,
                                                          evaluator_._prediction_hvos_array)],
                'MCC (Hit/Silence Classification)': [
                    matthews_corrcoef(true_values[:, :9].flatten(),
                                      predictions[:, :9].flatten())
                    for (true_values, predictions) in zip(evaluator_._gt_hvos_array,
                                                          evaluator_._prediction_hvos_array)],
                'MCC (Correct Number of Instruments at each step)': [
                    matthews_corrcoef(true_values[:, :9].sum(axis=1).flatten(),
                                      predictions[:, :9].sum(axis=1).flatten())
                    for (true_values, predictions) in zip(evaluator_._gt_hvos_array,
                                                          evaluator_._prediction_hvos_array)]

            }}
        )
    for set_name, evaluator_ in sets_evals.items():

        Actual_P_array = []
        Total_predicted_array = []
        TP_array = []
        FP_array = []
        PPV_array = []
        FDR_array = []
        TPR_array = []
        FPR_array = []
        FP_over_N = []
        FN_over_P = []
        for (true_values, predictions) in zip(evaluator_._gt_hvos_array, evaluator_._prediction_hvos_array):
            true_values, predictions = np.array(flatten(true_values[:, :9])), np.array(flatten(predictions[:, :9]))
            flat_size = len(true_values)
            Actual_P = np.count_nonzero(true_values)
            Actual_N = flat_size - Actual_P
            TP = ((predictions == 1) & (true_values == 1)).sum()
            FP = ((predictions == 1) & (true_values == 0)).sum()
            FN = ((predictions == 0) & (true_values == 1)).sum()
            # https://en.wikipedia.org/wiki/Precision_and_recall
            PPV_array.append(TP / (TP + FP) if (TP + FP) > 0 else 0)
            FDR_array.append(FP / (TP + FP) if (TP + FP) > 0 else 0)
            TPR_array.append(TP / Actual_P)
            FPR_array.append(FP / Actual_N)
            TP_array.append(TP)
            FP_array.append(FP)
            FP_over_N.append(FP/Actual_N)
            FN_over_P.append(FN / Actual_P)
            Actual_P_array.append(Actual_P)
            Total_predicted_array.append((predictions == 1).sum())

        stats_sets[set_name].update({
            "TPR": TPR_array,
            "FPR": FPR_array,
            "PPV": PPV_array,
            "FDR": FDR_array,
            "Ratio of Silences Predicted as Hits": FP_over_N,
            "Ratio of Hits Predicted as Silences": FN_over_P,
            "GT Hits": Actual_P_array,
            "Predicted Hits": Total_predicted_array,
            "True Hits (Matching GT)": TP_array,
            "False Hits (Different from GT)": FP_array,
        })
    return stats_sets


def sample_uniformly(gmd_eval, num_samples):
    uniques = 0
    master_ids = []
    # get indices and corresponding master_id
    for ix, subset in enumerate(gmd_eval._prediction_subsets):

        print(gmd_eval._prediction_tags[ix])
        for index, hvo in enumerate(subset):
            master_ids.append(hvo.metadata.master_id)
        #print(len(master_ids), len(set(sorted(master_ids))))

    uniques = len(set(sorted(master_ids)))
    print(uniques)

    masterid_index_tuple = list(zip(master_ids, list(range(len(master_ids)))))

    all_pairs = sorted(masterid_index_tuple)

    sampled_pairs = []
    sampled_master_ids = []

    already_sampled = []

    sample_ix = 0
    while len(sampled_pairs) < num_samples:
        if sample_ix not in already_sampled:
            sample_tuple = masterid_index_tuple[sample_ix]

            if sample_tuple[0] not in sampled_master_ids:
                print(sample_ix, masterid_index_tuple[sample_ix])
                sampled_pairs.append(sample_tuple)
                sampled_master_ids.append(sample_tuple[0])

            if len(sampled_master_ids) >= uniques:
                print("sampled_master_ids", sampled_master_ids)
                sampled_master_ids = []

        sample_ix = (sample_ix + 1) % len(masterid_index_tuple)

    final_indices = []
    for sample_pair in sampled_pairs:
        final_indices.append(sample_pair[1])

    return final_indices

def sample_uniformly_gt(gmd_eval, num_samples):
    uniques = 0
    master_ids = []
    # get indices and corresponding master_id
    for ix, subset in enumerate(gmd_eval._gt_subsets):

        print(gmd_eval._gt_tags[ix])
        for index, hvo in enumerate(subset):
            master_ids.append(hvo.metadata.master_id)
        #print(len(master_ids), len(set(sorted(master_ids))))

    uniques = len(set(sorted(master_ids)))
    print(uniques)

    masterid_index_tuple = list(zip(master_ids, list(range(len(master_ids)))))

    all_pairs = sorted(masterid_index_tuple)

    sampled_pairs = []
    sampled_master_ids = []

    already_sampled = []

    sample_ix = 0
    while len(sampled_pairs) < num_samples:
        if sample_ix not in already_sampled:
            sample_tuple = masterid_index_tuple[sample_ix]

            if sample_tuple[0] not in sampled_master_ids:
                print(sample_ix, masterid_index_tuple[sample_ix])
                sampled_pairs.append(sample_tuple)
                sampled_master_ids.append(sample_tuple[0])

            if len(sampled_master_ids) >= uniques:
                print("sampled_master_ids", sampled_master_ids)
                sampled_master_ids = []

        sample_ix = (sample_ix + 1) % len(masterid_index_tuple)

    final_indices = []
    for sample_pair in sampled_pairs:
        final_indices.append(sample_pair[1])

    return final_indices

def get_cmap(n, name='tab20c'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)


def get_pd_feats_from_evaluator(evaluator_):
    # extracts the prediction features from a evaluator
    return evaluator_.prediction_SubSet_Evaluator.feature_extractor.get_global_features_dicts(True)


def get_gt_feats_from_evaluator(evaluator_):
    # extracts the ground truth features from a evaluator
    return evaluator_.gt_SubSet_Evaluator.feature_extractor.get_global_features_dicts(True)


def flatten_subset_genres(feature_dict, remove_nans=True):
    # combines the subset samples irregardless of their genre
    flattened_feature_dict = {x: np.array([]) for x in feature_dict.keys()}
    for feature_key in flattened_feature_dict.keys():
        for subset_key, subset_samples in feature_dict[feature_key].items():
            subset_samples = subset_samples[~np.isnan(subset_samples)] if remove_nans is True else subset_samples
            flattened_feature_dict[feature_key] = np.append(flattened_feature_dict[feature_key], subset_samples)
    return flattened_feature_dict


def get_absolute_measures_for_single_set(flat_feature_dict, csv_file=None):
    # Gets absolute measures of a set according to
    # Yang, Li-Chia, and Alexander Lerch. "On the evaluation of generative models in music."
    #           Neural Computing and Applications 32.9 (2020): 4773-4784.

    stats = []  # list of lists stats[i] corresponds to [mean, std, min, max, median, q1, q3]
    labels = []

    for key in flat_feature_dict.keys():
        data = flat_feature_dict[key]
        # Calc stats
        stats.append(
            [np.mean(data), np.std(data), np.min(data), np.max(data), np.percentile(data, 50), np.percentile(data, 25),
             np.percentile(data, 75)])
        labels.append(key)

    df2 = pd.DataFrame(np.round(np.array(stats),3).transpose(),
                       ["mean", "std", "min", "max", "median", "q1", "q3"],
                       labels).transpose()

    if csv_file is not None:
        df2.to_csv(csv_file)

    return df2


def get_absolute_measures_for_multiple_sets(sets_of_flat_feature_dict, csv_file=None):

    sets_dfs = []
    sets_df_keys = []
    for set_tag, set_feat_dict in sets_of_flat_feature_dict.items():
        sets_df_keys.append(set_tag)
        sets_dfs.append(get_absolute_measures_for_single_set(set_feat_dict))
        print(f"--------- Finished Calculating Absolute Measures for set {set_tag} --------------")

    pd_final = pd.concat(sets_dfs, keys=sets_df_keys)

    if csv_file is not None:
        pd_final.to_csv(csv_file)

    return pd_final


def get_intraset_distances_from_array(features_array):
    # Calculates l2 norm distance of each sample with every other sample
    intraset_distances = []
    features_array = features_array[np.logical_not(np.isnan(features_array))]
    ix = np.arange(features_array.size)
    for current_i, current_feature in enumerate(features_array):
        distance_to_all = np.abs(features_array[np.delete(ix, current_i)] - current_feature)
        intraset_distances.extend(distance_to_all)
    return np.array(intraset_distances)


def get_intraset_distances_from_set(flat_feature_dict):

    intraset_distances_feat_dict = {}

    for key, flat_feat_array in flat_feature_dict.items():
        intraset_distances_feat_dict[key] = get_intraset_distances_from_array(flat_feat_array)

    return intraset_distances_feat_dict


def get_interset_distances(flat_feature_dict_a, flat_feature_dict_b):

    interset_distances_feat_dict = {}

    for key, flat_feat_array in flat_feature_dict_a.items():
        flat_feat_array = flat_feat_array[np.logical_not(np.isnan(flat_feat_array))]
        flat_feat_array_b = flat_feature_dict_b[key][np.logical_not(np.isnan(flat_feature_dict_b[key]))]

        interset_distances = []
        for current_i, current_feature_in_a in enumerate(flat_feat_array):
            distance_to_all = np.abs(flat_feat_array_b - current_feature_in_a)
            interset_distances.extend(distance_to_all)

        interset_distances_feat_dict[key] = interset_distances

    return interset_distances_feat_dict


def kl_dist(A, B, pdf_A=None, pdf_B=None, num_sample=100):
    # Calculate KL distance between the two PDF

    # calc pdfs if necessary - helps to avoid redundant calculations for pdfs if already done
    pdf_A = stats.gaussian_kde(A) if pdf_A is None else pdf_A
    pdf_B = stats.gaussian_kde(B) if pdf_B is None else pdf_B

    sample_A = np.linspace(np.min(A), np.max(A), num_sample)
    sample_B = np.linspace(np.min(B), np.max(B), num_sample)

    return stats.entropy(pdf_A(sample_A), pdf_B(sample_B))


def overlap_area(A, B, pdf_A, pdf_B, max_sample_size=100):
    # Calculate overlap between the two PDF

    # calc pdfs if necessary - helps to avoid redundant calculations for pdfs if already done
    return integrate.quad(lambda x: min(pdf_A(x), pdf_B(x)), np.min((np.min(A), np.min(B))), np.max((np.max(A), np.max(B))))[0]


def convert_multi_feature_distances_to_pdf(distances_features_dict):
    pdf_dict = {}
    for feature_key, distances_for_feature in distances_features_dict.items():
        pdf_dict[feature_key] = stats.gaussian_kde(distances_for_feature)
    return pdf_dict


def get_KL_OA_for_multi_feature_distances(distances_dict_A, distances_dict_B,
                                          pdf_distances_dict_A, pdf_distances_dict_B,
                                          num_sample=1000):
    KL_dict = {}
    OA_dict = {}

    for feature_key in distances_dict_A.keys():
        KL_dict[feature_key] = kl_dist(
            distances_dict_A[feature_key], distances_dict_B[feature_key],
            pdf_A=pdf_distances_dict_A[feature_key], pdf_B=pdf_distances_dict_B[feature_key],
            num_sample=num_sample)
        print(f"KL_{feature_key}")
        OA_dict[feature_key] = overlap_area(
            distances_dict_A[feature_key], distances_dict_B[feature_key],
            pdf_A=pdf_distances_dict_A[feature_key], pdf_B=pdf_distances_dict_B[feature_key])
        print(f"OA_{feature_key}")

    return KL_dict, OA_dict

def compare_two_sets_against_ground_truth(gt, set1, set2, set_labels=['gt', 'set1', 'set2'], csv_path=None,
                                          calc_OA_downsample_size = 100):
    # generates a table similar to that of No.4 in Yang et. al.
    if not os.path.exists(os.path.dirname(csv_path)):
        os.makedirs(os.path.dirname(csv_path))

    gt_intra = get_intraset_distances_from_set(gt)
    print("calculated_gt_intra")
    set1_intra = get_intraset_distances_from_set(set1)
    print("calculated_set1_intra")
    set2_intra = get_intraset_distances_from_set(set2)
    print("calculated_set2_intra")
    set1_inter_gt = get_interset_distances(set1, gt)
    print("calculated_set1_inter_gt")
    set2_inter_gt = get_interset_distances(set2, gt)
    print("calculated_set2_inter_gt")
    pdf_gt_intra = convert_multi_feature_distances_to_pdf(gt_intra)
    print("gt_pdf")
    pdf_set1_inter_gt = convert_multi_feature_distances_to_pdf(set1_inter_gt)
    print("set1_pdf")
    pdf_set2_inter_gt = convert_multi_feature_distances_to_pdf(set2_inter_gt)
    print("set2_pdf")
    KL_set1inter_gt_intra, OA_set1inter_gt_intra = get_KL_OA_for_multi_feature_distances(
        set1_inter_gt, gt_intra,
        pdf_set1_inter_gt, pdf_gt_intra, num_sample=100)
    print("KL_set1")
    KL_set2inter_gt_intra, OA_set2inter_gt_intra = get_KL_OA_for_multi_feature_distances(
        set2_inter_gt, gt_intra,
        pdf_set2_inter_gt, pdf_gt_intra, num_sample=100)
    print("KL_set2")
    features = gt_intra.keys()

    data_for_feature = []
    for feature in features:
        try:
            data_row = []
            # calculate mean and std of gt_intra
            data_row.extend([np.round(np.mean(gt_intra[feature]), 3), np.round(np.std(gt_intra[feature]), 3)])
            data_row.extend([np.round(np.mean(set1_intra[feature]), 3), np.round(np.std(set1_intra[feature]), 3)])
            data_row.extend([np.round(KL_set1inter_gt_intra[feature], 3), np.round(OA_set1inter_gt_intra[feature], 3)])
            data_row.extend([np.round(np.mean(set2_intra[feature]), 3), np.round(np.std(set2_intra[feature]), 3)])
            data_row.extend([np.round(KL_set2inter_gt_intra[feature], 3), np.round(OA_set2inter_gt_intra[feature], 3)])

            data_for_feature.append(data_row)
        except:
            print(f"Can't calculate KL or OA for feature {feature}")
    header = pd.MultiIndex.from_arrays([
        np.array(
            [set_labels[0], set_labels[0], set_labels[1], set_labels[1], set_labels[1], set_labels[1],
             set_labels[2], set_labels[2], set_labels[2], set_labels[2]]
        ),
        np.array(
            ["Intra-set", "Intra-set", "Intra-set", "Intra-set", "Inter-set", "Inter-set", "Intra-set", "Intra-set",
             "Inter-set", "Inter-set"]
        ),
        np.array(
            ["mean", "STD", "mean", "STD", "KL", "OA", "mean", "STD", "KL", "OA"]
        ),
    ])

    index = [x.split("::")[-1] for x in features]
    df = pd.DataFrame(data_for_feature,
                      index=index,
                      columns=header)

    if csv_path is not None:
        df.to_csv(csv_path)

    raw_data = (gt_intra, set1_intra, set2_intra, set1_inter_gt, set2_inter_gt, pdf_gt_intra, pdf_set1_inter_gt, pdf_set2_inter_gt)
    return df, raw_data

#del plot_inter_intra_pdfs
def plot_inter_intra_pdfs(raw_data, fig_path, set_labels, show=True, ncols=4, figsize=(20,20),
                          legend_fs=12, fs=12, plot_all_intra=False, share_legend=False,bbox_to_anchor = (.9, 0.12)):
    # raw_data is a tuple of  (gt_intra, set1_intra, set2_intra, set1_inter_gt, set2_inter_gt, pdf_gt_intra, pdf_set1_inter_gt, pdf_set2_inter_gt)

    gt_intra, set1_intra, set2_intra, set1_inter_gt, set2_inter_gt, pdf_gt_intra, pdf_set1_inter_gt, pdf_set2_inter_gt = raw_data
    pdf_set1_intra = convert_multi_feature_distances_to_pdf(set1_intra)
    pdf_set2_intra = convert_multi_feature_distances_to_pdf(set2_intra)

    n_plots = len(list(gt_intra.keys()))
    nrows = int(np.ceil(n_plots / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharey=False)

    num_sample = 1000
    for i, key in tqdm(enumerate(gt_intra.keys())):
        handles = []
        labels = []
        if nrows == 1 and ncols == 1:
            ax_ = axes
        elif nrows == 1:
            ax_ = axes[(i % ncols)]
        elif ncols == 1:
            ax_ = axes[(i // ncols)]
        else:
            ax_ = axes[(i // ncols)][(i % ncols)]

        if plot_all_intra is True:
            x = np.linspace(np.min(set1_intra[key]), np.max(set1_intra[key]), num_sample)
            y1 = pdf_set1_intra[key](x)
            h, = ax_.plot(x, y1, c='b', label=f"pdf of Intra ({set_labels[1].replace('ClosedHH', 'IH').replace(' (Symbolic)', 'S')})", linestyle='dashed')
            handles.append(h)
            labels.append(f"pdf of Intra ({set_labels[1].replace('ClosedHH', 'IH').replace(' (Symbolic)', 'S')})")
        x = np.linspace(np.min(set1_inter_gt[key]), np.max(set1_inter_gt[key]), num_sample)
        y2 = pdf_set1_inter_gt[key](x)
        h, = ax_.plot(x, y2, c='b', label=f"pdf of Inter ({set_labels[1].replace('ClosedHH', 'IH').replace(' (Symbolic)', 'S')})", linestyle='solid')
        handles.append(h)
        labels.append(f"pdf of Inter ({set_labels[1].replace('ClosedHH', 'IH').replace(' (Symbolic)', 'S')})")

        if plot_all_intra is True:
            x = np.linspace(np.min(set2_intra[key]), np.max(set2_intra[key]), num_sample)
            h, = ax_.plot(x, pdf_set2_intra[key](x), c='c', label=f"Intra Distance ({set_labels[2].replace('ClosedHH', 'IH').replace(' (Symbolic)', 'S')})", linestyle='dashed')
            handles.append(h)
            labels.append(f"Intra Distance ({set_labels[2].replace('ClosedHH', 'IH').replace(' (Symbolic)', 'S')})")
        x = np.linspace(np.min(set2_inter_gt[key]), np.max(set2_inter_gt[key]), num_sample)
        h, = ax_.plot(x, pdf_set2_inter_gt[key](x), c='c', label=f"Inter Distance ({set_labels[2].replace('ClosedHH', 'IH').replace(' (Symbolic)', 'S')})", linestyle='solid')
        handles.append(h)
        labels.append(f"pdf of Inter ({set_labels[2].replace('ClosedHH', 'IH').replace(' (Symbolic)', 'S')})")

        x = np.linspace(np.min(gt_intra[key]), np.max(gt_intra[key]), num_sample)
        h, = ax_.plot(x, pdf_gt_intra[key](x), c='r', label=f"Intra Distance ({set_labels[0].replace('ClosedHH', 'IH').replace(' (Symbolic)', 'S')})", linestyle='dashdot', linewidth=2)
        handles.append(h)
        labels.append(f"pdf of Intra ({set_labels[0].replace('ClosedHH', 'IH').replace(' (Symbolic)', 'S')})")
        '''if share_legend is True:
            fig.legend(handles, labels, bbox_to_anchor=bbox_to_anchor, loc='lower right', prop={'size': legend_fs})
        '''
        ax_.legend(handles, labels, loc='upper right', prop={'size': legend_fs})

        type_key = key.split("::")[0]

        title = key.split("::")[-1] if  type_key is 'Statistical' else key
        title = title.split("::")[-1]
        ax_.set_title(f"{title}", fontsize=fs)
        ax_.set_xlabel("Euclidean Distance", fontsize=fs)
        ax_.set_ylabel("Density", fontsize=fs)

        for label in ax_.get_yticklabels():
          label.set_fontsize(fs * .75)
        for label in ax_.get_xticklabels():
          label.set_fontsize(fs * .75)

    if fig_path is not None:
        path = os.path.join(fig_path, "plots")
        os.makedirs(path , exist_ok=True)
        filename = os.path.join(path, f"{title}_{set_labels[0]}_{set_labels[1]}_{set_labels[2]}")
        plt.savefig(filename)

    if show is True:
        plt.show()

    return fig, axes

def plot_intersets(analysis_dataframe, fig_path, set_labels, set1_name='First Model', set2_name='Second Model',
                   show=False, ncols=2, figsize=(20, 10), max_features_in_plot=7, legend_fs=6, fs=8, add_legend=False,
                   legend_loc = 'lower right',
                   add_text=True, min_line_len_for_text=1, legend_ncols=1, force_xlim=None, force_ylim=None):

    df = analysis_dataframe

    cmap = get_cmap(max_features_in_plot, name="Dark2")
    lines = ['-', '--', '-.', ':']
    linecycler = cycle(lines)

    nfeatures = len(analysis_dataframe.index)
    nplots = int(np.ceil(nfeatures/max_features_in_plot))
    ncols = min(ncols, nplots)
    nrows = int(np.ceil(nplots/ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharey=False, sharex=True)

    handles, labels = [], []

    for i, index in tqdm(enumerate(df.index)):


        handles = [] if i % max_features_in_plot == 0 else handles
        labels = [] if i % max_features_in_plot == 0 else labels
        ax_ = None

        if nrows > 1 and ncols>1:
            print([i//(ncols*max_features_in_plot)], [i//max_features_in_plot])
            row_ix = i//(ncols*max_features_in_plot)
            ax_ = axes[row_ix][i//(max_features_in_plot)//nrows]
        if nrows == 1 and ncols == 1:
            ax_ = axes
        if nrows > 1 and ncols == 1:
            ax_ = axes[i//(ncols*max_features_in_plot)]
        if nrows == 1 and ncols > 1:
            ax_ = axes[i//max_features_in_plot]

        x1 = df[(set_labels[1], 'Inter-set', 'KL')][index]
        y1 = df[(set_labels[1], 'Inter-set', 'OA')][index]
        x2 = df[(set_labels[2], 'Inter-set', 'KL')][index]
        y2 = df[(set_labels[2], 'Inter-set', 'OA')][index]

        ax_.scatter(x1, y1, c=cmap(i % max_features_in_plot), marker="^")
        ax_.scatter(x2, y2, c=cmap(i % max_features_in_plot), marker="s")

        if add_text is True:
            if ((x2-x1)**2-(y2-y1)**2)**0.5 >= min_line_len_for_text or x1 > 0.5 or x2 > 0.5 or y1 < 0.6 or y2 < 0.6:
                dy = (y2 - y1)
                dx = (x2 - x1)
                right_point = (x1, y1) if x1 > x2 else (x2, y2)
                center_point = ((x1+x2)/2, (y1+y2)/2)
                center_point = ((right_point[0]+center_point[0])/2, (right_point[1]+center_point[1])/2)
                center_point = ((right_point[0] + center_point[0]) / 2, (right_point[1] + center_point[1]) / 2)
                loc = ((right_point[0] + center_point[0]) / 2, (right_point[1] + center_point[1]) / 2)
                rotn = np.degrees(np.arctan2(dy, dx))
                rotn = (rotn + 180.0) if 90 < rotn < 270 else rotn

                ax_.text(*right_point, index.split("::")[-1], c=cmap(i % max_features_in_plot), fontsize=legend_fs)

        h_, = ax_.plot([x1, x2], [y1, y2], c=cmap(i % max_features_in_plot), label=index.split("::")[-1],
                 linestyle=next(linecycler))  # , linewidth=.3*(i+1))

        handles.append(h_)
        lab_ = index.split("::")[-1] if index.split("::")[0]=="Statistical" else index.replace("::", " ")
        if lab_ == "Accuracy":
            lab_ = "uTiming Accuracy"
        labels.append(lab_)

        if add_legend is True:
            ax_.legend(handles, labels, loc=legend_loc, prop={'size': legend_fs}, ncol=legend_ncols)

        ax_.set_xlabel("KL", fontsize=fs)
        ax_.set_ylabel("OA", fontsize=fs)

    if ncols > 1 and nrows > 1 :
        for ax in axes:
            y_min, y_max = ax.get_ylim()
            x_min, x_max = ax.get_xlim()
            if add_legend is True:
                ax.set_xlim(x_min - 0.1 * (x_max - x_min), max(x_max * 1.2, 1))
                ax.set_ylim(y_min - 1 * (y_max - y_min), min(y_max* 1.2, 1.1) )
    else:
        ax = axes
        y_min, y_max = ax.get_ylim()
        x_min, x_max = ax.get_xlim()
        if add_legend is True:
            ax.set_xlim(x_min - 0.1 * (x_max - x_min), x_max * 1.2)
            ax.set_ylim(y_min - 1 * (y_max - y_min), min(y_max * 1.2, 1.1))

    if force_xlim is not None:
        ax.set_xlim(force_xlim[0], force_xlim[1])
    if force_ylim is not None:
        ax.set_ylim(force_ylim[0], force_ylim[1])

    fig.suptitle(r'$\bigtriangleup$' + "  " + f"{set1_name}" + "    " + r'$\boxdot$' + "  " + f"{set2_name}",
                 fontsize=fs)

    if fig_path is not None:
        os.makedirs(fig_path, exist_ok=True)
        now = datetime.now()
        time_txt = f"{now.day}_{now.month}_{now.year}-{now.hour}-{now.min}-{now.second}"
        filename = os.path.join(fig_path, f"Inter ({set_labels[1]},{set_labels[0]})_Inter ({set_labels[2]},{set_labels[0]})_vs_Intra ({set_labels[0]})_at_{time_txt}")
        plt.savefig(filename)

    if show is True:
        fig.show()

    return fig, axes
