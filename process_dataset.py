import copy
import json
from dataset import (
    GrooveMidiDatasetInfilling,
    GrooveMidiDatasetInfillingSymbolic,
    GrooveMidiDatasetInfillingRandom,
)

from preprocessed_dataset.Subset_Creators.subsetters import GrooveMidiSubsetter

## load dataset parameters and paths

with open("datasets/subset_info.json") as f:
    subset_info = json.load(f)
with open("datasets/dataset_parameters.json") as f:
    params = json.load(f)

# parse json values
for experiment in params.keys():
    # load subset_info to params
    params[experiment]["subset_info"] = subset_info
    if "thres_range" in params[experiment]:
        # convert thres range to tuple
        params[experiment]["thres_range"] = (
            params[experiment]["thres_range"][0],
            params[experiment]["thres_range"][1],
        )
    if (
        "voices_params" in params[experiment]
        and params[experiment]["voices_params"]["k"] == "None"
    ):
        params[experiment]["voices_params"]["k"] = None


def process_dataset(params, exp):
    _, subset_list = GrooveMidiSubsetter(
        pickle_source_path=params["subset_info"]["pickle_source_path"],
        subset=params["subset_info"]["subset"],
        hvo_pickle_filename=params["subset_info"]["hvo_pickle_filename"],
        list_of_filter_dicts_for_subsets=[params["subset_info"]["filters"]],
    ).create_subsets()

    if exp == "InfillingClosedHH_Symbolic":
        _dataset = GrooveMidiDatasetInfillingSymbolic(data=subset_list[0], **params)
    elif exp == "InfillingRandom" or exp == "InfillingRandomLow":
        _dataset = GrooveMidiDatasetInfillingRandom(data=subset_list[0], **params)
    else:
        _dataset = GrooveMidiDatasetInfilling(data=subset_list[0], **params)

    return _dataset


def load_processed_dataset(load_dataset_path, exp):
    if exp == "InfillingClosedHH_Symbolic":
        print("Loading GrooveMidiDatasetInfillingSymbolic...")
        _dataset = GrooveMidiDatasetInfillingSymbolic(
            load_dataset_path=load_dataset_path
        )
    elif exp == "InfillingRandom" or exp == "InfillingRandomLow":
        print("Loading GrooveMidiDatasetInfillingRandom...")
        _dataset = GrooveMidiDatasetInfillingRandom(load_dataset_path=load_dataset_path)
    else:
        print("Loading GrooveMidiDatasetInfilling...")
        _dataset = GrooveMidiDatasetInfilling(load_dataset_path=load_dataset_path)

    return _dataset


if __name__ == "__main__":

    testing = False

    # change experiment and split here
    exps = [
        "InfillingRandom",
        "InfillingRandomLow",
        "InfillingKicksAndSnares",
    ]
    splits = ["train", "test", "validation"]

    for exp in exps:
        if testing:
            params[exp]["subset_info"]["filters"]["master_id"] = [
                "drummer9/session1/8",
                "drummer9/session1/7",
                "drummer9/session1/12",
            ]
            params[exp]["dataset_name"] = params[exp]["dataset_name"] + "_testing"
            params[exp]["save_dataset_path"] = (
                "datasets/" + params[exp]["dataset_name"] + "/"
            )

        print(
            "------------------------\n"
            + params[exp]["dataset_name"]
            + "\n------------------------\n"
        )

        for split in splits:
            params_exp = copy.deepcopy(params[exp])
            params_exp["split"] = split
            params_exp["subset_info"]["subset"] = (
                params_exp["subset_info"]["subset"] + split
            )

            process_dataset(params_exp, exp=exp)
