import numpy as np
import pandas as pd
import os
import pickle
import pandas

from GrooveEvaluator import evaluator  # import your version of evaluator!!


def subset_to_midi(subset_tag, subsets_tags, subsets, path):
    subset_path = os.path.join(path, subset_tag)
    for subsetix, tag in enumerate(subsets_tags):
        # Reinitialize metadata
        tag_path = os.path.join(subset_path, tag)
        os.makedirs(tag_path, exist_ok=True)

        for ix, _hvo in enumerate(subsets[subsetix]):
            # add metadata
            if ix == 0:
                metadata = {ix: _hvo.metadata.__dict__}
            else:
                metadata.update({ix: _hvo.metadata.__dict__})

            # export midi
            _hvo.save_hvo_to_midi(os.path.join(tag_path, f"{ix}.mid"))

        metadata_for_samples = pandas.DataFrame(metadata).transpose()
        metadata_for_samples.to_csv(os.path.join(tag_path, "metadata.csv"))

def evaluator_to_midi(evaluator_path, evaluator_name, export_path):
    evaluator_loaded = pickle.load(open(os.path.join(evaluator_path, evaluator_name), "rb"))

    # extract and export subsets for gt
    gt_subsets = evaluator_loaded._gt_subsets
    gt_subsets_tags = evaluator_loaded._gt_tags
    gt_subset_tag = "gt"
    gt_path = os.path.join(export_path, evaluator_name.split(".Eval")[0])
    subset_to_midi(gt_subset_tag, gt_subsets_tags, gt_subsets, gt_path)

    # extract and export subsets for predictions
    prediction_subsets = evaluator_loaded._prediction_subsets
    prediction_subsets_tags = evaluator_loaded._prediction_tags
    prediction_subset_tag = "prediction"
    prediction_path = os.path.join(export_path, evaluator_name.split(".Eval")[0])
    subset_to_midi(prediction_subset_tag, prediction_subsets_tags, prediction_subsets, prediction_path)

if __name__ == '__main__':
    evaluator_path = "post_processing_scripts/evaluators_monotonic_groove_transformer_v1"
    export_path = os.path.join(evaluator_path, "exported_to_midi")

    evaluator_names = [
        "validation_set_evaluator_run_hopeful-gorge-252_Epoch_90.Eval",
        "validation_set_evaluator_run_misunderstood-bush-246_Epoch_26.Eval",
        "validation_set_evaluator_run_rosy-durian-248_Epoch_26.Eval",
        "validation_set_evaluator_run_solar-shadow-247_Epoch_41.Eval"]

    for evaluator_name in evaluator_names:
        evaluator_to_midi(evaluator_path, evaluator_name, export_path)

