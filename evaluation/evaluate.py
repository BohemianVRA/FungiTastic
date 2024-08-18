from typing import List

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

COLUMNS = ["observationID", "class_id"]
poisonous_lvl = pd.read_csv(
    "http://ptak.felk.cvut.cz/plants//DanishFungiDataset/poison_status_list.csv"
)
POISONOUS_SPECIES = poisonous_lvl[poisonous_lvl["poisonous"] == 1].class_id.unique()


def classification_error_with_unknown(
    merged_df, cost_unkwnown_misclassified=10, cost_misclassified_as_unknown=0.1
):
    num_misclassified_unknown = sum((merged_df.class_id_gt == -1) & (merged_df.class_id_pred != -1))
    num_misclassified_as_unknown = sum(
        (merged_df.class_id_gt != -1) & (merged_df.class_id_pred == -1)
    )
    num_misclassified_other = sum(
        (merged_df.class_id_gt != merged_df.class_id_pred)
        & (merged_df.class_id_pred != -1)
        & (merged_df.class_id_gt != -1)
    )
    return (
        num_misclassified_other
        + num_misclassified_unknown * cost_unkwnown_misclassified
        + num_misclassified_as_unknown * cost_misclassified_as_unknown
    ) / len(merged_df)


def classification_error(merged_df):
    return classification_error_with_unknown(
        merged_df, cost_misclassified_as_unknown=1, cost_unkwnown_misclassified=1
    )


def num_psc_decisions(merged_df):
    # Number of observations that were misclassified as edible, when in fact they are poisonous
    num_psc = sum(
        merged_df.class_id_gt.isin(POISONOUS_SPECIES)
        & ~merged_df.class_id_pred.isin(POISONOUS_SPECIES)
    )
    return num_psc


def num_esc_decisions(merged_df):
    # Number of observations that were misclassified as poisonus, when in fact they are edible
    num_esc = sum(
        ~merged_df.class_id_gt.isin(POISONOUS_SPECIES)
        & merged_df.class_id_pred.isin(POISONOUS_SPECIES)
    )
    return num_esc


def psc_esc_cost_score(merged_df, cost_psc=100, cost_esc=1):
    return (
        cost_psc * num_psc_decisions(merged_df) + cost_esc * num_esc_decisions(merged_df)
    ) / len(merged_df)


def evaluate_csv(test_annotation_file: str, user_submission_file: str) -> List[dict]:
    # load gt annotations
    gt_df = pd.read_csv(test_annotation_file, sep=",")
    for col in COLUMNS:
        assert col in gt_df, f"Test annotation file is missing column '{col}'."
    # keep only observation-based predictions
    gt_df = gt_df.drop_duplicates("observationID")

    # load user predictions
    try:
        is_tsv = user_submission_file.endswith(".tsv")
        user_pred_df = pd.read_csv(user_submission_file, sep="\t" if is_tsv else ",")
    except Exception:
        print("Could not read file submitted by the user.")
        raise ValueError("Could not read file submitted by the user.")

    # validate user predictions
    for col in COLUMNS:
        if col not in user_pred_df:
            print(f"File submitted by the user is missing column '{col}'.")
            raise ValueError(f"File submitted by the user is missing column '{col}'.")
    if len(gt_df) != len(user_pred_df):
        print(f"File submitted by the user should have {len(gt_df)} records.")
        raise ValueError(f"File submitted by the user should have {len(gt_df)} records.")
    missing_obs = gt_df.loc[
        ~gt_df["observationID"].isin(user_pred_df["observationID"]),
        "observationID",
    ]
    if len(missing_obs) > 0:
        if len(missing_obs) > 3:
            missing_obs_str = ", ".join(missing_obs.iloc[:3].astype(str)) + ", ..."
        else:
            missing_obs_str = ", ".join(missing_obs.astype(str))
        print(f"File submitted by the user is missing observations: {missing_obs_str}")
        raise ValueError(f"File submitted by the user is missing observations: {missing_obs_str}")

    # merge dataframes
    merged_df = pd.merge(
        gt_df,
        user_pred_df,
        how="outer",
        on="observationID",
        validate="one_to_one",
        suffixes=("_gt", "_pred"),
    )

    # evaluate accuracy_score and f1_score
    cls_error = classification_error(merged_df)
    cls_error_with_unknown = classification_error_with_unknown(merged_df)
    psc_esc_cost = psc_esc_cost_score(merged_df)

    result = [
        {
            "test_split": {
                "F1 Score": np.round(
                    f1_score(merged_df["class_id_gt"], merged_df["class_id_pred"], average="macro")
                    * 100,
                    2,
                ),
                "Track 1: Classification Error": np.round(cls_error, 4),
                "Track 2: Cost for Poisonousness Confusion": np.round(psc_esc_cost, 4),
                "Track 3: User-Focused Loss": np.round(cls_error + psc_esc_cost, 4),
                "Track 4: Classification Error with Special Cost for Unknown": np.round(
                    cls_error_with_unknown, 4
                ),
            }
        }
    ]

    print(f"Evaluated scores: {result[0]['test_split']}")

    return result


def evaluate(test_annotation_file, user_submission_file, phase_codename, **kwargs):
    """
    Evaluates the submission for a particular challenge phase and returns score
    Arguments:

        `test_annotations_file`: Path to test_annotation_file on the server
        `user_submission_file`: Path to file submitted by the user
        `phase_codename`: Phase to which submission is made

        `**kwargs`: keyword arguments that contains additional submission
        metadata that challenge hosts can use to send slack notification.
        You can access the submission metadata
        with kwargs['submission_metadata']

        Example: A sample submission metadata can be accessed like this:
        >>> print(kwargs['submission_metadata'])
        {
            'status': u'running',
            'when_made_public': None,
            'participant_team': 5,
            'input_file': 'https://abc.xyz/path/to/submission/file.json',
            'execution_time': u'123',
            'publication_url': u'ABC',
            'challenge_phase': 1,
            'created_by': u'ABC',
            'stdout_file': 'https://abc.xyz/path/to/stdout/file.json',
            'method_name': u'Test',
            'stderr_file': 'https://abc.xyz/path/to/stderr/file.json',
            'participant_team_name': u'Test Team',
            'project_url': u'http://foo.bar',
            'method_description': u'ABC',
            'is_public': False,
            'submission_result_file': 'https://abc.xyz/path/result/file.json',
            'id': 123,
            'submitted_at': u'2017-03-20T19:22:03.880652Z'
        }
    """
    print("Starting Evaluation.....")
    out = {}
    if phase_codename == "prediction-based":
        print("Evaluating for Prediction-based Phase")
        out["result"] = evaluate_csv(test_annotation_file, user_submission_file)

        # To display the results in the result file
        out["submission_result"] = out["result"][0]["test_split"]
        print("Completed evaluation")
    return out


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test-annotation-file",
        help="Path to test_annotation_file on the server.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--user-submission-file",
        help="Path to a file created by predict script.",
        type=str,
        required=True,
    )
    args = parser.parse_args()

    result = evaluate(
        test_annotation_file=args.test_annotation_file,
        user_submission_file=args.user_submission_file,
        phase_codename="prediction-based",
    )
    with open("scores.json", "w") as f:
        json.dump(result, f)