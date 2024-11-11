import torch
import numpy as np
import pandas as pd
from preprocess import MembershipDataset
from infer import infer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def likelihood_ratio_attack_label_aware(
    train_scores: pd.DataFrame, test_scores: pd.DataFrame,
) -> pd.DataFrame:
    """Perform the likelihood ratio attack.

    Args:
        train_scores (pd.DataFrame): The scores from the target model.
        test_scores (pd.DataFrame): The scores from the target model on the test dataset.

    Returns:
        pd.DataFrame: The results of the attack.
    """
    # add a new column "gt_score" to the train_scores DataFrame, which is a copy of "score_class_{label}" of each row
    train_scores["gt_score"] = train_scores.apply(lambda x: x[f"score_class_{int(x['label'])}"], axis=1)

    # do same for test_scores
    test_scores["gt_score"] = test_scores.apply(lambda x: x[f"score_class_{int(x['label'])}"], axis=1)

    # group train scores by label and membership
    group = train_scores.groupby(["label", "membership"])

    # calculate the mean and variance of the "gt_score" column for each group
    group_mean = group["gt_score"].mean()
    group_var = group["gt_score"].var()

    # given a test sample, first get the label. Using this label find the group in train_scores.
    # Then calculate the likelihoods of test sample belonging to membership group 0 distribution and membership group 1 distribution
    # Then calculate the likelihood ratio of the two likelihoods
    # we do not know membership of test sample

    def likelihood_ratio(row):
        label = row["label"]
        likelihoods = []
        for membership_pre in [1, 0]:
            mean = group_mean.loc[label, membership_pre]
            var = group_var.loc[label, membership_pre]
            score = row["gt_score"]
            likelihood = np.exp(-0.5 * ((score - mean) ** 2) / var) / np.sqrt(2 * np.pi * var)
            likelihoods.append(likelihood)
        return likelihoods[0] / likelihoods[1]

    test_scores["score"] = test_scores.apply(likelihood_ratio, axis=1)
    return test_scores

def likelihood_ratio_attack_label_unaware(
    train_scores: pd.DataFrame, test_scores: pd.DataFrame,
) -> pd.DataFrame:
    """Perform the likelihood ratio attack.

    Args:
        train_scores (pd.DataFrame): The scores from the target model.
        test_scores (pd.DataFrame): The scores from the target model on the test dataset.

    Returns:
        pd.DataFrame: The results of the attack.
    """
    # add a new column "gt_score" to the train_scores DataFrame, which is a copy of "score_class_{label}" of each row
    train_scores["gt_score"] = train_scores.apply(lambda x: x[f"score_class_{int(x['label'])}"], axis=1)

    # do same for test_scores
    test_scores["gt_score"] = test_scores.apply(lambda x: x[f"score_class_{int(x['label'])}"], axis=1)

    # group train scores by label and membership
    group = train_scores.groupby(["membership"])

    # calculate the mean and variance of the "gt_score" column for each group
    group_mean = group["gt_score"].mean()
    group_var = group["gt_score"].var()

    # given a test sample, first get the label. Using this label find the group in train_scores.
    # Then calculate the likelihoods of test sample belonging to membership group 0 distribution and membership group 1 distribution
    # Then calculate the likelihood ratio of the two likelihoods
    # we do not know membership of test sample

    def likelihood_ratio(row):
        likelihoods = []
        for membership_pre in [1, 0]:
            mean = group_mean.loc[membership_pre]
            var = group_var.loc[membership_pre]
            score = row["gt_score"]
            likelihood = np.exp(-0.5 * ((score - mean) ** 2) / var) / np.sqrt(2 * np.pi * var)
            likelihoods.append(likelihood)
        return likelihoods[0] / likelihoods[1]

    test_scores["score"] = test_scores.apply(likelihood_ratio, axis=1)
    return test_scores

if __name__ == "__main__":
    train_scores = pd.read_csv("train_scores.csv")
    # print unique values in "label" column
    print(train_scores["label"].unique())
    test_scores = pd.read_csv("test_scores.csv")
    # print unique values in "label" column
    print(test_scores["label"].unique())
    results = likelihood_ratio_attack_label_unaware(train_scores, test_scores)
    results.to_csv("lr_results.csv", index=False)