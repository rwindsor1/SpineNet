import numpy as np


def format_gt_scores(score):
    if score == "":
        return None
    for key in score:
        score[key] = np.array(list(reversed(score[key])))
    formatted_gt_scores = {}
    formatted_gt_scores["Pfirrmann"] = label_check(score["Pfirrmann"]) - 1
    formatted_gt_scores["Narrowing"] = label_check(score["Narrowing"])
    formatted_gt_scores["UpperEndplateDefect"] = label_binarize(
        score["UpperEndplateDefect"]
    )
    formatted_gt_scores["LowerEndplateDefect"] = label_binarize(
        score["LowerEndplateDefect"]
    )
    formatted_gt_scores["UpperMarrow"] = label_check_marrow(
        score["UpperModic1"],
        score["UpperModic2"],
        score["UpperModic3"],
        score["UpperModicM"],
    )
    formatted_gt_scores["LowerMarrow"] = label_check_marrow(
        score["LowerModic1"],
        score["LowerModic2"],
        score["LowerModic3"],
        score["LowerModicM"],
    )
    formatted_gt_scores["Spondylolisthesis"] = label_check_3(score["Spondylolisthesis"])
    formatted_gt_scores["CentralCanalStenosis"] = label_check(
        score["CentralCanalStenosis"]
    )
    formatted_gt_scores["ForaminalStenosisLeft"] = label_binarize(
        score["ForaminalStenosisLeft"]
    )
    formatted_gt_scores["ForaminalStenosisRight"] = label_binarize(
        score["ForaminalStenosisRight"]
    )
    formatted_gt_scores["Herniation"] = label_binarize(score["Herniation"])
    for key in formatted_gt_scores:
        # return none if scores contain nan
        if (np.sum(formatted_gt_scores[key]) != np.sum(formatted_gt_scores[key])).any():
            return None
    return formatted_gt_scores


def label_binarize(label):
    label = label_check(label)
    label[label > 0] = 1

    return label


def label_check(label):
    label[np.isnan(label)] = -100
    return label


def label_check_marrow(m1, m2, m3, mm):
    m = m1 + m2 + m3 + mm
    m = label_binarize(m)
    return m


def label_check_3(label):
    label = label_check(label)
    label[label > 1] = 2
    return label
