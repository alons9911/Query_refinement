from eeprivacy.mechanisms import LaplaceMechanism
from eeprivacy.operations import (
    PrivateClampedMean,
    PrivateHistogram,
)

import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats

import pandas as pd
import requests

from erica_backend.db_connector import get_query_results
from erica_backend.query_translator import build_query

np.random.seed(1234)  # Fix seed for deterministic documentation

mpl.style.use("seaborn-white")

MD = 28
LG = 36
plt.rcParams.update({
    "figure.figsize": [25, 50],
    "legend.fontsize": MD,
    "axes.labelsize": LG,
    "axes.titlesize": LG,
    "xtick.labelsize": LG,
    "ytick.labelsize": LG,
})

CONDS = [
    {
        "field": "grade1",
        "operator": ">=",
        "value": "13"
    },
    {
        "field": "grade2",
        "operator": ">=",
        "value": "13"
    },
    {
        "field": "age",
        "operator": "IN",
        "value": "[\"15-16\",\"17-18\"]"
    },
    {
        "field": "higherEdIntention",
        "operator": "IN",
        "value": "[\"yes\"]"
    }
]
TABLE_NAME = "students"
CONSTRAINTS = [
    {
        "groups": [
            {
                "field": "address",
                "value": "Rural"
            }
        ],
        "operator": ">=",
        "amount": "15"
    },
    {
        "groups": [
            {
                "field": "sex",
                "value": "F"
            }
        ],
        "operator": ">=",
        "amount": "30"
    },
    {
        "groups": [
            {
                "field": "*",
                "value": "*"
            }
        ],
        "operator": "<=",
        "amount": "100"
    }
]

DICT_QUERY = {
    "conds": CONDS,
    "table_name": TABLE_NAME,
    "constraints": CONSTRAINTS
}

EPSILON = 0.1


def get_bins(dataset):
    if any([isinstance(item, int) or isinstance(item, float) for item in dataset]):
        # return np.linspace(start=min(dataset), stop=max(dataset), num=max(dataset) - min(dataset) - 1)
        return np.arange(min(dataset) - 0.5, max(dataset) + 1.5)
    else:
        bins = list(set(dataset))
        bins.sort()
        return bins


def normalize_data(dataset):
    if any([isinstance(item, int) or isinstance(item, float) for item in dataset]):
        bins = get_bins(dataset)
        return dataset, bins, bins
    values = list(set(dataset))
    values.sort()
    values_mapping = {values[i]: i for i in range(len(values))}
    normalized_dataset = [values_mapping[item] for item in dataset]
    return normalized_dataset, get_bins(normalized_dataset), get_bins(dataset)


def plot(dataset, histogram_field, epsilon, refinement_index, figure, number_of_subplots, subplot_number):
    # bins = np.linspace(start=0, stop=100, num=30)
    normalized_dataset, numeric_bins, real_bins = normalize_data(dataset)
    # bins = get_bins(dataset)

    private_histogram_op = PrivateHistogram(
        bins=numeric_bins,
    )

    private_histogram = private_histogram_op.execute(
        values=normalized_dataset,
        epsilon=epsilon
    )
    ci = LaplaceMechanism.confidence_interval(
        epsilon=epsilon,
        sensitivity=1,
        confidence=0.95
    )
    print(f"95% Confidence Interval (Exact): {ci}")

    true_histogram = np.histogram(normalized_dataset, bins=numeric_bins)

    bin_centers = [(numeric_bins[i] + numeric_bins[i + 1]) / 2 for i in range(len(numeric_bins) - 1)]
    bin_width = numeric_bins[1] - numeric_bins[0] if len(numeric_bins) > 1 else 1

    ax = figure.add_subplot(number_of_subplots, 1, subplot_number)

    ax.bar(
        bin_centers if isinstance(real_bins[0], float) else real_bins,
        private_histogram,
        width=bin_width / 2,
        yerr=ci,
        color="r",
        label="Private Count"
    )
    ax.bar(
        [center + (bin_width / 2) for center in bin_centers],
        true_histogram[0],
        width=bin_width / 2,
        color="b",
        label="True Count"
    )

    plt.title(
        f"Field={histogram_field}, Epsilon={epsilon}, Ref No. {refinement_index}")
    plt.xlabel(f"{histogram_field}")
    plt.ylabel("Count")
    plt.legend()
    return fig


if __name__ == "__main__":

    str_query = build_query(CONDS, TABLE_NAME)
    res = requests.post('http://127.0.0.1:5000/run_query', json=DICT_QUERY).json()
    refinements = [r['query'] for r in res['refinements']]

    figs = []
    fields_for_histogram = [cond['field'] for cond in CONDS] + [group['field'] for cons in CONSTRAINTS for group in
                                                                cons['groups'] if group['field'] != '*']
    print(fields_for_histogram)
    for i in range(len(refinements)):
        fig = plt.figure(figsize=[25, 15 * len(fields_for_histogram)])
        figs.append(fig)

        refinement = refinements[i]
        results = get_query_results(refinement)
        for j in range(len(fields_for_histogram)):
            field = fields_for_histogram[j]
            values = [res[field] for res in results]
            plot(values, histogram_field=field, epsilon=EPSILON, refinement_index=i,
                 figure=fig, number_of_subplots=len(fields_for_histogram), subplot_number=j + 1)
    with PdfPages('output.pdf') as pdf:
        for fig in figs:
            pdf.savefig(fig, bbox_inches='tight')
    plt.show()
