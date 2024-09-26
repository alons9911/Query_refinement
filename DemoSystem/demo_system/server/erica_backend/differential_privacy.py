import itertools
import math
import random

import matplotlib.pyplot
import pandas as pd
from eeprivacy.mechanisms import LaplaceMechanism
from eeprivacy.operations import (
    PrivateClampedMean,
    PrivateHistogram,
)

import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages

from more_itertools import powerset

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

# students

# CONDS = [
#     {
#         "field": "grade1",
#         "operator": ">=",
#         "value": "13"
#     },
#     {
#         "field": "grade2",
#         "operator": ">=",
#         "value": "13"
#     },
#     {
#         "field": "age",
#         "operator": "IN",
#         "value": "[\"15-16\",\"17-18\"]"
#     },
#     {
#         "field": "higherEdIntention",
#         "operator": "IN",
#         "value": "[\"yes\"]"
#     }
# ]
# TABLE_NAME = "students"
# CONSTRAINTS = [
#     {
#         "groups": [
#             {
#                 "field": "address",
#                 "value": "Rural"
#             }
#         ],
#         "operator": ">=",
#         "amount": "15"
#     },
#     {
#         "groups": [
#             {
#                 "field": "sex",
#                 "value": "F"
#             }
#         ],
#         "operator": ">=",
#         "amount": "30"
#     },
#     {
#         "groups": [
#             {
#                 "field": "*",
#                 "value": "*"
#             }
#         ],
#         "operator": "<=",
#         "amount": "100"
#     }
# ]


# Adults
#
# CONDS = [
#     {
#         "field": "education",
#         "operator": "IN",
#         "value": "[\"Masters\"]"
#     },
# {
#         "field": "age",
#         "operator": ">=",
#         "value": "39"
#     },
# {
#         "field": "age",
#         "operator": "<=",
#         "value": "43"
#     },
# ]
# TABLE_NAME = "adults"
# CONSTRAINTS = [
#     {
#         "groups": [
#             {
#                 "field": "gender",
#                 "value": "Female"
#             }
#         ],
#         "operator": ">=",
#         "amount": "20"
#     },
#     {
#         "groups": [
#             {
#                 "field": "*",
#                 "value": "*"
#             }
#         ],
#         "operator": "<=",
#         "amount": "55"
#     }
# ]


# Synthetic adult

CONDS = [
    {
        "field": "education",
        "operator": "IN",
        "value": "[\"MASTER\"]"
    },
    {
        "field": "age",
        "operator": "<",
        "value": "40"
    },
]
TABLE_NAME = "syn_adults"
CONSTRAINTS = [
    {
        "groups": [
            {
                "field": "gender",
                "value": "F"
            }
        ],
        "operator": ">=",
        "amount": "15"
    },
    {
        "groups": [
            {
                "field": "*",
                "value": "*"
            }
        ],
        "operator": "<=",
        "amount": "30"
    }
]

DICT_QUERY = {
    "conds": CONDS,
    "table_name": TABLE_NAME,
    "constraints": CONSTRAINTS
}

EPSILON = 0.8


def get_bins(dataset):
    if any([isinstance(item, int) or isinstance(item, float) for item in dataset]):
        # return np.linspace(start=min(dataset), stop=max(dataset), num=max(dataset) - min(dataset) - 1)
        return np.arange(min(dataset) - 0.5, max(dataset) + 1.5)
    else:
        bins = list(set(dataset))
        bins.sort()
        return bins


def calc_bucket_start(item, bucket_size):
    return item - (item % bucket_size)


def calc_bucket_end(item, bucket_size):
    return item + (bucket_size - (item % bucket_size)) - 1


def bucket_dataset(dataset, bucket_size):
    # for item in dataset:
    #     print(f'{item} ---> {calc_bucket_start(item, bucket_size)}-{calc_bucket_end(item, bucket_size)}')
    bucketized_dataset = [f'{calc_bucket_start(item, bucket_size)}-{calc_bucket_end(item, bucket_size)}' for item in
                          dataset]
    return bucketized_dataset


def normalize_data(all_dataset, orig_dataset, refined_dataset, with_bucketing=False, bucket_size=5):
    if any([isinstance(item, int) or isinstance(item, float) for item in all_dataset]):
        if not with_bucketing:
            bins = get_bins(all_dataset)
            return orig_dataset, refined_dataset, bins, bins
        else:
            # print('ALL')
            all_dataset = bucket_dataset(all_dataset, bucket_size)
            # print()
            # print('ORIG')
            orig_dataset = bucket_dataset(orig_dataset, bucket_size)
            # print()
            # print('REFINED')
            refined_dataset = bucket_dataset(refined_dataset, bucket_size)
    values = list(set(all_dataset))
    values.sort()
    values_mapping = {values[i]: i for i in range(len(values))}
    normalized_all_dataset = [values_mapping[item] for item in all_dataset]
    normalized_orig_dataset = [values_mapping[item] for item in orig_dataset]
    normalized_refined_dataset = [values_mapping[item] for item in refined_dataset]
    return normalized_orig_dataset, normalized_refined_dataset, get_bins(normalized_all_dataset), get_bins(all_dataset)


def calc_private_histogram(dataset, bins, epsilon):
    return [sum(bins[index] <= item < bins[index + 1] for item in dataset) +
            np.random.laplace(0, 1.0 / epsilon, 1)[0]
            for index in range(len(bins) - 1)]


def calc_histogram(dataset, bins):
    return [sum(bins[index] <= item < bins[index + 1] for item in dataset)
            for index in range(len(bins) - 1)]


def plot(all_dataset, original_dataset, refined_dataset, histogram_field, epsilon, refinement_index, figure,
         number_of_subplots, subplot_number):
    # bins = np.linspace(start=0, stop=100, num=30)
    with_bucketing = True if histogram_field == 'age' else False
    normalized_original_dataset, normalized_refined_dataset, numeric_bins, real_bins = \
        normalize_data(all_dataset, original_dataset, refined_dataset, with_bucketing=with_bucketing)

    # bins = get_bins(dataset)

    # private_histogram_op = PrivateHistogram(
    #     bins=numeric_bins,
    # )
    private_refined_histogram = [max(item, 0) for item in
                                 calc_private_histogram(normalized_refined_dataset, numeric_bins, epsilon)]
    refined_sum = sum(private_refined_histogram)
    # private_refined_histogram = [(item / refined_sum) * 100 if refined_sum != 0 else 0 for item in private_refined_histogram]
    #
    private_original_histogram = [max(item, 0) for item in
                                  calc_private_histogram(normalized_original_dataset, numeric_bins, epsilon)]
    original_sum = sum(private_original_histogram)
    # private_original_histogram = [(item / original_sum) * 100 if original_sum != 0 else 0 for item in private_original_histogram]

    subtraction_histogram = [abs(original_item - refined_item) for original_item, refined_item in
                             zip(private_original_histogram, private_refined_histogram)]

    ci = LaplaceMechanism.confidence_interval(
        epsilon=epsilon,
        sensitivity=1,
        confidence=0.95
    )
    # print(f"95% Confidence Interval (Exact): {ci}")

    # true_histogram = np.histogram(normalized_refined_dataset, bins=numeric_bins)

    bin_centers = [(numeric_bins[i] + numeric_bins[i + 1]) / 2 for i in range(len(numeric_bins) - 1)]
    bin_width = numeric_bins[1] - numeric_bins[0] if len(numeric_bins) > 1 else 1

    ax = figure.add_subplot(number_of_subplots, 1, subplot_number)

    ax.bar(
        bin_centers if isinstance(real_bins[0], float) else real_bins,
        private_refined_histogram,
        width=bin_width / 2,
        yerr=ci,
        color="r",
        label="Private Refined"
    )
    ax.bar(
        [center + (bin_width / 2) for center in bin_centers],
        private_original_histogram,
        width=bin_width / 2,
        color="b",
        label="Private Original"
    )
    # ax.bar(
    #     [center + 2 * (bin_width / 3) for center in bin_centers],
    #     subtraction_histogram,
    #     width=bin_width / 3,
    #     color="g",
    #     label="Private Subtraction"
    # )
    ax.set_ylim(ymin=0)

    if refinement_index > -1:
        plt.title(
            f"Field={histogram_field}, Epsilon={epsilon}")
    else:
        plt.title(
            f"Field={histogram_field}, Epsilon={epsilon}, Original Query")

    plt.xlabel(f"{histogram_field}")
    plt.ylabel("Count")
    plt.legend()


def get_query_results_on_list(dataset, query):
    from sqlite3 import connect
    if len(dataset) == 0:
        return pd.DataFrame([])
    conn = connect(':memory:')
    df = pd.DataFrame(dataset)
    df.to_sql(name=TABLE_NAME, con=conn)
    return pd.read_sql(query, conn)


def score(dataset, original_query, refined_query, fields, debug=True):
    result = 0

    orig_results = get_query_results_on_list(dataset, original_query)
    ref_results = get_query_results_on_list(dataset, refined_query)

    for f, values in fields.items():
        for value in values:
            original_histogram = sum(item[f] == value for item in orig_results.to_dict('records'))
            refined_histogram = sum(item[f] == value for item in ref_results.to_dict('records'))
            result += abs(original_histogram - refined_histogram)
    if debug:
        print(f'dataset size: {len(dataset)}, score: {result}')
    return result

def score_max(dataset, original_query, refined_query, fields, debug=True):
    result = 0

    orig_results = get_query_results_on_list(dataset, original_query)
    ref_results = get_query_results_on_list(dataset, refined_query)

    for f, values in fields.items():
        max_subtraction = 0
        for value in values:
            original_histogram = sum(item[f] == value for item in orig_results.to_dict('records'))
            refined_histogram = sum(item[f] == value for item in ref_results.to_dict('records'))
            max_subtraction = max(max_subtraction, abs(original_histogram - refined_histogram))
        result += max_subtraction
    if debug:
        print(f'dataset size: {len(dataset)}, score: {result}')
    return result

def powerset_with_max_size(dataset, max_size):
    # for size in range(max_size + 1):
    #    for comb in [list(c) for c in itertools.combinations(dataset, size)]:
    #        yield comb
    for comb in itertools.combinations(dataset, max_size):
        yield list(comb)


def exponential_mechanism(dataset, original_query, refined_query, epsilon, delta, fields_list):
    fields = {f: list(set(item[f] for item in dataset)) for f in fields_list}
    # population = list(powerset_with_max_size(dataset[:50], 15))
    population = []
    weights = []
    # print(len(population))
    max_score = 0
    k = 0
    for comb in powerset_with_max_size(dataset[:15], 10):
        k += 1
        if k % 100 == 0:
            print(k)
        dataset_score = score_max(comb, original_query, refined_query, fields, debug=False)
        if dataset_score > max_score:
            max_score = dataset_score
            print(f'new max score: {max_score}')

        population.append(comb)
        weights.append(pow(math.e, (epsilon / (2 * delta) * dataset_score)))

    choice = random.choices(population=population, weights=weights, k=1)[0]
    choice_score = score_max(choice, original_query, refined_query, fields, debug=True)
    print(f'Chosen dataset score: {choice_score}')

    orig_results = get_query_results_on_list(choice, original_query)
    ref_results = get_query_results_on_list(choice, refined_query)

    print('Original query results:')
    print(orig_results)
    print('Refined query results:')
    print(ref_results)
    print()
    print()

    return choice


def run_exponential_mechanism():
    str_query = build_query(CONDS, TABLE_NAME)
    str_all_query = f"SELECT * \nFROM '{TABLE_NAME}';"
    res = requests.post('http://127.0.0.1:5000/run_query', json=DICT_QUERY).json()
    refinement = [r['query'] for r in res['refinements']][0]

    fields_for_histogram = [cond['field'] for cond in CONDS] + [group['field'] for cons in CONSTRAINTS for group in
                                                                cons['groups'] if group['field'] != '*']

    all_results = get_query_results(str_all_query)

    chosen_dataset = exponential_mechanism(all_results, str_query, refinement, EPSILON, 90, fields_for_histogram)
    print(pd.DataFrame(chosen_dataset))

    figs = [plt.figure(figsize=[25, 15 * len(fields_for_histogram)])]

    plt.figtext(0.5, 0.9, f"Original Query:\n {str_query}\n\n"
                          f"Refinement:\n {refinement}\n\n\n", wrap=True, ha="center", fontsize=30, style="italic",
                bbox={'facecolor': 'grey', 'alpha': 0.3, 'pad': 5})
    for m in range(len(fields_for_histogram)):
        field = fields_for_histogram[m]
        all_values = [r[field] for r in get_query_results_on_list(chosen_dataset, str_all_query).to_dict('records')]
        refined_values = [r[field] for r in get_query_results_on_list(chosen_dataset, refinement).to_dict('records')]
        original_values = [r[field] for r in get_query_results_on_list(chosen_dataset, str_query).to_dict('records')]
        plot(all_values, original_values, refined_values, histogram_field=field, epsilon=EPSILON,
             refinement_index=1,
             figure=figs[0], number_of_subplots=len(fields_for_histogram), subplot_number=m + 1)
    with PdfPages('output.pdf') as pdf:
        for fig in figs:
            pdf.savefig(fig, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    run_exponential_mechanism()
    raise Exception()

    str_query = build_query(CONDS, TABLE_NAME)
    str_all_query = f"SELECT * \nFROM '{TABLE_NAME}';"
    res = requests.post('http://127.0.0.1:5000/run_query', json=DICT_QUERY).json()
    refinements = [r['query'] for r in res['refinements']]

    figs = []
    fields_for_histogram = [cond['field'] for cond in CONDS] + [group['field'] for cons in CONSTRAINTS for group in
                                                                cons['groups'] if group['field'] != '*']

    # ORIGINAL QUERY
    # fig = plt.figure(figsize=[25, 15 * len(fields_for_histogram)])
    # figs.append(fig)
    original_results = get_query_results(str_query)
    all_results = get_query_results(str_all_query)

    exponential_mechanism(all_results, str_query, refinements[0], 0.5, 2, ['age', 'gender', 'education'])
    # plt.figtext(0.5, 0.01, f"The original query is:\n{str_query}", ha="center", fontsize=30)
    # for j in range(len(fields_for_histogram)):
    #     field = fields_for_histogram[j]
    #     values = [res[field] for res in results]
    #     plot(values, histogram_field=field, epsilon=EPSILON, refinement_index=-1,
    #          figure=fig, number_of_subplots=len(fields_for_histogram), subplot_number=j + 1)

    print(fields_for_histogram)
    for i in range(len(refinements)):
        fig = plt.figure(figsize=[25, 15 * len(fields_for_histogram)])
        figs.append(fig)

        refinement = refinements[i]
        refined_results = get_query_results(refinement)
        plt.figtext(0.5, 0.9, f"Original Query:\n {str_query}\n\n"
                              f"Refinement:\n {refinement}\n\n\n", wrap=True, ha="center", fontsize=30, style="italic",
                    bbox={'facecolor': 'grey', 'alpha': 0.3, 'pad': 5})
        for j in range(len(fields_for_histogram)):
            field = fields_for_histogram[j]
            all_values = [res[field] for res in all_results]
            refined_values = [res[field] for res in refined_results]
            original_values = [res[field] for res in original_results]
            plot(all_values, original_values, refined_values, histogram_field=field, epsilon=EPSILON,
                 refinement_index=i,
                 figure=fig, number_of_subplots=len(fields_for_histogram), subplot_number=j + 1)
    with PdfPages('output.pdf') as pdf:
        for fig in figs:
            pdf.savefig(fig, bbox_inches='tight')
    plt.show()
