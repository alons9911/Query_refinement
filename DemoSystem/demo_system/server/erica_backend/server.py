import copy
import json
import math
import sys
from pathlib import Path

from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin

sys.path.append('../../../../Algorithm')

from Algorithm.ProvenanceSearchValues import FindMinimalRefinement, is_int

#from DemoSystem.demo_system.minimal_refinement import FindMinimalRefinement
from db_connector import get_query_results
from json_decoder import Decoder
from query_translator import translate_minimal_refinements, build_query, \
    set_fields_to_dict_query, set_constraints_to_dict_template, translate_dict_query, get_fields_from_dict_query, \
    create_str_query_as_dict, translate_minimal_refinement_to_dict

import pandas as pd

app = Flask(__name__)
CORS(app, origins='*')

data_file_prefix = '../../../dbs/'

QUERY_TEMPLATE = {
    "tables": [],
    "categorical_attributes": {},
    "selection_numeric_attributes": {
    },
    "selection_categorical_attributes": {
    }
}

CONSTRAINTS_TEMPLATE = {"all_sensitive_attributes": [],
                        "number_fairness_constraints": 0,
                        "fairness_constraints": []
                        }
# -------- Saving state -----------
last_refinements = []
form_fields = []
table = ''
selected_fields = []
original_str_query_as_dict = {}
constraints = []
# -------- Saving state -----------


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"



def save_refinements(refinements):
    global last_refinements
    last_refinements = refinements


def save_form_fields(conds):
    global form_fields
    form_fields = conds


@app.get("/load_form_fields")
def load_form_fields():
    return form_fields


def save_constraints(cons):
    global constraints
    constraints = cons


@app.get("/load_constraints")
def load_constraints():
    return constraints


@app.post("/save_selected_fields")
def save_selected_fields():
    global selected_fields
    fields = request.json.get('selected_fields', [])
    selected_fields = fields
    return ''

@app.get("/load_refinements_viewer_state")
def load_refinements_viewer_state():
    return {'refinements': last_refinements,
            'table': table,
            'selected_fields': selected_fields,
            'form_fields': form_fields,
            'original_str_query_as_dict': original_str_query_as_dict}


def save_table(table_name):
    global table
    table = table_name


@app.get("/load_table")
def load_table():
    return table

def save_original_str_query_as_dict(str_query_as_dict):
    global original_str_query_as_dict
    original_str_query_as_dict = str_query_as_dict

@app.get("/load_original_str_query_as_dict")
def load_original_query_dict():
    return original_str_query_as_dict

@app.get("/load_refinements")
def load_refinements():
    return last_refinements

@app.post("/get_db_preview")
@cross_origin(origins='*')
def get_db_preview():
    table_name = request.json['table_name']
    results = get_query_results(f"SELECT * FROM '{table_name}' LIMIT 100")
    response = jsonify(results)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return results


@app.post("/build_query")
def build():
    conds = request.json['conds']
    table_name = request.json['table_name']
    query = build_query(conds, table_name)
    results = get_query_results(query)
    query_dict = set_fields_to_dict_query(conds, table_name, QUERY_TEMPLATE)
    str_query_as_dict = create_str_query_as_dict(table_name, query_dict, query_dict)
    save_original_str_query_as_dict(str_query_as_dict)

    return json.dumps({'query': query,
                       'results': results,
                       'str_query_as_dict': str_query_as_dict,
                        })


def build_results(query_results, original_results):
    removed_from_original = [res for res in original_results if res not in query_results]
    for result in query_results:
        result['new_result'] = result not in original_results

    return {'query_results': query_results, 'removed_from_original': removed_from_original}


def unlikely_changed_fields_sorting_key(unlikely_fields, original_query_dict, numeric_fields, categorical_fields):
    def inner(query):
        score = 0
        for i in range(len(unlikely_fields)):
            field = unlikely_fields[i].replace('_', '-')
            if field in [f.replace('_', '-') for f in numeric_fields]:
                if set(query['query_dict']['selection_numeric_attributes'][field]) != set(original_query_dict['selection_numeric_attributes'][field]):
                    score -= (float(pow(10,i)) + abs(query['query_dict']['selection_numeric_attributes'][field][1] - original_query_dict['selection_numeric_attributes'][field][1]))
            else:
                selection_categorical_attributes = set(query['query_dict']['selection_categorical_attributes'][field])
                original_selection_categorical_attributes = set(original_query_dict['selection_categorical_attributes'][field])
                if selection_categorical_attributes != original_selection_categorical_attributes:
                    diff = selection_categorical_attributes.symmetric_difference(original_selection_categorical_attributes)
                    print(diff)
                    score -= (pow(10, i) + len(diff))
        print(f"score is: {score}")
        return score
    return inner



def group_cardinality_sorting_key(group):
    def inner(query):
        cardinality_satisfaction = query['cardinality_satisfaction']
        for card in cardinality_satisfaction:
            if card['group'] == group:
                return card['amount']
    return inner

def get_jaccard_similarity_func(original_results):
    def inner(query):
        original_results_strs = [json.dumps(r) for r in original_results]
        query_results_strs = [json.dumps(r) for r in query['results']]
        union = set(original_results_strs).union(query_results_strs)
        intersection = set(original_results_strs).intersection(query_results_strs)
        if len(union) == 0:
            return 9999999
        return round(len(intersection) / len(union), ndigits=2)
    return inner


def get_sorting_key(primary_func, secondary_func):
    def inner(query):
        return primary_func(query), secondary_func(query)
    return inner



def calculate_fairness_constraints_satisfaction(file, query_info, constraint_info):
    satisfaction_info = []
    fairness_constraints = constraint_info['fairness_constraints']
    categorical_attributes = query_info['categorical_attributes']
    selection_numeric_attributes = query_info['selection_numeric_attributes']
    selection_categorical_attributes = query_info['selection_categorical_attributes']

    data = pd.read_csv(file, index_col=False)
    # if all values of a categorical attribute are integer, change that column type to string
    for att in categorical_attributes:
        if data[att].apply(is_int).all():
            data[att] = data[att].apply(lambda x: str(int(x)) if isinstance(x, float) else str(x))
    pe_dataframe = copy.deepcopy(data)
    # get data selected
    for att in selection_numeric_attributes:
        if selection_numeric_attributes[att][0] == '>':
            pe_dataframe = pe_dataframe[pe_dataframe[att] > selection_numeric_attributes[att][1]]
        elif selection_numeric_attributes[att][0] == ">=":
            pe_dataframe = pe_dataframe[pe_dataframe[att] >= selection_numeric_attributes[att][1]]
        elif selection_numeric_attributes[att][0] == "<":
            pe_dataframe = pe_dataframe[pe_dataframe[att] < selection_numeric_attributes[att][1]]
        else:
            pe_dataframe = pe_dataframe[pe_dataframe[att] <= selection_numeric_attributes[att][1]]
    for att in selection_categorical_attributes:
        pe_dataframe = pe_dataframe[pe_dataframe[att].isin(selection_categorical_attributes[att])]

    for fc in fairness_constraints:
        df = copy.deepcopy(pe_dataframe)
        sensitive_attributes = fc['sensitive_attributes']
        if 'all' in sensitive_attributes.keys() and len(sensitive_attributes) == 1:
            satisfaction_info.append({'group': 'Result Size',
                                      'amount': len(df)})
        else:
            for att in sensitive_attributes:
                df = df[df[att] == sensitive_attributes[att]]
            num = len(df)
            group = f"( {'AND '.join([f'{field} = {value}' for field, value in sensitive_attributes.items()])} )"
            satisfaction_info.append({'group': group,
                                      'amount': num})
    return satisfaction_info

# def calculate_fairness_constraints_satisfaction(file, query_info, constraint_info):
#     data = pd.read_csv(file, index_col=False)
#
#     satisfaction_info = []
#
#     numeric_attributes = []
#     categorical_attributes = {}
#     selection_numeric_attributes = {}
#     selection_categorical_attributes = {}
#     if 'selection_numeric_attributes' in query_info:
#         selection_numeric_attributes = query_info['selection_numeric_attributes']
#         numeric_attributes = list(selection_numeric_attributes.keys())
#     if 'selection_categorical_attributes' in query_info:
#         selection_categorical_attributes = query_info['selection_categorical_attributes']
#         categorical_attributes = query_info['categorical_attributes']
#     selected_attributes = numeric_attributes + [x for x in categorical_attributes]
#
#     fairness_constraints = constraint_info['fairness_constraints']
#
#     pd.set_option('display.float_format', '{:.2f}'.format)
#
#
#     # get data selected
#     def select(row):
#         for att in selection_numeric_attributes:
#             if pd.isnull(row[att]):
#                 return 0
#             if not eval(
#                     str(row[att]) + selection_numeric_attributes[att][0] + str(selection_numeric_attributes[att][1])):
#                 return 0
#         for att in selection_categorical_attributes:
#             if pd.isnull(row[att]):
#                 return 0
#             if row[att] not in selection_categorical_attributes[att]:
#                 return 0
#         return 1
#
#     data['satisfy_selection'] = data[selected_attributes].apply(select, axis=1)
#     data_selected = data[data['satisfy_selection'] == 1]
#     # whether satisfy fairness constraint
#     for fc in fairness_constraints:
#         sensitive_attributes = fc['sensitive_attributes']
#         if 'all' in sensitive_attributes.keys() and len(sensitive_attributes) == 1:
#             satisfaction_info.append({'group': 'Result Size',
#                                       'amount': len(data_selected)})
#         else:
#             df1 = data_selected[list(sensitive_attributes.keys())]
#             df2 = pd.DataFrame([sensitive_attributes])
#             data_selected_satisfying_fairness_constraint = df1.merge(df2)
#             num = len(data_selected_satisfying_fairness_constraint)
#
#             group = f"( { 'AND '.join([f'{field} = {value}' for field, value in sensitive_attributes.items()])} )"
#             satisfaction_info.append({'group': group,
#                                      'amount': num})
#
#     return satisfaction_info


@app.post("/sort_refinements")
def sort_refinements():
    refinements = request.json.get('refinements')
    sorting_func = request.json.get('sorting_func', 'Result Similarity')
    conds = request.json['conds']
    table_name = request.json['table_name']
    data_file = f'{data_file_prefix}{table_name}.csv'

    query_dict = set_fields_to_dict_query(conds, table_name, QUERY_TEMPLATE)

    queries_with_results = [{'query': query['query'],
                             'query_dict': query['query_dict'],
                             'cardinality_satisfaction': query['cardinality_satisfaction'],
                             'str_query_as_dict': query['str_query_as_dict'],
                             'results': get_query_results(query['query'])} for query in refinements]

    original_query_str = translate_dict_query(table_name, query_dict)
    original_results = get_query_results(original_query_str)

    if sorting_func == 'Change In Result':
        queries_with_results.sort(
            key=get_jaccard_similarity_func(original_results),
            reverse=True)
    elif sorting_func == 'Change In Selection Conditions':
        unlikely_changed_fields = request.json.get('unlikely_changed_fields', [])
        numeric_fields, categorical_fields, _ = get_fields_from_dict_query(query_dict)
        if unlikely_changed_fields:
            queries_with_results.sort(
                key=get_sorting_key(
                    unlikely_changed_fields_sorting_key(unlikely_changed_fields, query_dict, numeric_fields, categorical_fields),
                    get_jaccard_similarity_func(original_results)),
                reverse=True)
        else:
            queries_with_results.sort(
                key=get_jaccard_similarity_func(original_results),
                reverse=True)
    elif sorting_func == 'Change In Groups Cardinality':
        cardinality_group = request.json.get('cardinality_group')

        queries_with_results.sort(
            key=get_sorting_key(
                group_cardinality_sorting_key(cardinality_group),
                get_jaccard_similarity_func(original_results)),
            reverse=True)
    res = [{'query': query['query'],
            'query_dict': query['query_dict'],
            'str_query_as_dict': query['str_query_as_dict'],
            'jaccard_similarity': get_jaccard_similarity_func(original_results)(query),
            'cardinality_satisfaction': calculate_fairness_constraints_satisfaction(data_file, query['query_dict'], constraints),
            'results': build_results(query['results'], original_results)} for query in queries_with_results]
    return res


@app.post("/run_query")
def run_query():
    conds = request.json['conds']
    table_name = request.json['table_name']
    constraints_dict = request.json['constraints']
    sorting_func = request.json.get('sorting_func', 'Jaccard Similarity')


    print("\n\n")
    print("--- conditions ---")
    print(conds)
    print("--- constraints ---")
    print(constraints_dict)
    print("--- table name ---")
    print(table_name)
    print("\n\n")

    query_dict = set_fields_to_dict_query(conds, table_name, QUERY_TEMPLATE)
    cons = set_constraints_to_dict_template(constraints_dict, CONSTRAINTS_TEMPLATE)
    save_constraints(cons)

    separator = ','
    data_format = '.csv'
    time_limit = 60 * 60 * 5
    minimal_refinements, order_in_results, _, assign_to_provenance_num, _, _ = FindMinimalRefinement(data_file_prefix, separator,
                                                                                   query_dict, cons,
                                                                                   data_format, time_limit)
    queries = translate_minimal_refinements(minimal_refinements, query_dict, order_in_results, table_name)
    print(str(queries))

    ####
    original_query_str = translate_dict_query(table_name, query_dict)
    original_results = get_query_results(original_query_str)

    queries_with_results = [{'query': query['query_str'],
                             'query_dict': query['query_dict'],
                             'str_query_as_dict': query['str_query_as_dict'],
                             'results': get_query_results(query['query_str'])} for query in queries]

    queries_with_results.sort(
        key=get_jaccard_similarity_func(original_results),
        reverse=True)

    data_file = f'{data_file_prefix}{table_name}.csv'

    res = [{'query': query['query'],
            'query_dict': query['query_dict'],
            'str_query_as_dict': query['str_query_as_dict'],
            'jaccard_similarity': get_jaccard_similarity_func(original_results)(query),
            'original_results': original_results,
            'cardinality_satisfaction': calculate_fairness_constraints_satisfaction(data_file, query['query_dict'], cons),
            'results': build_results(query['results'], original_results)} for query in queries_with_results]

    ###

    print("----- MINIMAL REFINEMENTS -----")
    print(str(res))
    res = Decoder().decode(res)
    save_refinements(res)
    save_table(table_name)
    save_form_fields(conds)
    print(len(res))
    return {'refinements': res,
            'original_cardinality_satisfaction': calculate_fairness_constraints_satisfaction(data_file, query_dict, cons)
            }


# def run_query():
#     query = request.args.get('query')
#     constrains = request.args.get('constrains')
#     q = json.loads(query)
#     c = json.loads(constrains)
#     minimal_refinements, _, assign_to_provenance_num, _, _ = FindMinimalRefinement(data_file, q, c)
#     return str(translate_minimal_refinements(minimal_refinements, q, "compas-scores"))


if __name__ == "__main__":
    app.run("0.0.0.0", "5000")
