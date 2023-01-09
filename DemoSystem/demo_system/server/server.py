import json

from flask import Flask, request

from DemoSystem.demo_system.minimal_refinement import FindMinimalRefinement
from db_connector import get_query_results
from query_translator import translate_minimal_refinements, build_query, \
    set_fields_to_dict_query, set_constraints_to_dict_template, translate_dict_query, get_fields_from_dict_query, \
    create_str_query_as_dict, translate_minimal_refinement_to_dict

app = Flask(__name__)

QUERY_TEMPLATE = {
    "categorical_attributes": {
        "higher": ["yes", "no"],

    },
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
# -------- Saving state -----------


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


data_file = r"dbs/students.csv"


def save_refinements(refinements):
    global last_refinements
    last_refinements = refinements


def save_form_fields(conds):
    global form_fields
    form_fields = conds


@app.get("/load_form_fields")
def load_form_fields():
    return form_fields


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
def get_db_preview():
    table_name = request.json['table_name']
    results = get_query_results(f"SELECT * FROM '{table_name}'")
    return results


@app.post("/build_query")
def build():
    conds = request.json['conds']
    table_name = request.json['table_name']
    query = build_query(conds, table_name)
    results = get_query_results(query)
    query_dict = set_fields_to_dict_query(conds, QUERY_TEMPLATE)
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
    unlikely_fields.reverse()
    def inner(query):
        score = 0
        for i in range(len(unlikely_fields)):
            field = unlikely_fields[i].replace('_', '-')
            if field in [f.replace('_', '-') for f in numeric_fields]:
                if set(query['query_dict']['selection_numeric_attributes'][field]) == set(original_query_dict['selection_numeric_attributes'][field]):
                    score += 10 ^ i
            else:
                if set(query['query_dict']['selection_categorical_attributes'][field]) == set(original_query_dict['selection_categorical_attributes'][field]):
                    score += 10 ^ i
        return score
    return inner


def get_jaccard_similarity_func(original_results):
    def inner(query):
        original_results_strs = [json.dumps(r) for r in original_results]
        query_results_strs = [json.dumps(r) for r in query['results']]
        union = set(original_results_strs).union(query_results_strs)
        intersection = set(original_results_strs).intersection(query_results_strs)
        if len(union) == 0:
            return 9999999
        return len(intersection) / len(union)
    return inner


def get_sorting_key(primary_func, secondary_func):
    def inner(query):
        return primary_func(query), secondary_func(query)
    return inner

@app.post("/sort_refinements")
def sort_refinements():
    refinements = request.json.get('refinements')
    sorting_func = request.json.get('sorting_func', 'Jaccard Similarity')
    conds = request.json['conds']
    table_name = request.json['table_name']

    query_dict = set_fields_to_dict_query(conds, QUERY_TEMPLATE)

    queries_with_results = [{'query': query['query'],
                             'query_dict': query['query_dict'],
                             'str_query_as_dict': query['str_query_as_dict'],
                             'results': get_query_results(query['query'])} for query in refinements]

    original_query_str = translate_dict_query(table_name, query_dict)
    original_results = get_query_results(original_query_str)

    if sorting_func == 'Jaccard Similarity':
        queries_with_results.sort(
            key=get_jaccard_similarity_func(original_results),
            reverse=True)
    elif sorting_func == 'Unlikely Changed Fields':
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
    res = [{'query': query['query'],
            'query_dict': query['query_dict'],
            'str_query_as_dict': query['str_query_as_dict'],
            'jaccard_similarity': get_jaccard_similarity_func(original_results)(query),
            'results': build_results(query['results'], original_results)} for query in queries_with_results]
    return res


@app.post("/run_query")
def run_query():
    conds = request.json['conds']
    table_name = request.json['table_name']
    constraints = request.json['constraints']
    sorting_func = request.json.get('sorting_func', 'Jaccard Similarity')

    print("\n\n")
    print("--- conditions ---")
    print(conds)
    print("--- constraints ---")
    print(constraints)
    print("--- table name ---")
    print(table_name)
    print("\n\n")

    query_dict = set_fields_to_dict_query(conds, QUERY_TEMPLATE)
    c = set_constraints_to_dict_template(constraints, CONSTRAINTS_TEMPLATE)
    minimal_refinements, _, assign_to_provenance_num, _, _ = FindMinimalRefinement(data_file, query_dict, c)
    queries = translate_minimal_refinements(minimal_refinements, query_dict, table_name)
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

    res = [{'query': query['query'],
            'query_dict': query['query_dict'],
            'str_query_as_dict': query['str_query_as_dict'],
            'jaccard_similarity': get_jaccard_similarity_func(original_results)(query),
            'original_results': original_results,
            'results': build_results(query['results'], original_results)} for query in queries_with_results]

    ###

    print("----- MINIMAL REFINEMENTS -----")
    print(str(res))
    save_refinements(res)
    save_table(table_name)
    save_form_fields(conds)
    return res


# def run_query():
#     query = request.args.get('query')
#     constrains = request.args.get('constrains')
#     q = json.loads(query)
#     c = json.loads(constrains)
#     minimal_refinements, _, assign_to_provenance_num, _, _ = FindMinimalRefinement(data_file, q, c)
#     return str(translate_minimal_refinements(minimal_refinements, q, "compas-scores"))


if __name__ == "__main__":
    app.run("0.0.0.0", "5000")
