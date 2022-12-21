import json

from flask import Flask, request

from DemoSystem.demo_system.minimal_refinement import FindMinimalRefinement
from db_connector import get_query_results
from query_translator import translate_minimal_refinements, build_query, \
    set_fields_to_dict_query, set_constraints_to_dict_template, translate_dict_query

app = Flask(__name__)

QUERY_TEMPLATE = {
    "categorical_attributes": {
        "c-charge-degree": ["F", "M", "O"]
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


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


data_file = r"dbs/compas-scores.csv"


@app.post("/build_query")
def build():
    conds = request.json['conds']
    table_name = request.json['table_name']
    query = build_query(conds, table_name)
    results = get_query_results(query)
    return json.dumps({'query': query, 'results': results})


def build_results(query_results, original_results):
    for result in query_results:
        result['new_result'] = result not in original_results
    return query_results

@app.post("/run_query")
def run_query():
    conds = request.json['conds']
    table_name = request.json['table_name']
    constraints = request.json['constraints']

    print("\n\n")
    print("--- conditions ---")
    print(conds)
    print("--- constraints ---")
    print(constraints)
    print("--- table name ---")
    print(table_name)
    print("\n\n")

    q = set_fields_to_dict_query(conds, QUERY_TEMPLATE)
    c = set_constraints_to_dict_template(constraints, CONSTRAINTS_TEMPLATE)
    minimal_refinements, _, assign_to_provenance_num, _, _ = FindMinimalRefinement(data_file, q, c)
    queries = translate_minimal_refinements(minimal_refinements, q, table_name)
    print(str(queries))

    ####
    original_query_str = translate_dict_query(table_name, q)
    original_results = get_query_results(original_query_str)

    queries_with_results = [{'query': query, 'results': get_query_results(query)} for query in queries]

    def jacob_distance(query):
        original_results_strs = [json.dumps(r) for r in original_results]
        query_results_strs = [json.dumps(r) for r in query['results']]
        union = set(original_results_strs).union(query_results_strs)
        intersection = set(original_results_strs).intersection(query_results_strs)
        if len(union) == 0:
            return 9999999
        return len(intersection) / len(union)

    queries_with_results.sort(key=jacob_distance, reverse=True)
    res = [{'query': query['query'], 'distance_to_original': jacob_distance(query), 'results': build_results(query['results'], original_results)} for query in queries_with_results]

    ###

    print("----- MINIMAL REFINEMENTS -----")
    print(str(res))
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
