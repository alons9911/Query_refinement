from flask import Flask, request
import json

from DemoSystem.demo_system.minimal_refinement import FindMinimalRefinement
from DemoSystem.demo_system.query_translator import translate_minimal_refinements, build_query, \
    set_fields_to_dict_query, set_constraints_to_dict_template

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
    return build_query(conds, table_name)


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
    res = translate_minimal_refinements(minimal_refinements, q, table_name)
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
