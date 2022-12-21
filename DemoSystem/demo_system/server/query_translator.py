import json
from typing import List, Dict, Any
import copy

DICT_QUERY = {
    "categorical_attributes": {
        "c-charge-degree": ["F", "M", "O"]
    },
    "selection_numeric_attributes": {
        "juv-fel-count": [">=", 4, 1],
        "decile-score": [">=", 8, 1]
    },
    "selection_categorical_attributes": {
        "c-charge-degree": ["O"]
    }
}

DICT_CONSTRAINTS = {"all_sensitive_attributes": ["race", "sex"],
                    "number_fairness_constraints": 2,
                    "fairness_constraints": [
                        {"sensitive_attributes": {"race": "African-American"}, "symbol": ">=", "number": 30},
                        {"sensitive_attributes": {"sex": "Male"}, "symbol": ">=", "number": 45}
                    ]}

SQL_TEMPLATE = "SELECT * FROM '{table_name}' AS {table_char}\n" + \
               "WHERE {conds};"


def get_fields_from_dict_query(dict_query: Dict):
    numeric_conds = dict_query['selection_numeric_attributes']
    categorical_conds = dict_query['selection_categorical_attributes']

    selection_categorical_attributes = [field for field in categorical_conds.keys()]
    selection_numeric_attributes = [field for field in numeric_conds.keys()]

    fields = selection_categorical_attributes + selection_numeric_attributes

    return selection_numeric_attributes, selection_categorical_attributes, fields


def set_fields_to_dict_query(fields: List, dict_query: Dict):
    dict_query = copy.deepcopy(dict_query)
    numeric_conds = dict_query['selection_numeric_attributes']
    categorical_conds = dict_query['selection_categorical_attributes']

    for field in fields:
        if field['operator'] in ['in', 'IN', 'In']:
            categorical_conds[field['field'].replace("_", "-")] = json.loads(field['value'])
        else:
            numeric_conds[field['field'].replace("_", "-")] = [field['operator'], float(field['value']), 1]
    return dict_query


def set_constraints_to_dict_template(constraints: List, template: Dict):
    template = copy.deepcopy(template)

    template['number_fairness_constraints'] = len(constraints)
    all_sensitive_attributes: list = template['all_sensitive_attributes']
    fairness_constraints: list = template['fairness_constraints']

    for con in constraints:
        all_sensitive_attributes.append(con['field'])
        fairness_constraint = {"sensitive_attributes": {con['field']: con['value']},
                               "symbol": con['operator'],
                               "number": int(con['amount'])}
        fairness_constraints.append(fairness_constraint)
    return template


def translate_dict_query(table_name: str, dict_query: Dict):
    table_char = table_name[0]

    selection_numeric_attributes, selection_categorical_attributes, _ = get_fields_from_dict_query(dict_query)
    numeric_conds = dict_query['selection_numeric_attributes']
    categorical_conds = dict_query['selection_categorical_attributes']

    conds = []

    for attr in selection_numeric_attributes:
        field = attr.replace('-', '_')
        operator, value = numeric_conds[attr][0], numeric_conds[attr][1]
        conds.append(f"{table_char}.{field} {operator} {value}")
    for attr in selection_categorical_attributes:
        field = attr.replace('-', '_')
        values = '(' + ','.join([f"'{val}'" for val in categorical_conds[attr]]) + ')'
        conds.append(f"{table_char}.{field} IN {values}")

    return SQL_TEMPLATE.format(table_name=table_name, table_char=table_char, conds=' AND '.join(conds))


def translate_minimal_refinement(minimal_refinement: List[int], dict_query: Dict[Any, None], table_name):
    selection_numeric_attributes, selection_categorical_attributes, _ = get_fields_from_dict_query(dict_query)

    # assert len(minimal_refinement) == len(selection_categorical_attributes) + len(selection_numeric_attributes)

    index = 0
    for attr in selection_numeric_attributes:
        dict_query['selection_numeric_attributes'][attr][1] = minimal_refinement[index]
        index += 1

    for attr in selection_categorical_attributes:
        selected_options = dict_query['selection_categorical_attributes'][attr]
        all_options = dict_query['categorical_attributes'][attr]
        missing_options = [option for option in all_options if option not in selected_options]

        additional_options = []
        for option in missing_options:
            if minimal_refinement[index] == 1:
                additional_options.append(option)
            index += 1
        dict_query['selection_categorical_attributes'][attr] = selected_options + additional_options
    return translate_dict_query(table_name, dict_query)


def translate_minimal_refinements(minimal_refinements: List[List[int]], dict_query: Dict[Any, None], table_name):
    return [translate_minimal_refinement(min_ref, copy.deepcopy(dict_query), table_name)
            for min_ref in minimal_refinements]


def build_query(conds, table_name):
    table_char = table_name[0]
    for c in conds:
        if c['value'] and c['value'][0] == '[':
            c['value'] = '(' + c['value'][1:-1] + ')'
    conds = [f"{table_char}.{c['field']} {c['operator']} {c['value']}" for c in conds]
    return SQL_TEMPLATE.format(table_name=table_name, table_char=table_char, conds=' AND '.join(conds))

if __name__ == "__main__":
    print(translate_dict_query("compas-scores", DICT_QUERY))
    print()
    res = translate_minimal_refinements(
        [[3.0, 8, 1.0, 1.0], [1.0, 8, 0.0, 1.0], [0.0, 8, 0.0, 0.0], [3.0, 6, 1.0, 0.0], [2.0, 8, 1.0, 0.0]],
        DICT_QUERY, "compas-scores")
    for r in res:
        print(r)
        print()
