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

SQL_TEMPLATE = "SELECT * \nFROM '{table_name}' AS {table_char}\n" + \
               "WHERE {conds};"



def get_fields_from_dict_query(dict_query: Dict):
    numeric_conds = dict_query['selection_numeric_attributes']
    categorical_conds = dict_query['selection_categorical_attributes']

    selection_categorical_attributes = [field for field in categorical_conds.keys()]
    selection_numeric_attributes = [field for field in numeric_conds.keys()]

    fields = selection_categorical_attributes + selection_numeric_attributes

    return selection_numeric_attributes, selection_categorical_attributes, fields


def set_fields_to_dict_query(fields: List, table_name: str, dict_query: Dict):
    dict_query = copy.deepcopy(dict_query)
    dict_query['tables'].append(table_name)
    numeric_conds = dict_query['selection_numeric_attributes']
    categorical_conds = dict_query['selection_categorical_attributes']

    for field in fields:
        if field['operator'] in ['in', 'IN', 'In']:
            categorical_conds[field['field'].replace("_", "-")] = json.loads(field['value'].replace('(', '[').replace(')', ']'))
        else:
            numeric_conds[field['field'].replace("_", "-")] = [field['operator'], float(field['value']), 1]
    return dict_query


def set_constraints_to_dict_template(constraints: List, template: Dict):
    template = copy.deepcopy(template)

    template['number_fairness_constraints'] = len(constraints)
    all_sensitive_attributes: list = template['all_sensitive_attributes']
    fairness_constraints: list = template['fairness_constraints']

    for con in constraints:
        for group in con['groups']:
            if group['value'] == '*' and group['field'] == '*':
                group['field'] = 'all'
                group['value'] = 'yes'
        all_sensitive_attributes += [group['field'] for group in con['groups']]
        fairness_constraint = {"sensitive_attributes": {group['field']: group['value'] for group in con['groups']},
                               "symbol": con['operator'],
                               "number": int(con['amount'])}
        fairness_constraints.append(fairness_constraint)
    print(template)
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


def create_str_query_as_dict(table_name: str, dict_query: Dict, original_query: Dict):
    table_char = table_name[0]

    selection_numeric_attributes, selection_categorical_attributes, _ = get_fields_from_dict_query(dict_query)
    numeric_conds = dict_query['selection_numeric_attributes']
    categorical_conds = dict_query['selection_categorical_attributes']

    where_clauses = []

    for attr in selection_numeric_attributes:
        field = attr.replace('-', '_')
        operator, value = numeric_conds[attr][0], numeric_conds[attr][1]
        where_clauses.append({
            'clause': [f"{table_char}.{field}", f"{operator}", f"{value}"],
            'bold': not (attr in original_query['selection_numeric_attributes'] and
                         original_query['selection_numeric_attributes'][attr][0] == operator and
                         original_query['selection_numeric_attributes'][attr][1] == value)
        })
    for attr in selection_categorical_attributes:
        field = attr.replace('-', '_')
        categorical_conds[attr].sort()
        values = '(' + ','.join([f"'{val}'" for val in categorical_conds[attr]]) + ')'
        where_clauses.append({
            'clause': [f"{table_char}.{field}", "IN", f"{values}"],
            'bold': not (attr in original_query['selection_categorical_attributes'] and
                         sorted(original_query['selection_categorical_attributes'][attr]) == sorted(categorical_conds[attr]))
        })

    return {
        'select': "*",
        'from': f"'{table_name}' AS {table_char}",
        'where': where_clauses
    }


def translate_minimal_refinement_to_dict(minimal_refinement: List[int], dict_query: Dict[Any, None], order_in_results: List[str], table_name):
    selection_numeric_attributes, selection_categorical_attributes, _ = get_fields_from_dict_query(dict_query)

    # assert len(minimal_refinement) == len(selection_categorical_attributes) + len(selection_numeric_attributes)

    index = 0
    attrs = [attr[:attr.find('__')] if attr.find('__') != -1 else attr for attr in order_in_results]
    def remove_dups(items):
        seen = set()
        seen_add = seen.add
        return [item for item in items if not (item in seen or seen_add(item))]
    attrs = remove_dups(attrs)
    for attr in attrs:
        if attr in selection_numeric_attributes:
            dict_query['selection_numeric_attributes'][attr][1] = minimal_refinement[index]
            index += 1
        else:
            dict_query['selection_categorical_attributes'][attr] = []
            all_options = [op[op.find('__') + 2:] for op in order_in_results if op.startswith(attr)]
            #selected_options = dict_query['selection_categorical_attributes'][attr]
            #missing_options = [option for option in all_options if option not in selected_options]

            refinement_options = []
            for option in all_options:
                if minimal_refinement[index] == 1:
                    refinement_options.append(option)
                index += 1
            dict_query['selection_categorical_attributes'][attr] = refinement_options

    # for attr in selection_categorical_attributes:
    #     selected_options = dict_query['selection_categorical_attributes'][attr]
    #     all_options = dict_query['categorical_attributes'][attr]
    #     missing_options = [option for option in all_options if option not in selected_options]
    #
    #     for attr in selection_numeric_attributes:
    #         dict_query['selection_numeric_attributes'][attr][1] = minimal_refinement[index]
    #         index += 1
    #
    #     refinement_options = []
    #     for option in missing_options:
    #         if minimal_refinement[index] == 1:
    #             refinement_options.append(option)
    #         index += 1
    #     dict_query['selection_categorical_attributes'][attr] = selected_options + refinement_options
    return dict_query


def translate_minimal_refinement(minimal_refinement: List[int], dict_query: Dict[Any, None], order_in_results: List[str], table_name):
    return translate_dict_query(table_name,
                                translate_minimal_refinement_to_dict(minimal_refinement, dict_query, order_in_results, table_name))


def translate_minimal_refinements(minimal_refinements: List[List[int]], dict_query: Dict[Any, None], order_in_results: List[str], table_name):
    refinements = [{"query_str": translate_minimal_refinement(min_ref, copy.deepcopy(dict_query), order_in_results, table_name),
                    "query_dict": translate_minimal_refinement_to_dict(min_ref, copy.deepcopy(dict_query), order_in_results, table_name)}
                   for min_ref in minimal_refinements]
    for ref in refinements:
        ref["str_query_as_dict"] = create_str_query_as_dict(table_name, ref["query_dict"], dict_query)
    return refinements


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
