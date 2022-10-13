import pandas as pd
import json
import time



def TestRefinement(data_file, query_file, constraint_file, relaxation):
    data = pd.read_csv(data_file)
    with open(query_file) as f:
        query_info = json.load(f)

    selection_numeric_attributes = query_info['selection_numeric_attributes']
    selection_categorical_attributes = query_info['selection_categorical_attributes']
    numeric_attributes = list(selection_numeric_attributes.keys())
    categorical_attributes = query_info['categorical_attributes']
    selected_attributes = numeric_attributes + [x for x in categorical_attributes]
    print("selected_attributes", selected_attributes)

    with open(constraint_file) as f:
        constraint_info = json.load(f)

    sensitive_attributes = constraint_info['all_sensitive_attributes']
    fairness_constraints = constraint_info['fairness_constraints']

    PVT_head = numeric_attributes.copy()
    for att, domain in categorical_attributes.items():
        for value in domain:
            if value in selection_categorical_attributes[att]:
                continue
            else:
                PVT_head.append(att + "_" + value)

    relaxation_dict = dict(zip(PVT_head, relaxation))
    print("relaxation_dict: {}".format(relaxation_dict))

    pd.set_option('display.float_format', '{:.2f}'.format)

    def select(row):
        for att in selection_numeric_attributes:
            if pd.isnull(row[att]):
                return 0
            if eval(str(row[att]) + selection_numeric_attributes[att][0] + str(relaxation_dict[att])):
                continue
            else:
                return 0
        for att in selection_categorical_attributes:
            if pd.isnull(row[att]):
                return 0
            if row[att] in selection_categorical_attributes[att]:
                continue
            else:
                if relaxation_dict[att+'_'+row[att]] == 0:
                    return 0
                else:
                    continue
        return 1

    data['satisfy_selection'] = data[selected_attributes].apply(select, axis=1)
    data_selected = data[data['satisfy_selection'] == 1]
    print("num data_selected: {}".format(len(data_selected)))
    # whether satisfy fairness constraint
    for fc in fairness_constraints:
        sensitive_attributes = fc['sensitive_attributes']
        df1 = data_selected[list(sensitive_attributes.keys())]
        df2 = pd.DataFrame([sensitive_attributes])
        data_selected_satisfying_fairness_constraint = df1.merge(df2)
        num = len(data_selected_satisfying_fairness_constraint)
        print("num of {} = {}".format(fc, num))
        if not eval(str(num) + fc['symbol'] + str(fc['number'])):
            return False
    return True





data_file = r"../InputData/Pipelines/healthcare/incomeK/before_selection_incomeK.csv"
query_file = r"../InputData/Pipelines/healthcare/incomeK/relaxation/query4.json"
constraint_file = r"../InputData/Pipelines/healthcare/incomeK/relaxation/constraint2.json"


relaxation = [300, 4, 8, 0, 0]
re = TestRefinement(data_file, query_file, constraint_file, relaxation)
print(re)

