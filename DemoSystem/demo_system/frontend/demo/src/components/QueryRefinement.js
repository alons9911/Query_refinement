import React, {useState} from 'react';

function QueryRefinement(props) {
    const [formFields, setFormFields] = useState([
        {field: '', operator: '', value: ''},
    ])
    const [formConstraints, setFormConstraints] = useState([
        {groups: [{field: '', value: ''}, {field: '', value: ''}], operator: '', amount: ''},
    ])
    const [table, setTable] = useState('compas-scores');
    const [tableFields, setTableFields] = useState(["id", "sex", "juv_fel_count", "c_jail_out", "age", "age_cat", "c_arrest_date", "c_case_number",
        "c-charge-desc", "c-days-from-compas", "c-offense-date", "c_charge_degree", "c_jail_in",
        "compas_screening_date", "days_b_screening_arrest", "decile-score", "decile_score", "dob", "first", "is-recid",
        "is-violent-recid", "juv_fel_count", "juv_misd_count", "juv_other_count", "last", "name", "num-r-cases",
        "num-vr-cases", "priors_count", "r-case-number", "r-charge-degree", "r-charge-desc", "r-days-from-arrest",
        "r-jail-in", "r-jail-out", "r-offense-date", "race", "score-text", "screening-date", "type-of-assessment",
        "v-decile-score", "v-score-text", "v-screening-date", "v-type-of-assessment", "vr-case-number", "vr-charge-degree",
        "vr-charge-desc", "vr-offense-date"]);

    const [query, setQuery] = useState('');
    const [originalQueryResults, setOriginalQueryResults] = useState([]);

    const [refinements, setRefinements] = useState([]);
    const [err, setErr] = useState('');

    const optionalOperators = ['>', '>=', '=', '<', '<=', 'IN']


    return (
        <div className="query-table-container">
            <table className="query-table">
                <thead>
                <tr>{ThData()}</tr>
                </thead>
                <tbody>
                {tdData()}
                </tbody>
            </table>
        </div>
    )
}

export default QueryRefinement;