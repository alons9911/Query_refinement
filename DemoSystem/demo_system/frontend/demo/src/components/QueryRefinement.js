import React, {useEffect, useState} from 'react';
import QueryForm from './QueryForm';
import {BrowserRouter, Route, Routes, useLocation} from "react-router-dom";
import HeaderNavBar from "./HeaderNavBar";
import RefinementsViewer from "./RefinmentsViewer";


function QueryRefinement(props) {
    const [formFields, setFormFields] = useState([
        {field: '', operator: '', value: ''},
    ])
    useEffect(() => {
        setFormFields(JSON.parse(window.localStorage.getItem('formFields')));

    }, []);
    useEffect(() => {
        window.localStorage.setItem('formFields', JSON.stringify(formFields));
    }, [formFields]);

    const [formConstraints, setFormConstraints] = useState([
        {groups: [{field: '', value: ''}, {field: '', value: ''}], operator: '', amount: ''},
    ])
    useEffect(() => {
        console.log(window.localStorage.getItem('formConstraints'));
        setFormConstraints(JSON.parse(window.localStorage.getItem('formConstraints')));

    }, []);
    useEffect(() => {
        window.localStorage.setItem('formConstraints', JSON.stringify(formConstraints));
    }, [formFields]);

    const [table, setTable] = useState('students');
    useEffect(() => {
        setTable(window.localStorage.getItem('table'));

    }, []);
    useEffect(() => {
        window.localStorage.setItem('table', table);
    }, [table]);

    /*const [tableFields, setTableFields] = useState(["id", "sex", "juv_fel_count", "c_jail_out", "age", "age_cat", "c_arrest_date", "c_case_number",
        "c-charge-desc", "c-days-from-compas", "c-offense-date", "c_charge_degree", "c_jail_in",
        "compas_screening_date", "days_b_screening_arrest", "decile-score", "decile_score", "dob", "first", "is-recid",
        "is-violent-recid", "juv_fel_count", "juv_misd_count", "juv_other_count", "last", "name", "num-r-cases",
        "num-vr-cases", "priors_count", "r-case-number", "r-charge-degree", "r-charge-desc", "r-days-from-arrest",
        "r-jail-in", "r-jail-out", "r-offense-date", "race", "score-text", "screening-date", "type-of-assessment",
        "v-decile-score", "v-score-text", "v-screening-date", "v-type-of-assessment", "vr-case-number", "vr-charge-degree",
        "vr-charge-desc", "vr-offense-date"]);*/

    const [tableFields, setTableFields] = useState(["school", "sex", "age", "address", "famsize", "Pstatus", "Medu", "Fedu", "Mjob", "Fjob", "reason", "guardian", "traveltime", "studytime", "failures", "schoolsup", "famsup", "paid", "activities", "nursery", "higher", "internet", "romantic", "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences", "G1", "G2", "G3"]);
    useEffect(() => {
        setTableFields(JSON.parse(window.localStorage.getItem('tableFields')));

    }, []);
    useEffect(() => {
        window.localStorage.setItem('tableFields', JSON.stringify(tableFields));
    }, [tableFields]);

    const [query, setQuery] = useState({});
    useEffect(() => {
        setQuery(JSON.parse(window.localStorage.getItem('query')));

    }, []);
    useEffect(() => {
        window.localStorage.setItem('query', JSON.stringify(query));
    }, [query]);

    const [originalQueryResults, setOriginalQueryResults] = useState([]);
    useEffect(() => {
        setOriginalQueryResults(JSON.parse(window.localStorage.getItem('originalQueryResults')));
    }, []);
    useEffect(() => {
        window.localStorage.setItem('originalQueryResults', JSON.stringify(originalQueryResults));
    }, [originalQueryResults]);

    const [refinements, setRefinements] = useState([]);

    const [err, setErr] = useState('');
    useEffect(() => {
        setErr(JSON.parse(window.localStorage.getItem('err')));

    }, [err]);
    useEffect(() => {
        window.localStorage.setItem('err', JSON.stringify(err));
    }, []);


    const optionalOperators = ['>', '>=', '=', '<', '<=', 'IN']

    return (
        <div className="erica">
            <HeaderNavBar></HeaderNavBar>
            <BrowserRouter>
                <Routes>
                    <Route path="/" element={<QueryForm formFields={formFields}
                                                        setFormFields={setFormFields}
                                                        formConstraints={formConstraints}
                                                        setFormConstraints={setFormConstraints}
                                                        table={table}
                                                        setTable={setTable}
                                                        tableFields={tableFields}
                                                        setTableFields={setTableFields}
                                                        query={query}
                                                        setQuery={setQuery}
                                                        originalQueryResults={originalQueryResults}
                                                        setOriginalQueryResults={setOriginalQueryResults}
                                                        refinements={refinements}
                                                        setRefinements={setRefinements}
                                                        err={err}
                                                        setErr={setErr}
                                                        optionalOperators={optionalOperators}/>}>
                        <Route index element={<QueryForm formFields={formFields}
                                                         setFormFields={setFormFields}
                                                         formConstraints={formConstraints}
                                                         setFormConstraints={setFormConstraints}
                                                         table={table}
                                                         setTable={setTable}
                                                         tableFields={tableFields}
                                                         setTableFields={setTableFields}
                                                         query={query}
                                                         setQuery={setQuery}
                                                         originalQueryResults={originalQueryResults}
                                                         setOriginalQueryResults={setOriginalQueryResults}
                                                         refinements={refinements}
                                                         setRefinements={setRefinements}
                                                         err={err}
                                                         setErr={setErr}
                                                         optionalOperators={optionalOperators}/>}/>
                        <Route path="queries" element={<QueryForm formFields={formFields}
                                                                  setFormFields={setFormFields}
                                                                  formConstraints={formConstraints}
                                                                  setFormConstraints={setFormConstraints}
                                                                  table={table}
                                                                  setTable={setTable}
                                                                  tableFields={tableFields}
                                                                  setTableFields={setTableFields}
                                                                  query={query}
                                                                  setQuery={setQuery}
                                                                  originalQueryResults={originalQueryResults}
                                                                  setOriginalQueryResults={setOriginalQueryResults}
                                                                  refinements={refinements}
                                                                  setRefinements={setRefinements}
                                                                  err={err}
                                                                  setErr={setErr}
                                                                  optionalOperators={optionalOperators}/>}/>
                    </Route>
                    <Route path="advanced" element={<RefinementsViewer/>}/>
                </Routes>
            </BrowserRouter>
        </div>
    )
}

export default QueryRefinement;