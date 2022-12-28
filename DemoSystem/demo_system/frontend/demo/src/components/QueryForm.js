import './QueryForm.css';
import Button from 'react-bootstrap/Button';
import Form from 'react-bootstrap/Form';
import {
    MDBBtn, MDBIcon,
    MDBInputGroup,
} from 'mdb-react-ui-kit';


import React, {useState} from 'react';
import ShowQueryTable from "./ShowQueryTable";
import HeaderNavBar from "./HeaderNavBar";
import {ButtonGroup, ButtonToolbar, Col, Row} from "react-bootstrap";
import {FloatButton, Select} from "antd";
import Container from "react-bootstrap/Container";

function QueryForm() {
    const [formFields, setFormFields] = useState([
        {field: '', operator: '', value: ''},
    ])
    const [formConstraints, setFormConstraints] = useState([
        {field: '', value: '', operator: '', amount: ''},
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

    const handleFieldsFormChange = (event, index) => {
        let data = [...formFields];
        console.log(event);
        data[index][event.target.name] = event.target.value;
        setFormFields(data);
    }

    const handleConstrainsFormChange = (event, index) => {
        let data = [...formConstraints];
        data[index][event.target.name] = event.target.value;
        setFormConstraints(data);
    }


    const setDefault = () => {
        let fields = [
            {field: 'juv_fel_count', operator: '>=', value: '4'},
            {field: 'decile_score', operator: '>=', value: '8'},
            {field: 'c_charge_degree', operator: 'IN', value: '["O"]'},
        ]
        let constraints = [
            {field: 'race', value: 'African-American', operator: '>=', amount: '30'},
            {field: 'sex', value: 'Male', operator: '>=', amount: '45'},
        ]
        setFormFields(fields);
        setFormConstraints(constraints);
    }


    const submit = async (e) => {
        e.preventDefault();
        setErr('')
        console.log(formFields)
        try {
            const build_query_response = await fetch('http://127.0.0.1:5000/build_query', {
                method: 'POST',
                body: JSON.stringify({'conds': formFields, 'table_name': table}),
                headers: {
                    'Content-Type': 'application/json',
                    Accept: 'application/json',
                },
            });

            if (!build_query_response.ok) {
                throw new Error(`Error! status: ${build_query_response.status}`);
            }

            const result = await build_query_response.text();
            console.log('query is: ', result);
            setQuery(JSON.parse(result)["query"]);
            setOriginalQueryResults(JSON.parse(result)["results"]);


            const run_query_response = await fetch('http://127.0.0.1:5000/run_query', {
                method: 'POST',
                body: JSON.stringify({'conds': formFields, 'table_name': table, 'constraints': formConstraints}),
                headers: {
                    'Content-Type': 'application/json',
                    Accept: 'application/json',
                },
            });

            if (!run_query_response.ok) {
                throw new Error(`Error! status: ${run_query_response.status}`);
            }
            const refinements = await run_query_response.text();
            console.log('query is: ', JSON.parse(refinements));
            setRefinements(JSON.parse(refinements));


        } catch (err) {
            setErr(err.message);
        }
    }

    const reset = () => {

        setFormFields([])
        setFormConstraints([])
    }

    const addFields = () => {
        let object = {
            field: '',
            operator: '',
            value: ''
        }
        setFormFields([...formFields, object])
    }

    const removeFields = (index) => {
        let data = [...formFields];
        data.splice(index, 1)
        setFormFields(data)
    }

    const addConstraints = () => {
        let object = {
            field: '',
            value: '',
            operator: '',
            amount: ''
        }

        setFormConstraints([...formConstraints, object])
    }

    const removeConstraints = (index) => {
        let data = [...formConstraints];
        data.splice(index, 1)
        setFormConstraints(data)
    }

    const getSelectedFields = () => formFields.map(f => f.field).concat(formConstraints.map(f => f.field));

    return (
        <div className="QueryRefinement">
            <HeaderNavBar></HeaderNavBar>
            <div className="QueryForm">
                <Container>
                    <Row>
                        <Col sm={7}>
                            <Form onSubmit={submit}>
                                <Form.Group as={Row} className="mb3">
                                    <Row>
                                        <Col xs={5}>
                                            <Form.Label htmlFor="Select">Select DB</Form.Label>
                                        </Col>
                                        <Col>
                                            <Button onClick={() => {
                                            }}
                                                    className='upload-btn rounded-circle' color="secondery" floating
                                                    tag='a'>
                                                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20"
                                                     fill="currentColor" className="bi bi-cloud-upload"
                                                     viewBox="0 0 16 16">
                                                    <path fill-rule="evenodd"
                                                          d="M4.406 1.342A5.53 5.53 0 0 1 8 0c2.69 0 4.923 2 5.166 4.579C14.758 4.804 16 6.137 16 7.773 16 9.569 14.502 11 12.687 11H10a.5.5 0 0 1 0-1h2.688C13.979 10 15 8.988 15 7.773c0-1.216-1.02-2.228-2.313-2.228h-.5v-.5C12.188 2.825 10.328 1 8 1a4.53 4.53 0 0 0-2.941 1.1c-.757.652-1.153 1.438-1.153 2.055v.448l-.445.049C2.064 4.805 1 5.952 1 7.318 1 8.785 2.23 10 3.781 10H6a.5.5 0 0 1 0 1H3.781C1.708 11 0 9.366 0 7.318c0-1.763 1.266-3.223 2.942-3.593.143-.863.698-1.723 1.464-2.383z"/>
                                                    <path fill-rule="evenodd"
                                                          d="M7.646 4.146a.5.5 0 0 1 .708 0l3 3a.5.5 0 0 1-.708.708L8.5 5.707V14.5a.5.5 0 0 1-1 0V5.707L5.354 7.854a.5.5 0 1 1-.708-.708l3-3z"/>
                                                </svg>
                                            </Button>
                                        </Col>
                                    </Row>
                                    <Row>
                                    <Col xs={8}>
                                    <Form.Select id="db-select">
                                        <option>compas-scores</option>
                                    </Form.Select>
                        </Col></Row>
                    </Form.Group>
                    <Form.Group as={Row} className="mb3">
                        <Form.Label htmlFor="Select">Select Conditions</Form.Label>
                        {formFields.map((form, index) => {
                            return (
                                <MDBInputGroup key={index} className='mb-3'>
                                    <Button onClick={() => removeFields(index)}
                                            className='remove-btn rounded-circle' color="secondery" floating
                                            tag='a'>
                                        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16"
                                             fill="currentColor" className="bi bi-trash"
                                             viewBox="0 0 16 16">
                                            <path
                                                d="M5.5 5.5A.5.5 0 0 1 6 6v6a.5.5 0 0 1-1 0V6a.5.5 0 0 1 .5-.5zm2.5 0a.5.5 0 0 1 .5.5v6a.5.5 0 0 1-1 0V6a.5.5 0 0 1 .5-.5zm3 .5a.5.5 0 0 0-1 0v6a.5.5 0 0 0 1 0V6z"/>
                                            <path fill-rule="evenodd"
                                                  d="M14.5 3a1 1 0 0 1-1 1H13v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V4h-.5a1 1 0 0 1-1-1V2a1 1 0 0 1 1-1H6a1 1 0 0 1 1-1h2a1 1 0 0 1 1 1h3.5a1 1 0 0 1 1 1v1zM4.118 4 4 4.059V13a1 1 0 0 0 1 1h6a1 1 0 0 0 1-1V4.059L11.882 4H4.118zM2.5 3V2h11v1h-11z"/>
                                        </svg>
                                    </Button>
                                    <input
                                        className='form-conditions-control'
                                        name='field'
                                        placeholder='Field'
                                        onChange={event => handleFieldsFormChange(event, index)}
                                        value={form.field}
                                    />
                                    <input
                                        className='form-conditions-control-short'
                                        name='operator'
                                        placeholder='op'
                                        onChange={event => handleFieldsFormChange(event, index)}
                                        value={form.operator}
                                    />
                                    <input
                                        className='form-conditions-control'
                                        name='value'
                                        placeholder='value'
                                        onChange={event => handleFieldsFormChange(event, index)}
                                        value={form.value}
                                    />
                                </MDBInputGroup>
                            );
                        })}
                    </Form.Group>
                    <Button className='add-btn rounded-circle' onClick={addFields} floating tag='a'>
                        <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" fill="currentColor"
                             className="bi bi-plus-circle-fill" viewBox="0 0 16 16">
                            <path
                                d="M16 8A8 8 0 1 1 0 8a8 8 0 0 1 16 0zM8.5 4.5a.5.5 0 0 0-1 0v3h-3a.5.5 0 0 0 0 1h3v3a.5.5 0 0 0 1 0v-3h3a.5.5 0 0 0 0-1h-3v-3z"/>
                        </svg>
                    </Button>

                    <Form.Group as={Row} className="mb3">
                        <Form.Label htmlFor="Select">Select Constraints</Form.Label>


                        {formConstraints.map((form, index) => {
                            return (
                                <MDBInputGroup key={index} className='mb-3'>
                                    <Button onClick={() => removeConstraints(index)}
                                            className='remove-btn rounded-circle' color="secondery" floating
                                            tag='a'>
                                        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16"
                                             fill="currentColor" className="bi bi-trash"
                                             viewBox="0 0 16 16">
                                            <path
                                                d="M5.5 5.5A.5.5 0 0 1 6 6v6a.5.5 0 0 1-1 0V6a.5.5 0 0 1 .5-.5zm2.5 0a.5.5 0 0 1 .5.5v6a.5.5 0 0 1-1 0V6a.5.5 0 0 1 .5-.5zm3 .5a.5.5 0 0 0-1 0v6a.5.5 0 0 0 1 0V6z"/>
                                            <path fill-rule="evenodd"
                                                  d="M14.5 3a1 1 0 0 1-1 1H13v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V4h-.5a1 1 0 0 1-1-1V2a1 1 0 0 1 1-1H6a1 1 0 0 1 1-1h2a1 1 0 0 1 1 1h3.5a1 1 0 0 1 1 1v1zM4.118 4 4 4.059V13a1 1 0 0 0 1 1h6a1 1 0 0 0 1-1V4.059L11.882 4H4.118zM2.5 3V2h11v1h-11z"/>
                                        </svg>
                                    </Button>
                                    <input
                                        className='form-constraints-control'
                                        name='field'
                                        placeholder='Field'
                                        onChange={event => handleConstrainsFormChange(event, index)}
                                        value={form.field}
                                    /><b>=</b>
                                    <input
                                        className='form-constraints-control'
                                        name='value'
                                        placeholder='value'
                                        onChange={event => handleConstrainsFormChange(event, index)}
                                        value={form.value}
                                    />
                                    <input
                                        className='form-constraints-control-short'
                                        name='operator'
                                        placeholder='op'
                                        onChange={event => handleConstrainsFormChange(event, index)}
                                        value={form.operator}
                                    />
                                    <input
                                        className='form-constraints-control-short'
                                        name='amount'
                                        placeholder='amount'
                                        onChange={event => handleConstrainsFormChange(event, index)}
                                        value={form.amount}
                                    />
                                </MDBInputGroup>
                            );
                        })}
                    </Form.Group>
                    <Button className='add-btn rounded-circle' onClick={addConstraints} floating tag='a'>
                        <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" fill="currentColor"
                             className="bi bi-plus-circle-fill" viewBox="0 0 16 16">
                            <path
                                d="M16 8A8 8 0 1 1 0 8a8 8 0 0 1 16 0zM8.5 4.5a.5.5 0 0 0-1 0v3h-3a.5.5 0 0 0 0 1h3v3a.5.5 0 0 0 1 0v-3h3a.5.5 0 0 0 0-1h-3v-3z"/>
                        </svg>
                    </Button>


                    <ButtonGroup className="me-2">
                        <Button className='set-default-btn rounded-circle' onClick={setDefault} tag='a'>
                            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16"
                                 fill="currentColor"
                                 className="bi bi-shuffle" viewBox="0 0 16 16">
                                <path fill-rule="evenodd"
                                      d="M0 3.5A.5.5 0 0 1 .5 3H1c2.202 0 3.827 1.24 4.874 2.418.49.552.865 1.102 1.126 1.532.26-.43.636-.98 1.126-1.532C9.173 4.24 10.798 3 13 3v1c-1.798 0-3.173 1.01-4.126 2.082A9.624 9.624 0 0 0 7.556 8a9.624 9.624 0 0 0 1.317 1.918C9.828 10.99 11.204 12 13 12v1c-2.202 0-3.827-1.24-4.874-2.418A10.595 10.595 0 0 1 7 9.05c-.26.43-.636.98-1.126 1.532C4.827 11.76 3.202 13 1 13H.5a.5.5 0 0 1 0-1H1c1.798 0 3.173-1.01 4.126-2.082A9.624 9.624 0 0 0 6.444 8a9.624 9.624 0 0 0-1.317-1.918C4.172 5.01 2.796 4 1 4H.5a.5.5 0 0 1-.5-.5z"/>
                                <path
                                    d="M13 5.466V1.534a.25.25 0 0 1 .41-.192l2.36 1.966c.12.1.12.284 0 .384l-2.36 1.966a.25.25 0 0 1-.41-.192zm0 9v-3.932a.25.25 0 0 1 .41-.192l2.36 1.966c.12.1.12.284 0 .384l-2.36 1.966a.25.25 0 0 1-.41-.192z"/>
                            </svg>
                        </Button>
                        <Button className='remove-btn rounded-circle' onClick={reset} tag='a'>
                            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16"
                                 fill="currentColor" className="bi bi-trash" viewBox="0 0 16 16">
                                <path
                                    d="M5.5 5.5A.5.5 0 0 1 6 6v6a.5.5 0 0 1-1 0V6a.5.5 0 0 1 .5-.5zm2.5 0a.5.5 0 0 1 .5.5v6a.5.5 0 0 1-1 0V6a.5.5 0 0 1 .5-.5zm3 .5a.5.5 0 0 0-1 0v6a.5.5 0 0 0 1 0V6z"/>
                                <path fill-rule="evenodd"
                                      d="M14.5 3a1 1 0 0 1-1 1H13v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V4h-.5a1 1 0 0 1-1-1V2a1 1 0 0 1 1-1H6a1 1 0 0 1 1-1h2a1 1 0 0 1 1 1h3.5a1 1 0 0 1 1 1v1zM4.118 4 4 4.059V13a1 1 0 0 0 1 1h6a1 1 0 0 0 1-1V4.059L11.882 4H4.118zM2.5 3V2h11v1h-11z"/>
                            </svg>
                        </Button>
                        <Button className='submit-btn rounded-circle' onClick={submit} tag='a'>
                            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20"
                                 fill="currentColor"
                                 className="bi bi-send-fill" viewBox="0 0 16 16">
                                <path
                                    d="M15.964.686a.5.5 0 0 0-.65-.65L.767 5.855H.766l-.452.18a.5.5 0 0 0-.082.887l.41.26.001.002 4.995 3.178 3.178 4.995.002.002.26.41a.5.5 0 0 0 .886-.083l6-15Zm-1.833 1.89L6.637 10.07l-.215-.338a.5.5 0 0 0-.154-.154l-.338-.215 7.494-7.494 1.178-.471-.47 1.178Z"/>
                            </svg>
                        </Button>
                    </ButtonGroup>
                </Form>
            </Col>
            <Col sm={5}>
                <div>{query !== '' ?
                    <div><br/><h3>Your requested query</h3> <br/>{query}<br/><br/>
                        <ShowQueryTable data={originalQueryResults}
                                        selectedFields={getSelectedFields()}></ShowQueryTable></div> : ''}</div>
            </Col>
        </Row>
</Container>


    <br/>
    <br/>
    <div>{err !== '' ? "Error: " + err : ''}</div>
    <br/>
    <div>{query !== '' ?
        <>
        <h3>We found some minimal refinements:</h3>
        {refinements.map(function (ref, i) {
            return <><p>{i}: {ref['query']} <br/><b>Similarity to Original
                Query: {ref['distance_to_original']}</b><br/></p><ShowQueryTable
                data={ref['results']} selectedFields={getSelectedFields()}></ShowQueryTable></>;
        })}</>:''}
    </div>
</div>
</div>
)
    ;
}

export default QueryForm;
