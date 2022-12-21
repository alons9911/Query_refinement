import './QueryForm.css';

import React, {useState} from 'react';
import DynamicTable from "./DynamicTable";
import ShowQueryTable from "./ShowQueryTable";

function QueryForm() {
  const [formFields, setFormFields] = useState([
    { field: '', operator: '', value:'' },
  ])
  const [formConstraints, setFormConstraints] = useState([
    { field: '', value: '', operator:'', amount:'' },
  ])
  const [table, setTable] = useState('compas-scores');

  const [query, setQuery] = useState('');
  const [originalQueryResults, setOriginalQueryResults] = useState([]);

  const [refinements, setRefinements] = useState([]);
  const [err, setErr] = useState('');

  const handleFieldsFormChange = (event, index) => {
    let data = [...formFields];
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
        {field: 'juv_fel_count', operator: '>=', value:'4'},
        {field: 'decile_score', operator: '>=', value:'8'},
        {field: 'c_charge_degree', operator: 'IN', value:'["O"]'},
    ]
    let constraints = [
        {field: 'race',  value:'African-American', operator: '>=', amount: '30'},
        {field: 'sex',  value:'Male', operator: '>=', amount: '45'},
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
      value:''
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
      operator:'',
      amount:''
    }

    setFormConstraints([...formConstraints, object])
  }

  const removeConstraints = (index) => {
    let data = [...formConstraints];
    data.splice(index, 1)
    setFormConstraints(data)
  }

  return (
    <div className="QueryForm">
      <form onSubmit={submit}>
        <h3>Choose DB Table:</h3>
        <input
                name='table'
                value={table}
        />
        <h3>Choose Conditions:</h3>
        {formFields.map((form, index) => {
          return (
            <div key={index}>
              <input
                name='field'
                placeholder='Field'
                onChange={event => handleFieldsFormChange(event, index)}
                value={form.field}
              />
              <input
                name='operator'
                placeholder='operator'
                onChange={event => handleFieldsFormChange(event, index)}
                value={form.operator}
              />
              <input
                name='value'
                placeholder='value'
                onChange={event => handleFieldsFormChange(event, index)}
                value={form.value}
              />
              <button onClick={() => removeFields(index)}>Remove</button>
            </div>
          )
        })}
        <button onClick={addFields}>Add More Conditions..</button>

        <br/>
        <h3>Choose Constraints:</h3>
        {formConstraints.map((form, index) => {
          return (
            <div key={index}>
              <input
                name='field'
                placeholder='Field'
                onChange={event => handleConstrainsFormChange(event, index)}
                value={form.field}
              />
              <input
                name='value'
                placeholder='value'
                onChange={event => handleConstrainsFormChange(event, index)}
                value={form.value}
              />
              <input
                name='operator'
                placeholder='operator'
                onChange={event => handleConstrainsFormChange(event, index)}
                value={form.operator}
              />
              <input
                name='amount'
                placeholder='amount'
                onChange={event => handleConstrainsFormChange(event, index)}
                value={form.amount}
              />
              <button onClick={() => removeConstraints(index)}>Remove</button>
            </div>
          )
        })}
      </form>
      <button onClick={addConstraints}>Add More Constraints..</button>
      <br/><br/>
      <button onClick={setDefault}>Set Default</button> <button onClick={submit}>Submit</button> <button onClick={reset}>Reset</button>
      <br/><br/>
      <div>{query !== '' ? <div> <h3>Your requested query:</h3> {query} <br/>Query Results:<br/><ShowQueryTable data={originalQueryResults}></ShowQueryTable></div>:''}</div>
    <div>{err !== '' ? "Error: " + err: ''}</div>
      <br/>
      <div>
        <h3>We found some minimal refinements:</h3>
        {refinements.map(function(ref, i){
          return <p>{i}: {ref['query']} <br/><b>Distance To Original Query: {ref['distance_to_original']}</b><br/> <ShowQueryTable data={ref['results']}></ShowQueryTable> <br/></p>;
        })}
      </div>
    </div>
  );
}
export default QueryForm;
