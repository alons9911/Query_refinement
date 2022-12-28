import React from 'react';
import DynamicTable from "./DynamicTable";
import {useState} from "react";
import {Form} from "react-bootstrap";
import "./ShowQueryTable.css"

function ShowQueryTable(props) {
    const [checked, setChecked] = useState(false);
    const handleChange = () => {
        setChecked(!checked);
    };

    return (
        <>
            <Form>
                <Form.Label xs={5} htmlFor="Switch">Show Query Results</Form.Label>
                <Form.Check
                    type="checkbox"
                    checked={checked}
                    id="show-original-query-results-switch"
                    onChange={handleChange}
                />
            </Form>
            <br/>
            {checked ?
                props.data.length === 0 ?
                    "No Results Found" :
                    <DynamicTable data={props.data} selectedFields={props.selectedFields}/>
                : ''}
        </>
    )
}

export default ShowQueryTable;