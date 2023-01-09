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

    const alwaysShow = props.alwaysShow || false;
    return (
        <>
            {alwaysShow !== true ?
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
            <br/></>:""}
            {alwaysShow === true || checked ?
                props.data.length === 0 ?
                    "No Results Found" :
                    <DynamicTable className={props.containerClassName}
                                  data={props.data}
                                  removedFromOriginal={props.removedFromOriginal}
                                  selectedFields={props.selectedFields}/>
                : ''}
        </>
    )
}

export default ShowQueryTable;