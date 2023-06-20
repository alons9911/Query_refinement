import React from 'react';
import DynamicTable from "./DynamicTable";
import {useState} from "react";
import {Form} from "react-bootstrap";
import "./ShowQueryTable.css"

function QueryView({queryDict}) {
    return (
        <div class="query-view">
            <div class="inner-query-view">
                SELECT {queryDict['select']} <br/>
                FROM {queryDict['from']} <br/>
                WHERE {queryDict['where'].map((clause, indx) => {
                return <>{
                    clause['clause'][0] + " " +
                    clause['clause'][1] + " "}
                    {clause['bold'] ?
                        <b><font class='bold-clause'>{clause['clause'][2]}</font></b> :
                        <>{clause['clause'][2]}</>}
                    {indx !== queryDict['where'].length - 1 ? ' AND ' : ''}
                </>
            })}
            </div>
        </div>
    )
}

export default QueryView;