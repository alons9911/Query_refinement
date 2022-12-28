import React from 'react';
import "./DynamicTable.css";

function DynamicTable(props) {

// get table column
    const column = Object.keys(props.data[0]).sort();
    const selectedFields = props.selectedFields;

    // get table heading data
    const ThData = () => {
        return (column.includes('id') ? [<th key={'id'}>id</th>] : []).concat(
            column.map((data, index) => {
                return selectedFields.includes(data) ? <th key={data}>{data}</th> : ''
            }))
    }

// get table row data
    const tdData = () => {


        return props.data.map((data) => {
            return (
                <tr class={data['new_result'] ? "active-row" : "not-active-row"}>
                    {
                        (column.includes('id') ? [<td>{data['id']}</td>] : []).concat(
                            column.map((v, index) => {
                                return selectedFields.includes(v) ? <td>{data[v]}</td> : ''
                            })
                        )
                    }
                </tr>
            )
        })
    }


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

export default DynamicTable;