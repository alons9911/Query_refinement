import React from 'react';
import "./DynamicTable.css";

function DynamicTable(props) {

// get table column
    const column = Object.keys(props.data[0]).sort();
    const selectedFields = props.selectedFields;
    const removedRows = props.removedFromOriginal !== undefined ? props.removedFromOriginal : [];
    console.log(removedRows);
    // get table heading data
    const ThData = () => {
        return (column.includes('id') ? [<th key={'id'}>id</th>] : []).concat(
            column.map((data, index) => {
                return selectedFields.includes(data) ? <th key={data}>{data}</th> : ''
            }))
    }

// get table row data
    const tdData = () => {


        return props.data.concat(props.removedFromOriginal).sort((a, b) => a['id'] - b['id']).map((row) => {
            return (
                props.data.includes(row) ?
                    <tr class={row['new_result'] ? "active-row" : "not-active-row"}>
                        {
                            (column.includes('id') ? [<td>{row['id']}</td>] : []).concat(
                                column.map((v, index) => {
                                    return selectedFields.includes(v) ? <td>{row[v]}</td> : ''
                                })
                            )
                        }
                    </tr> :
                    <tr className="removed-row">
                        {
                            (column.includes('id') ? [<td>{row['id']}</td>] : []).concat(
                                column.map((v, index) => {
                                    return selectedFields.includes(v) ? <td>{row[v]}</td> : ''
                                })
                            )
                        }
                    </tr>
            )
        })
    }


    return (
        <div className={props.className}>
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