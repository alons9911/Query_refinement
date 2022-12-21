import "./DynamicTable.css";

function DynamicTable(props){

// get table column
 const column = Object.keys(props.data[0]).sort();

 // get table heading data
 const ThData =()=>{

     return column.map((data)=>{
         return data === 'new_result' ? '' :<th key={data}>{data}</th>
     })
 }

// get table row data
const tdData =() =>{

     return props.data.map((data)=>{
       return(
           <tr class={data['new_result'] ? "active-row" : "not-active-row"}>
                {
                   column.map((v)=>{
                       return v === 'new_result' ? '' : <td>{data[v]}</td>
                   })
                }
           </tr>
       )
     })
}


  return (
      <table className="query-table">
        <thead>
         <tr>{ThData()}</tr>
        </thead>
        <tbody>
        {tdData()}
        </tbody>
       </table>
  )
}
export default DynamicTable;