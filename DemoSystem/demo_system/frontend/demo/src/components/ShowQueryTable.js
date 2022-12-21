import DynamicTable from "./DynamicTable";
import {useState} from "react";

function ShowQueryTable(props) {
    const [checked, setChecked] = useState(false);
    const handleChange = () => {setChecked(!checked);};

    return (
        <div>
            <input id="box" onChange={handleChange} type="checkbox"/>
            {checked ?
                props.data.length === 0 ?
                    "No Results Found" :
                        <DynamicTable data={props.data}/>
                : ''}
        </div>
    )
}

export default ShowQueryTable;