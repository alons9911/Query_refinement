import './QueryForm.css';


import React, {useState} from 'react';
import Form from "react-bootstrap/Form";
import {Col, Row} from "react-bootstrap";
import {Select} from "antd";
import {MdClose} from "react-icons/md";
import ShowQueryTable from "./ShowQueryTable";
import QueryView from "./QueryView";
import Popover from "react-bootstrap/Popover";
import OverlayTrigger from "react-bootstrap/OverlayTrigger";
import Button from "react-bootstrap/Button";


class RefinementsViewer extends React.Component {

    sortByOptions = ['Change In Result', 'Change In Selection Conditions', 'Change In Groups Cardinality']

    // Constructor
    constructor(props) {
        super(props);

        this.state = {
            selectedSortBy: this.sortByOptions[1],
            unlikelyChangedFields: [],
            refinements: [],
            loading: false,
            tableName: '',
            formFields: [],
            selectedFields: [],
            higherToLowerSorting: true
        };

    }

    setRefinements(refs) {
        this.setState({refinements: refs});
    }

    swapSortingOrder(currentOrder){
        this.setState({higherToLowerSorting: !currentOrder})
    }

    setSelectedSortBy(selected) {
        this.setState({selectedSortBy: selected});
    }

    setUnlikelyChangedFields(fields) {
        this.setState({unlikelyChangedFields: fields});
    }


    swapRefinementsIfNeeded(refs) {
        return this.state.higherToLowerSorting ? refs : refs.reverse();
    }

    // ComponentDidMount is used to
    // execute the code
    componentDidMount() {
        fetch('http://127.0.0.1:5000/load_refinements_viewer_state', {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json',
                Accept: 'application/json',
            },
        }).then((refinemenets_response) => {
            if (!refinemenets_response.ok) {
                throw new Error(`Error! status: ${refinemenets_response.status}`);
            }
            return refinemenets_response.text()
        }).then(
            result => JSON.parse(result)
        ).then((result) => {
            console.log('refinemenets loaded: ', result['refinements']);
            console.log('form_fields loaded: ', result['form_fields']);
            console.log('table loaded: ', result['table']);
            console.log('selected_fields loaded: ', result['selected_fields']);
            console.log('original_str_query_as_dict loaded: ', result['original_str_query_as_dict']);
            this.setState({
                refinements: result['refinements'],
                formFields: result['form_fields'],
                tableName: result['table'],
                selectedFields: result['selected_fields'],
                originalQuery: result['original_str_query_as_dict'],
                loading: true
            });
        });
    }


    sendSortRefirementsRequest = async (unlikelyFields) => {
        const sort_refinements_response = await fetch('http://127.0.0.1:5000/sort_refinements', {
            method: 'POST',
            body: JSON.stringify({
                'conds': this.state.formFields,
                'table_name': this.state.tableName,
                'refinements': this.state.refinements,
                'sorting_func': this.state.selectedSortBy,
                'unlikely_changed_fields': unlikelyFields
            }),
            headers: {
                'Content-Type': 'application/json',
                Accept: 'application/json',
            },
        });

        if (!sort_refinements_response.ok) {
            throw new Error(`Error! status: ${sort_refinements_response.status}`);
        }
        const sorted_refinements = await sort_refinements_response.text();
        console.log('sorted refinements: ', JSON.parse(sorted_refinements));
        this.setRefinements(JSON.parse(sorted_refinements));
    }
    handleSortBySelection = async (event) => {
        this.setSelectedSortBy(event);
        await this.sendSortRefirementsRequest([]);
    }

    // Add or remove tags by using the key
    handleUnlikelyChangedFields = async event => {
        if (event.key === "Enter" && event.target.value !== "" && this.state.unlikelyChangedFields.length < 10) {
            let fields = [...this.state.unlikelyChangedFields];
            this.setUnlikelyChangedFields([...this.state.unlikelyChangedFields, event.target.value]);
            await this.sendSortRefirementsRequest([...fields, event.target.value]);
            event.target.value = "";
        } else if (event.key === "Backspace" && this.state.unlikelyChangedFields.length && event.target.value === 0) {
            const unlikelyChangedFieldsCopy = [...this.state.unlikelyChangedFields];
            unlikelyChangedFieldsCopy.pop();
            event.preventDefault();
            this.setUnlikelyChangedFields(unlikelyChangedFieldsCopy);
        }
    };

    //Remove tags by clicking the cross sign
    removeUnlikelyChangedFields = async index => {
        this.setUnlikelyChangedFields([...this.state.unlikelyChangedFields
            .filter(field => this.state.unlikelyChangedFields.indexOf(field) !== index)]);
        await this.sendSortRefirementsRequest(this.state.unlikelyChangedFields
            .filter(field => this.state.unlikelyChangedFields.indexOf(field) !== index));
    };

    render() {
        const {
            selectedSortBy,
            unlikelyChangedFields,
            refinements,
            loading,
            tableName,
            formFields,
            selectedFields,
            higherToLowerSorting
        } = this.state;


        if (!loading) return <div className="QueryRefinement">
            <h1> Loading.... </h1></div>;

        return (
            <div className="QueryRefinement">
                <div>{refinements.length !== 0 ?
                    <>
                        <h3>We found some minimal refinements:</h3>
                        <Form>
                            <Row>
                                <Col>
                                    <Form.Group as={Row} className="mb3">


                                        <Form.Label htmlFor="Select">Sort By
                                            <Button onClick={() => {this.swapSortingOrder(this.state.higherToLowerSorting)}}
                                                    className='sort-direction-btn rounded-circle' color="secondery"
                                                    tag='a'>
                                                {this.state.higherToLowerSorting ?
                                                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16"
                                                         fill="red" className="bi bi-arrow-down">
                                                        <path fill-rule="evenodd"
                                                              d="M8 1a.5.5 0 0 1 .5.5v11.793l3.146-3.147a.5.5 0 0 1 .708.708l-4 4a.5.5 0 0 1-.708 0l-4-4a.5.5 0 0 1 .708-.708L7.5 13.293V1.5A.5.5 0 0 1 8 1z"/>
                                                    </svg> :
                                                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16"
                                                         fill="green" className="bi bi-arrow-up">
                                                        <path fill-rule="evenodd"
                                                              d="M8 15a.5.5 0 0 0 .5-.5V2.707l3.146 3.147a.5.5 0 0 0 .708-.708l-4-4a.5.5 0 0 0-.708 0l-4 4a.5.5 0 1 0 .708.708L7.5 2.707V14.5a.5.5 0 0 0 .5.5z"/>
                                                    </svg>
                                                }
                                            </Button>
                                        </Form.Label>
                                        <Select className="sort-by-select"
                                                defaultValue={selectedSortBy}
                                                options={this.sortByOptions.map((o) => {
                                                    return {value: o, label: o}
                                                })}
                                                onChange={this.handleSortBySelection}
                                        >
                                        </Select>
                                    </Form.Group>
                                </Col>
                                <Col sm={7}>
                                    {selectedSortBy === 'Change In Selection Conditions' ?
                                        <Form.Group as={Row} className="mb3">
                                            <Form.Label htmlFor="Select"> Choose predicate to sort</Form.Label>
                                            <div className="tags">
                                                {unlikelyChangedFields.map((field, index) => (
                                                    <div className="single-tag" key={index}>
                                                        <span>{field}</span>
                                                        <i
                                                            onClick={() => this.removeUnlikelyChangedFields(index)}
                                                        >
                                                            <MdClose/>
                                                        </i>
                                                    </div>
                                                ))}

                                                <input
                                                    className="unlikely-changed-fields-input"
                                                    type="text"
                                                    onKeyDown={event => this.handleUnlikelyChangedFields(event)}
                                                    placeholder="Select predicates and press enter"
                                                />
                                            </div>
                                        </Form.Group>

                                        : ''}
                                </Col>
                            </Row>
                        </Form>

                        <OverlayTrigger trigger="hover" placement="right"
                                        overlay={<Popover id="popover-basic">
                                            <Popover.Header as="h3">Original Query</Popover.Header>
                                            <Popover.Body>
                                                <QueryView queryDict={this.state.originalQuery}></QueryView>
                                            </Popover.Body>
                                        </Popover>}>
                            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="green"
                                 className="bi bi-lightbulb-fill" viewBox="0 0 16 16">
                                <path
                                    d="M2 6a6 6 0 1 1 10.174 4.31c-.203.196-.359.4-.453.619l-.762 1.769A.5.5 0 0 1 10.5 13h-5a.5.5 0 0 1-.46-.302l-.761-1.77a1.964 1.964 0 0 0-.453-.618A5.984 5.984 0 0 1 2 6zm3 8.5a.5.5 0 0 1 .5-.5h5a.5.5 0 0 1 0 1l-.224.447a1 1 0 0 1-.894.553H6.618a1 1 0 0 1-.894-.553L5.5 15a.5.5 0 0 1-.5-.5z"/>
                            </svg>

                        </OverlayTrigger>
                        <br/>
                        <ol>
                            {this.swapRefinementsIfNeeded(refinements).map(function (ref, i) {
                                return <Row className={"refinement-row"}>
                                    <Col sm={7}>
                                        <li>
                                            <QueryView queryDict={ref['str_query_as_dict']}/><br/>
                                            <div className="align-center-div"><b>Result Similarity
                                                Score: {ref['jaccard_similarity']}</b><br/>
                                                <>{ref['cardinality_satisfaction'].map((con) => <div>{con['group']} = {con['amount']}<br/></div>)}</>

                                            </div>
                                            <br/></li>
                                    </Col>
                                    <Col sm={5}>
                                        <ShowQueryTable containerClassName={"refinements-dynamic-table"}
                                                        data={ref['results']['query_results']}
                                                        selectedFields={selectedFields}
                                                        alwaysShow={true}
                                                        removedFromOriginal={ref['results']['removed_from_original']}></ShowQueryTable>

                                    </Col>
                                </Row>;

                            })}
                        </ol>


                    </> : ''}
                </div>
            </div>
        );
    }
}


export default RefinementsViewer;
