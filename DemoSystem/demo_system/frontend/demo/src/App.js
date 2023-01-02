import React from 'react';
import './App.css';
import QueryForm from "./components/QueryForm";
import 'bootstrap/dist/css/bootstrap.min.css';
import 'antd/dist/reset.css';
import HeaderNavBar from "./components/HeaderNavBar";
import {
    BrowserRouter as Router, Route, Link, Routes, BrowserRouter
} from "react-router-dom";

const App = props => {
    return (<html lang="en">
    <head>
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.0.0/dist/css/bootstrap.min.css"
              integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm"
              crossorigin="anonymous"/>
        <title>Query Refinement</title>

    </head>
    <body>
    <script src="https://code.jquery.com/jquery-3.2.1.slim.min.js"
            integrity="sha384-KJ3o2DKtIkvYIK3UENzmM7KCkRr/rE9/Qpg6aAZGJwFDMVNA/GpGFF93hXpG5KkN"
            crossorigin="anonymous"></script>
    <script src="https://cdn.jsdelivr.net/npm/popper.js@1.12.9/dist/umd/popper.min.js"
            integrity="sha384-ApNbgh9B+Y1QKtv3Rn7W3mgPxhU9K/ScQsAP7hUibX39j7fakFPskvXusvfa0b4Q"
            crossorigin="anonymous"></script>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@4.0.0/dist/js/bootstrap.min.js"
            integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl"
            crossorigin="anonymous"></script>
    <div>
        <HeaderNavBar></HeaderNavBar>
        <BrowserRouter>
            <Routes>
                <Route path="/" element={<QueryForm/>}>
                    <Route index element={<QueryForm/>}/>
                    <Route path="queries" element={<QueryForm/>}/>
                </Route>
            </Routes>
        </BrowserRouter>
    </div>
    </body>
    </html>);
};

export default App;
