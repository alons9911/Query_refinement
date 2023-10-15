# Erica: Query Refinement for Diversity Constraint Satisfaction

Accepted by VLDB 2023 demo track. https://dl.acm.org/doi/abs/10.14778/3611540.3611623

For technical report of the research paper, please refer to https://github.com/JinyangLi01/Query_refinement/blob/master/FullPaper/Query_Refinement.pdf.

## Abstract
Relational queries are commonly used to support decision making in critical domains like hiring and college admissions. For example, a college admissions officer may need to select a subset of the applicants for in-person interviews, who individually meet the qualification requirements (e.g., have a sufficiently high GPA) and are collectively demographically diverse (e.g., include a sufficient number of candidates of each gender and of each race). However, traditional relational queries only support selection conditions checked against each input tuple, and they do not support diversity conditions checked against multiple, possibly overlapping, groups of output tuples. To address this shortcoming, we present ERICA, an interactive system that proposes minimal modifications for selection queries to have them satisfy constraints on the cardinalities of multiple groups in the result. We demonstrate the effectiveness of ERICA using several real-life datasets and diversity requirements.


## Algorithms
baseline algorithm: traverse all possible refinements. located in Algorithm/Baseline.py

our algorithm: use provenance expressions and PVL to accelerate the searching. 
located in Algorithm/ProvenanceSearchValues.py


## How to run Erica
In order to run the system, there are two components you need to run:
- The backend, which can be found in DemoSystem\demo_system\server\server.py.
you can run it using 'python3 server.py'
- The frontend, which can be found in DemoSystem\demo_system\frontend\demo\.
you can run it using the standard 'npm build' and 'npm start' commands.




