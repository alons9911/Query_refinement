-- TPC TPC-H Parameter Substitution (Version 3.0.0 build 0)
-- using 1651050030 as a seed to the RNG
-- $ID$
-- TPC-H/TPC-R Pricing Summary Report Query (Q1)
-- Functional Query Definition
-- Approved February 1998

use tpch;


select *
from
	lineitem
where
	l_shipdate <= date '1998-12-01' - interval '63' day
;

