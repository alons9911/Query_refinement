-- TPC TPC-H Parameter Substitution (Version 3.0.0 build 0)
-- using 1651050030 as a seed to the RNG
-- $ID$
-- TPC-H/TPC-R Shipping Priority Query (Q3)
-- Functional Query Definition
-- Approved February 1998

-- use tpch;

select *
from
	customer,
	orders,
	lineitem
where
	c_mktsegment = 'AUTOMOBILE'
	and c_custkey = o_custkey
	and l_orderkey = o_orderkey
	and o_orderdate < date '1995-03-10'
	and l_shipdate > date '1995-03-10'
;

