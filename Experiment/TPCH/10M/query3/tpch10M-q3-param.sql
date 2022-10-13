select *
from
	customer,
	orders,
	lineitem
where
	c_mktsegment in %(categorical1)s
	and c_custkey = o_custkey
	and l_orderkey = o_orderkey
	and o_orderdate < date %(numeric1)s
	and l_shipdate > date %(numeric2)s
;

