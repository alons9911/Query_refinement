select distinct {}
from
	customer,
	orders,
	lineitem
where
        c_custkey = o_custkey
	and l_orderkey = o_orderkey

order by {}

