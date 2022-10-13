import mysql.connector
import json
from decimal import Decimal
import pandas as pd

# Your connection code here...

host_args = {
    "host": "localhost",
    "user": "root",
    "password": "ljy19980228"
}

con = mysql.connector.connect(**host_args)

cur = con.cursor(dictionary=True, buffered=True)


sql_filename = '../../query1/tpch10M-q1.sql'
sql_file = open(sql_filename, 'r')
sql_file_read = sql_file.read()
sql_file.close()

result_iterator = cur.execute(sql_file_read, multi=True)
for res in result_iterator:
    print("Running query: ", res)
    if res.with_rows:
        fetch_result = res.fetchall()
        df = pd.DataFrame(fetch_result)
        print(df)
    elif res.rowcount > 0:
        print(f"Affected {res.rowcount} rows")

con.commit()



