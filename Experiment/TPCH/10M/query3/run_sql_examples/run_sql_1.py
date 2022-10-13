import mysql.connector
import json

host_args = {
    "host": "localhost",
    "user": "root",
    "password": "ljy19980228"
}

con = mysql.connector.connect(**host_args)

cur = con.cursor(dictionary=True, buffered=True)


# Your connection code here...

sql_filename = '../../query1/tpch10M-q1.sql'
sql_file = open(sql_filename, 'r')
sql_file_read = sql_file.read()
sql_file.close()
result_iterator = cur.execute(sql_file_read, multi=True)
for res in result_iterator:
    print("Running query: ", res)  # Will print out a short representation of the query
    print(f"Affected {res.rowcount} rows")

con.commit()  # Remember to commit all your changes!


