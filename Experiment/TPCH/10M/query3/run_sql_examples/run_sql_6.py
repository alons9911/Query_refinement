import pymysql
import json
from decimal import Decimal
import pandas as pd

# Your connection code here...

host_args = {
    "host": "localhost",
    "user": "root",
    "password": "ljy19980228",
    'db': 'tpch'
}

sql_connection = pymysql.connect(**host_args)


cursor = sql_connection.cursor()



sql_getinfo_filename = '../tpch10M-q3-getinfo.sql'
sql_getinfo_file = open(sql_getinfo_filename, 'r')
sql_getinfo_file_read = sql_getinfo_file.read()
sql_getinfo_file.close()



sql_getinfo_file_read = sql_getinfo_file_read.format("o_orderdate", "o_orderdate")
print(sql_getinfo_file_read)

cursor.execute(sql_getinfo_file_read)
fetch_result = cursor.fetchall()

lst = [x[0] for x in fetch_result]
print(lst)


sql_connection.commit()





