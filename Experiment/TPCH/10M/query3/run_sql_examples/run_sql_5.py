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




sql_filename = '../tpch10M-q3-param.sql'
sql_file = open(sql_filename, 'r')
sql_file_read = sql_file.read()
sql_file.close()
params = {
        'numeric1': "1995-03-10",
        'numeric2': "1995-03-10",
        'categorical1': tuple(["AUTOMOBILE"])
    }
cursor.execute(sql_file_read, params)
fetch_result = cursor.fetchall()
df = pd.DataFrame(fetch_result)
print(df)


sql_connection.commit()





