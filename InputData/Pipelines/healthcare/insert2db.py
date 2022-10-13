import pymysql
import string
import random
import pandas as pd
from pandas.io import sql
from sqlalchemy import create_engine

host = 'localhost'
port = 3306
user = 'root'
passwd = 'ljy19980228'
db = 'Proj2'
charset = 'utf8'

file = r"before_selection.csv"
data = pd.read_csv(file)

# data.to_sql(name='Proj2', con=conn, if_exists='replace', index=False, flavor='mysql')
#


engine = create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}"
                       .format(user=user,
                               pw=passwd,
                               db=db))
data.to_sql(con=engine, name='healthcare', if_exists='replace')
