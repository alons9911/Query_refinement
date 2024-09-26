import sqlite3
from sqlite3 import Error

import pandas
import requests

from erica_backend.query_translator import build_query


def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by the db_file
    :param db_file: database file
    :return: Connection object or None
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except Error as e:
        print(e)

    return conn


def get_query_results(query):
    conn = create_connection("../../../demo.sqlite")
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(query)

    rows = cur.fetchall()
    results = []
    if len(rows) > 0:
        keys = rows[0].keys()
        results = [{keys[i]: row[i] for i in range(len(keys))} for row in rows]
    # print(results)
    return results


if __name__ == "__main__":
    res = get_query_results(
        "SELECT * FROM 'compas_scores' AS c WHERE c.juv_fel_count >= 3.0 AND c.decile_score >= 8.0 AND c.c_charge_degree in ('O', 'F', 'M');")
    print(res)