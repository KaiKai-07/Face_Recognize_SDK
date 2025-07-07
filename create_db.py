import sqlite3
import numpy as np
import io

def adapt_array(arr):
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())

def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)


# Converts np.array to TEXT when inserting
sqlite3.register_adapter(np.ndarray, adapt_array)

# Converts TEXT to np.array when selecting
sqlite3.register_converter("array", convert_array)

con = sqlite3.connect("face.db", detect_types=sqlite3.PARSE_DECLTYPES)
cur = con.cursor()

cur.execute("""
    CREATE TABLE IF NOT EXISTS face (
        id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
        name TEXT NOT NULL,
        feature_type TEXT NOT NULL,
        feature ARRAY NOT NULL
    )
""")
con.commit()
con.close()
