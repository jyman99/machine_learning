import sqlite3 as lite
con = lite.connect('./mxm_dataset.db')
cur = con.cursor()
QUERY = "SELECT * FROM lyrics" 
cur.execute(QUERY)


# replace with clean_date() or something
for row in cur.fetchall():
    print(row)
