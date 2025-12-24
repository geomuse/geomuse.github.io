import psycopg2

conn = psycopg2.connect(
    dbname="demodb",
    user="postgres",
    password="123",
    host="localhost",
    port="5432"
)
cursor = conn.cursor()
select_all_query = "SELECT * FROM users;"

cursor.execute(select_all_query)

all_users = cursor.fetchall()

for row in all_users:
    print(row)