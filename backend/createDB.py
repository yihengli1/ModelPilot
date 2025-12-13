import os
import pymysql
from dotenv import load_dotenv

load_dotenv(".env")

connection = pymysql.connect(
    host=os.getenv("DB_HOST"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    port=int(os.getenv("DB_PORT", 3306)),
    cursorclass=pymysql.cursors.DictCursor
)

try:
    with connection.cursor() as cursor:
        db_name = "modelpilot"
        print(f"Creating database: {db_name}...")
        cursor.execute(
            f"CREATE DATABASE IF NOT EXISTS {db_name} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;")
        print("Success! Database created.")
finally:
    connection.close()
