import pandas as pd
import sqlite3

# Charger le CSV
df = pd.read_csv("machine_learning\features.csv")

conn = sqlite3.connect("image_features.db")
 
 
df.to_sql("image_features", conn, if_exists="replace", index=False)

conn.close()
