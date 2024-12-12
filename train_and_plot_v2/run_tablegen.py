import sys
import json
import csv

if len(sys.argv) != 2:
    print(f"usage: {sys.argv[0]} table.json")
    exit(1)

table_config_file = sys.argv[1]
table_config = json.loads(open(table_config_file).read())

metric = table_config["metric"]

table = {row["row_name"]: {} for row in table_config["rows"]}

for row in table_config["rows"]:
    for col in row["cols"]:
        table_row = table[row["row_name"]]
        col_name = col["col_name"]
        
        if col_name in table_row:
            raise Exception("collision in table")
        else:
            results_file = col["results_file"]
            results = json.loads(open(results_file).read())
            value = results["best_model"][metric]
            table_row[col_name] = value

if not (min(len(row_values) for row_values in table.values())
    == max(len(row_values) for row_values in table.values())):
    raise Exception("uneven table rows")

csv_rows = []
any_table_row = next(row_values for row_values in table.values())
col_names = ["name"] + list(any_table_row.keys())
csv_rows.append(col_names)

row_names = list(table.keys())

for i, row_values in enumerate(table.values()):
   row_name = row_names[i] 
   # works because all row dicts keep their insertion order
   # thus all values are in the correct order for each row
   values = list(row_values.values())
   value_row = [row_name] + values
   csv_rows.append(value_row)

csv_file = table_config_file.rsplit(".", 1)[0] + ".csv"
with open(csv_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(csv_rows)
