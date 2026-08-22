import csv
from collections import Counter

def run(payload: dict) -> dict:
    column = payload.get("column")
    csv_text = payload.get("csv_text")
    
    if not csv_text or not column:
        raise ValueError("Missing required fields: column or csv_text")
    
    # Split the CSV text into lines
    lines = csv_text.strip().split("\n")
    
    # Parse the header to get column indices
    if not lines:
        raise ValueError("Empty CSV text")
    
    header = lines[0].split("\t") if "\t" in lines[0] else lines[0].split(",")
    
    # Find the index of the specified column
    try:
        col_index = header.index(column)
    except ValueError:
        raise ValueError(f"Column '{column}' not found in CSV")
    
    # Extract values from the specified column
    values = []
    for line in lines[1:]:
        row = line.split("\t") if "\t" in line else line.split(",")
        if len(row) > col_index:
            values.append(row[col_index])
    
    # Compute the mode (most common value)
    if not values:
        raise ValueError("No data found in the specified column")
    
    value_counts = Counter(values)
    most_common_value = value_counts.most_common(1)[0][0]
    
    return {"most_common_value": most_common_value}]}{