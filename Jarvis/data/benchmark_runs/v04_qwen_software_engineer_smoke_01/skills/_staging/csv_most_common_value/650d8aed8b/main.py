import csv
import io

def run(payload: dict) -> dict:
    column = payload.get("column")
    csv_text = payload.get("csv_text")
    
    if not csv_text or not column:
        raise ValueError("Missing required fields: column or csv_text")
    
    # Parse CSV text using io.StringIO to simulate file input
    try:
        csv_file = io.StringIO(csv_text)
        reader = csv.DictReader(csv_file)
        
        # Extract values from the specified column and convert to int
        values = []
        for row in reader:
            value = row.get(column)
            if value is not None:
                # Convert string value to integer (assumed to be numeric based on test cases)
                values.append(int(value))
        
        if not values:
            raise ValueError(f"No data found in column '{column}'")
        
        # Count frequencies
        frequency = {}
        for val in values:
            frequency[val] = frequency.get(val, 0) + 1
        
        # Find the most common value
        most_common_value = max(frequency, key=frequency.get)
        
        return {"most_common_value": most_common_value}
    except Exception as e:
        raise ValueError(f"Error processing CSV: {str(e)}")