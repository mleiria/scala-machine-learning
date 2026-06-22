import os
import csv
import re
from datetime import datetime

def clean_header(header):
    # Remove prefixes like com.samsung.health.exercise.
    # Look for the last dot that is followed by a name
    match = re.search(r'\.([a-z0-9_]+)$', header)
    if match:
        return match.group(1)
    # Also handle the case where there is no prefix but it's just a name
    return header.replace('.', '_')

def infer_type(values):
    non_empty = [v for v in values if v and v.strip()]
    if not non_empty:
        return "String"
    
    # Try LocalDateTime
    try:
        for v in non_empty:
            datetime.strptime(v.strip(), '%Y-%m-%d %H:%M:%S.%f')
        return "java.time.LocalDateTime"
    except ValueError:
        pass
    
    # Try Int/Long
    try:
        for v in non_empty:
            int(v.strip())
        return "Long" if any(len(v.strip()) > 9 for v in non_empty) else "Int"
    except ValueError:
        pass
    
    # Try Double
    try:
        for v in non_empty:
            float(v.strip())
        return "Double"
    except ValueError:
        pass
    
    return "String"

def get_class_name(filename):
    # com.samsung.shealth.tracker.pedometer_step_count.20250930125884.csv
    parts = filename.split('.')
    # Remove 'com', 'samsung', 'shealth', 'health' and the timestamp at the end
    meaningful = [p for p in parts if p not in ('com', 'samsung', 'shealth', 'health', 'csv')]
    # The last part is usually the timestamp, remove it if it's numeric
    if meaningful and meaningful[-1].isdigit():
        meaningful.pop()
    
    # Join remaining and camelcase
    name = "_".join(meaningful)
    return "".join(word.capitalize() for word in name.split('_'))

csv_dir = '/home/manuel/Downloads/Databases/samsunghealth_manuel.leiria_20250930125884/csv'
files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]

for f in sorted(files):
    path = os.path.join(csv_dir, f)
    with open(path, 'r', encoding='utf-8', errors='ignore') as csvfile:
        lines = csvfile.readlines()
        if len(lines) < 2:
            continue
        
        # First line is metadata, second is header
        header_line = lines[1].strip()
        # Handle potential BOM or leading chars
        if header_line.startswith('﻿'):
            header_line = header_line[1:]
            
        headers = header_line.split(',')
        
        # Sample data from the 3rd line onwards
        data_rows = []
        for line in lines[2:12]:
            data_rows.append(line.strip().split(','))
        
        class_name = get_class_name(f)
        print(f"case class {class_name}(")
        
        fields = []
        for i in range(len(headers)):
            h = headers[i]
            clean_h = clean_header(h)
            
            # Get values for this column across sampled rows
            col_values = [row[i] if i < len(row) else "" for row in data_rows]
            t = infer_type(col_values)
            
            # Use Option for types that have many empties in sample
            empty_count = col_values.count("")
            if empty_count > 0:
                fields.append(f"  {clean_h}: Option[{t}]")
            else:
                fields.append(f"  {clean_h}: {t}")
        
        print(",\n".join(fields))
        print(")")
        print()

