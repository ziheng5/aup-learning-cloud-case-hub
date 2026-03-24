import pandas as pd
import json
import os

# Read JSONL file
data = []
with open('data_collection/processed_data/training_data.jsonl', 'r') as f:
    for line in f:
        data.append(json.loads(line))

# Convert to DataFrame
df = pd.DataFrame(data)

# Save as Parquet
output_path = 'data_collection/processed_data/training_data.parquet'
df.to_parquet(output_path, index=False)

print(f"Data converted to Parquet format: {output_path}")
print(f"Number of training samples: {len(df)}")
