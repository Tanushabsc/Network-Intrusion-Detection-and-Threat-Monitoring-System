
import pandas as pd
import json

df = pd.read_csv("data/processed/test.csv")

# Take one real network record
sample = df.drop(columns=["label"]).iloc[0]

# Convert it into JSON
features = json.loads(sample.to_json())

print(json.dumps({"features": features}, indent=2))
