# caffeinate -d -i python src/clamp_irradiance.py

import pandas as pd

# 1) Load irradiance.csv previously generated
irr = pd.read_csv("data/processed/irradiance.csv")

# 2) Count how many negative values exist (optional check)
print("Negative GHI:", (irr["ghi"] < 0).sum())
print("Negative DNI:", (irr["dni"] < 0).sum())
print("Negative DHI:", (irr["dhi"] < 0).sum())

# 3) Clamp: replace any value < 0 with 0
irr["ghi"] = irr["ghi"].clip(lower=0)
irr["dni"] = irr["dni"].clip(lower=0)
irr["dhi"] = irr["dhi"].clip(lower=0)

# 4) Save clamped irradiance CSV
irr.to_csv("data/processed/irradiance_clamped.csv", index=False)
print("Negative irradiance values replaced with zero.")

