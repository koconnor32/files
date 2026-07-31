import pandas as pd

df = pd.read_csv("/scratch.global/ocon0444/peat_modeling/00_data/processed/peat_depths_with_covariates_v2_fixed.csv", low_memory=False)

# One-hot encode the new variable
dummies = pd.get_dummies(df['MN_organic_soils_classified_FIXED'], prefix='MN_organic_soils_classified_FIXED', dtype=int)
df = pd.concat([df, dummies], axis=1)
df = df.drop(columns=['MN_organic_soils_classified_FIXED'])

# Save
df.to_csv("/scratch.global/ocon0444/peat_modeling/00_data/processed/peat_depths_processed_v2.csv", index=False)
print(f"Done! Shape: {df.shape}")
