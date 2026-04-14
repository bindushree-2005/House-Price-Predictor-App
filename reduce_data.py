import pandas as pd

data = pd.read_csv("house_data.csv")

# take small sample
small_data = data.sample(5000)

small_data.to_csv("small_data.csv", index=False)

print("Done!")