import pandas as pd

data = {
    "Name": ["Aman", "Riya", "Karan", None, "Riya"],
    "Age": [20, 21, None, 22, 21],
    "Marks": [85, 90, 78, None, 90]
}

df = pd.DataFrame(data)

print("Original Dataset:")
print(df)

print("\nMissing Values:")
print(df.isnull().sum())

df = df.dropna()
df = df.drop_duplicates()

print("\nCleaned Dataset:")
print(df)

df.to_csv("cleaned_dataset.csv", index=False)

print("\nCleaned dataset saved as 'cleaned_dataset.csv'")