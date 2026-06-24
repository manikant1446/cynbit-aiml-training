import pandas as pd

data = {
    "Name": ["Manikant", "Nimisha", "kanika", "Kritika", "Oman"],
    "Age": [19, 20, 18, 21, 19],
    "Marks": [85, 78, 92, 74, 88]
}
df = pd.DataFrame(data)

df.to_csv("students.csv", index=False)

loaded_df = pd.read_csv("students.csv")

print("Head of Dataset:")
print(loaded_df.head())