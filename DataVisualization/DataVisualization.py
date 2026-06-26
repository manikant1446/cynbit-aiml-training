import pandas as pd
import matplotlib.pyplot as plt

data = {
    "Name": ["Manikant", "Nimisha", "kanika", "Kritika", "Oman"],
    "Age": [19, 20, 18, 21, 19],
    "Marks": [85, 78, 92, 74, 88]
}

df = pd.DataFrame(data)

#Bar Chart(Student Marks)
plt.figure(figsize=(10, 5))

plt.bar(df["Name"], df["Marks"], color=['#4C72B0', '#55A868', '#C44E52', '#8172B3', '#CCB974'])

plt.title('Student Marks Overview', fontsize=14)
plt.xlabel('Student Name', fontsize=12)
plt.ylabel('Marks', fontsize=12)

plt.savefig('bar_chart.png')
plt.show()

#Pie Chart(Marks Distribution)
plt.figure(figsize=(8, 8))

plt.pie(df["Marks"], labels=df["Name"], autopct='%1.1f%%', startangle=140)
plt.title('Marks Distribution Among Students', fontsize=14)

plt.savefig('pie_chart.png')
plt.show()