
students=[
    {"ID": "96", "name": "Manikant Singh", "age": 20, "branch": "CSE"},
    {"ID": "97", "name": "Manish Kumar", "age": 19, "branch": "CSE"},
    {"ID": "95", "name": "Manana Gupta", "age": 20, "branch": "CSE"}
]
print("Student Record System")
print("------------------------")
for student in students:
    print("ID:", student["ID"])
    print("Name:", student["name"])
    print("Age:", student["age"])
    print("Branch:", student["branch"])
    print()