def add(a, b):
    return a + b
def subtract(a, b):
    return a - b
def multiply(a, b):
    return a * b
def divide(a, b):
    if b == 0:
        return "Error! Division by zero."
    else:
        return a / b

a=float(input("Enter first number: "))
b=float(input("Enter second number: "))

print("Select operation:")
print("1. Add")
print("2. Subtract")
print("3. Multiply")
print("4. Divide")
choice=input("Enter choice(1/2/3/4): ")

if choice == '1':
    print("Addition:", add(a, b))
elif choice == '2':
    print("Subtraction:", subtract(a, b))
elif choice == '3':
    print("Multiplication:", multiply(a, b))
elif choice == '4':
    print("Division:", divide(a, b))
else:
    print("Invalid input")