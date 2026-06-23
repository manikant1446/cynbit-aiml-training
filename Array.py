import numpy as np

my_array = np.array([1, 2, 3, 4, 5])
print("NumPy Array:", my_array)

print("\nBasic Operations on NumPy Array")

print("Addition:", my_array + 10)
print("Subtraction:", my_array - 2)
print("Multiplication:", my_array * 3)
print("Division:", my_array / 2)

array_a = np.array([1, 2, 3])
array_b = np.array([10, 20, 30])

sum_array = array_a + array_b
print("Addition of two arrays:", sum_array)

mult_array = array_a * 5
print("Array multiplied by 5:", mult_array)

# Basic Math/Stats functions jo NumPy easy bana deta hai
print("\n---  Basic Stats ---")
print("Max value in array_b:", array_b.max())
print("Min value in array_a:", array_a.min())
print("Mean (Average) of array_b:", array_b.mean())