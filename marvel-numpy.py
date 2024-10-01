# Using Numpy Generate an array by repeating a small array across each dimension and 
# Generate an array with element indexes such that the array elements appear in ascending order.

import numpy as np

arr = np.array([[1,2],[3,4]])
print(arr)
x = np.repeat(arr,2,axis=0) # Repeats array 3 times 
print("Repeated array: \n", x,"\n")

arr2 = np.array([1,7,3,4,5])
print(arr2)
y = np.argsort(arr2)
print("Array sorted by indices: ",y)