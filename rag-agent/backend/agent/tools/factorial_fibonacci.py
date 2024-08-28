import math
import sys
import numpy as np

sys.set_int_max_str_digits(50000)

''' 
Matrix Fibonacci:
Matrix exponentiation is a highly efficient method for computing Fibonacci numbers 
in logarithmic time, making it ideal for very large n.
https://www.geeksforgeeks.org/matrix-exponentiation/
https://www.youtube.com/watch?v=eMXNWcbw75E
'''
def get_matrix_fibonacci(n):
    def matrix_mult(A, B):
        return np.dot(A, B)
    
    def matrix_power(matrix, power):
        result = np.identity(len(matrix), dtype=object)
        while power:
            if power % 2:
                result = matrix_mult(result, matrix)
            matrix = matrix_mult(matrix, matrix)
            power //= 2
        return result
    
    if n == 0:
        return 0
    if n == 1:
        return 1
    
    F = np.array([[1, 1], [1, 0]], dtype=object)
    result_matrix = matrix_power(F, n - 1)
    return result_matrix[0][0]

# n = 10000
# result = get_matrix_fibonacci(n)
# print(f"Fibonacci number {n} has {len(str(result))} digits.")


def get_factorial(n):

    return math.factorial(n)


# if __name__ == '__main__':
#     n = 1000
#     result = large_factorial(n)
#     print(f"Factorial of {n} has {len(str(result))} digits.")
#     print(f'Factorial of {n} is equal:\n{result}')

    