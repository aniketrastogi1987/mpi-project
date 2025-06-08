import numpy as np
import time
import sys

def serial_matmul(A, B):
    return np.dot(A, B)

if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 1024  # You can change this for testing
    print(f"Generating two random {N}x{N} matrices...")
    A = np.random.rand(N, N)
    B = np.random.rand(N, N)

    print("Starting serial matrix multiplication...")
    start = time.time()
    C = serial_matmul(A, B)
    end = time.time()
    print(f"Serial multiplication took {end - start:.4f} seconds")