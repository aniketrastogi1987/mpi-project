from mpi4py import MPI
import numpy as np
import time
import sys

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

N = int(sys.argv[1]) if len(sys.argv) > 1 else 1024  # You can change this for testing

# Only the root process initializes the matrices
if rank == 0:
    print(f"Generating two random {N}x{N} matrices...")
    A = np.random.rand(N, N)
    B = np.random.rand(N, N)
else:
    A = None
    B = None

# Scatter rows of A to all processes
rows_per_proc = N // size
if rank == size - 1:
    # Last process may take extra rows if N is not divisible by size
    rows_per_proc = N - (size - 1) * (N // size)

local_A = np.zeros((rows_per_proc, N), dtype='float64')

# Prepare counts and displacements for uneven scatter
counts = [(N // size) * N] * size
counts[-1] = (N - (size - 1) * (N // size)) * N
displs = [sum(counts[:i]) for i in range(size)]

if rank == 0:
    comm.Scatterv([A, counts, displs, MPI.DOUBLE], local_A, root=0)
else:
    comm.Scatterv([None, counts, displs, MPI.DOUBLE], local_A, root=0)

# Broadcast B to all processes
if rank != 0:
    B = np.empty((N, N), dtype='float64')
comm.Bcast(B, root=0)

# Each process computes its part
comm.Barrier()  # Synchronize before timing
start = time.time()
local_C = np.dot(local_A, B)
end = time.time()
local_time = end - start

# Gather the results
if rank == 0:
    C = np.empty((N, N), dtype='float64')
else:
    C = None

comm.Gatherv(local_C, [C, counts, displs, MPI.DOUBLE], root=0)

# Gather timing info
times = comm.gather(local_time, root=0)

if rank == 0:
    print(f"Distributed multiplication with {size} processes took {max(times):.4f} seconds")