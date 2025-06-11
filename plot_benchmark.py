import pandas as pd
import matplotlib.pyplot as plt

# Read the benchmark data
benchmark_file = 'benchmark.csv'
df = pd.read_csv(benchmark_file)

# Plot 1: Execution Time vs Matrix Size
plt.figure(figsize=(10, 6))
serial = df[df['type'] == 'serial']
plt.plot(serial['N'], serial['time'], marker='o', label='Serial', color='black')

for p in sorted(df[df['type'] == 'mpi']['processes'].unique()):
    mpi = df[(df['type'] == 'mpi') & (df['processes'] == p)]
    plt.plot(mpi['N'], mpi['time'], marker='o', label=f'MPI ({p} procs)')

plt.xlabel('Matrix Size (N x N)')
plt.ylabel('Execution Time (s)')
plt.title('Execution Time vs Matrix Size')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('execution.png')
plt.close()

# Plot 2: Speedup vs Matrix Size (for each process count and serial)
plt.figure(figsize=(10, 6))
# Serial speedup is always 1.0
plt.plot(serial['N'], [1.0]*len(serial), marker='o', linestyle='--', color='black', label='Serial (speedup=1.0)')
for p in sorted(df[df['type'] == 'mpi']['processes'].unique()):
    mpi = df[(df['type'] == 'mpi') & (df['processes'] == p)]
    plt.plot(mpi['N'], mpi['speedup'], marker='o', label=f'MPI ({p} procs)')

plt.xlabel('Matrix Size (N x N)')
plt.ylabel('Speedup (Serial Time / MPI Time)')
plt.title('Speedup vs Matrix Size')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('speedup.png')
plt.close()

# Plot 3: Efficiency vs Matrix Size (for each process count and serial)
plt.figure(figsize=(10, 6))
# Serial speedup is always 1.0
plt.plot(serial['N'], [1.0]*len(serial), marker='o', linestyle='--', color='black', label='Serial (speedup=1.0)')
for p in sorted(df[df['type'] == 'mpi']['processes'].unique()):
    mpi = df[(df['type'] == 'mpi') & (df['processes'] == p)]
    plt.plot(mpi['N'], mpi['efficiency'], marker='o', label=f'MPI ({p} procs)')

plt.xlabel('Matrix Size (N x N)')
plt.ylabel('Efficiency (SpeedUp / No. of Processes)')
plt.title('Efficiency vs Matrix Size')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('efficiency.png')
plt.close()



print('Plots saved as execution_time_vs_matrix_size.png and speedup_vs_matrix_size.png') 