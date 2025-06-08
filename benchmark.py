import subprocess
import csv
import sys

def get_user_input():
    # Use matrix sizes 2^2 to 2^10 by default
    matrix_sizes = [2 ** i for i in range(2, 11)]
    print(f"Default matrix sizes for benchmarking: {matrix_sizes}")

    # Get max process count for MPI (up to 8)
    max_procs = input("Enter the maximum number of processes for MPI (2-8): ")
    try:
        max_procs = int(max_procs)
        if max_procs < 2:
            print("Minimum process count is 2. Using 2.")
            max_procs = 2
        elif max_procs > 8:
            print("Maximum process count is 8. Using 8.")
            max_procs = 8
    except ValueError:
        print("Invalid input. Using 2 processes as default.")
        max_procs = 2
    process_counts = list(range(2, max_procs + 1))
    print(f"Process counts for MPI: {process_counts}")
    return matrix_sizes, process_counts

def run_serial(N):
    print(f"Running serial for N={N}")
    result = subprocess.run(
        ["python3", "serial_matrix_mult.py", str(N)],
        capture_output=True, text=True
    )
    for line in result.stdout.splitlines():
        if "Serial multiplication took" in line:
            time_taken = float(line.split()[-2])
            return time_taken
    return None

def run_mpi(N, p):
    print(f"Running MPI for N={N} with {p} processes")
    result = subprocess.run(
        ["mpiexec", "-n", str(p), "python3", "mpi_matrix_mult.py", str(N)],
        capture_output=True, text=True
    )
    for line in result.stdout.splitlines():
        if "Distributed multiplication with" in line:
            time_taken = float(line.split()[-2])
            return time_taken
    return None

def main():
    # Get user input
    matrix_sizes, process_counts = get_user_input()

    results = []
    serial_times = {}

    # Serial benchmarks
    for N in matrix_sizes:
        time_taken = run_serial(N)
        if time_taken is not None:
            serial_times[N] = time_taken
            results.append({
                "type": "serial",
                "N": N,
                "processes": 1,
                "time": time_taken,
                "speedup": 1.0
            })

    # MPI benchmarks
    for N in matrix_sizes:
        for p in process_counts:
            time_taken = run_mpi(N, p)
            if time_taken is not None:
                speedup = serial_times[N] / time_taken if time_taken > 0 else None
                results.append({
                    "type": "mpi",
                    "N": N,
                    "processes": p,
                    "time": time_taken,
                    "speedup": speedup
                })

    # Save to CSV
    with open("benchmark.csv", "w", newline="") as csvfile:
        fieldnames = ["type", "N", "processes", "time", "speedup"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print("Benchmarking complete. Results saved to benchmark.csv.")

if __name__ == "__main__":
    main()