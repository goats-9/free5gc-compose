import random
import os
random.seed(42)  # Set a seed for reproducibility

N = 5000

# Create directory if it doesn't exist
os.makedirs('ftp', exist_ok=True)

for i in range(1, N + 1):
    # Print i random kilobytes of data to a file named 'file_i.txt'
    with open(f'ftp/file_{i}.txt', 'wb') as f:
        f.write(random.randbytes(1024 * i))