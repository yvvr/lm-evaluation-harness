
# #!/bin/bash

# for gpuid in `seq 1 7`
# do
#    echo $gpuid
#   python kgr.py $gpuid > /dev/null 2>&1 &
# done

import torch
import time
import sys

gpuid = sys.argv[1]

#python kgr_infer_matmul.py $gpuid > /dev/null 2>&1 &

# Check if CUDA is available
device = torch.device(f"cuda:{gpuid}" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create two random 1024x1024 matrices on the GPU
A = torch.randn(1024, 1024, device=device)
B = torch.randn(1024, 1024, device=device)

# Number of iterations
num_iters = 100

# Warm-up (optional for more accurate timing)
_ = torch.matmul(A, B)

# Time the loop
start = time.time()

# for i in range(num_iters):
while True:
    C = torch.matmul(A, B)

# Synchronize to ensure all GPU operations are done
torch.cuda.synchronize()

end = time.time()

print(f"Completed {num_iters} multiplications in {end - start:.4f} seconds.")