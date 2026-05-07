# Run single-GPU
`main.py` contains the code for single-GPU (or CPU) operation. The EA can be enabled, or a manual genotype can be used for full training

# Run multi-GPU
**NOTE**: The multi-GPU code has NOT been tested recently due to unavailability of multiple GPUs on the HPC

Run the following command from the root directory to run the multi-GPU training script:

    torchrun --nnodes=1 --nproc-per-node=2 ./train_multi_gpu.py

If you want to use more than 2 GPUs, change the `--nproc-per-node=<n>` parameter.

# Notes
- **NOTE**: Change the GPU numbers before running the script to prevent hijacking someone else's GPU!
- The single GPU training file does not work as of now (unless necessary in the future, I will not fix that).

