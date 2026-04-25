# 18-646 Project: Build and Run Guide

This project has two executable paths:
- CPU baseline renderer (`main.cpp`, built as `build/tinyrenderer`)
- CUDA tiled rasterizer (`main.cu`, built as `main_<TILE_WIDTH>x<TILE_HEIGHT>.x`)

## Machine / Environment Notes

This project is run under ECE019 machine with the following CPU and GPU:
- CPU: 16-core Intel Xeon Silver 4208
- GPU: NVIDIA Tesla T4 (Turing architecture, 40 SMs, 2,560 CUDA cores, 16 GB GDDR6)

Recommended environment:
- CMU ECE/Andrew Linux machine with NVIDIA GPU access
- CUDA toolkit with `nvcc` available in `PATH`
- CMake (3.12+)
- C++ compiler supporting C++20 (for CPU path)
- OpenMP-capable compiler (optional but used when available)

## Input Model Naming

The code expects a base model name and loads files as:
- `<base_name>_16.obj`
- `<base_name>_32.obj`
- `<base_name>_64.obj`
- `<base_name>_128.obj`

In this repository, Utah teapot files are under:
- `./utah_teapot_16.obj`
- `./utah_teapot_32.obj`
- `./utah_teapot_64.obj`
- `./utah_teapot_128.obj`


## Build and Run: GPU (CUDA)

Build with tile size (default is 16x16):

```bash
make build_gpu TILE_WIDTH=32 TILE_HEIGHT=32
make run_gpu TILE_WIDTH=32 TILE_HEIGHT=32
```

Outputs:
- CSV timing file: `results_<tile>.csv` (example: `results_32x32.csv`)
- Images: `utah_teapot_results_tile_<tile>/res_*/gpu_out_*.tga`

## Build and Run: CPU Baseline

Build and run:

```bash
make build_cpu
make run_cpu
```

Outputs:
- CSV timing file: `results_baseline.csv`
- Images: `obj/utah_teapot/utah_teapot_scaled_results_baseline/res_*/cpu_out_*.tga`

## Compare GPU and CPU Outputs

1) Edit paths in `verify_rasterization.py` if needed.
2) Run:

```bash
python3 verify_rasterization.py
```

## Plot Timing Results

```bash
python3 plot_results.py
```

This generates summary plots from `results*.csv` files.

## Common Issues

- `nvcc: command not found`
  - Load CUDA module or use a machine with CUDA toolkit installed.

- `Model ... not found. Skipping resolution ...`
  - Ensure your base name points to files ending in `_16.obj`, `_32.obj`, `_64.obj`, `_128.obj`.

- No GPU visible in `nvidia-smi`
  - You are likely on a login node or non-GPU machine. Move to an ECE machine/node with GPU access.
