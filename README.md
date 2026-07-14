# Parallel Image Stitching

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-Image%20Processing-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

An exploration of parallelizing the Image Stitching pipeline in Python. This project evaluates multiple concurrency paradigms to overcome language-specific bottlenecks such as the Global Interpreter Lock (GIL) and Inter-Process Communication (IPC) serialization overhead, benchmarking both Task and Data Parallelism against a sequential baseline.


## Overview

Image stitching combines multiple images with overlapping fields of view to produce a high-resolution panorama. However, the standard sequential pipeline is computationally expensive and scales poorly with large image sets. 

This study targets and benchmarks the parallelization of key stages of the stitching process:
1. **Feature Detection & Description** (SIFT)
2. **Warping & Blending**

The code is evaluated under different concurrency architectures using the **OpenDroneMap Dataset** to analyze end-to-end speedups, phase-specific scalability and architectural trade-offs.


## Implemented Architectures

The repository contains six different architectural approaches to handle the stitching workload:

### 1. Sequential Baseline (`sequential.py`)
The traditional $O(n)$ linear-fold approach. Images are processed one by one, continuously matching, warping and blending onto a monotonically growing canvas.

### 2. Standard Multiprocessing (`parallel.py`)
Parallelizes SIFT extraction and matching across multiple processes using Python's native `ProcessPoolExecutor` to bypass the Global Interpreter Lock (GIL) for CPU-bound phases.

### 3. Joblib Pipeline (`joblib_pipeline.py`)
Utilizes the `joblib` library (with `loky` backend) to simplify process management and automatic memory mapping for large arrays.

### 4. Shared Memory Pipeline (`shared_memory_pipeline.py`)
Implements Python's `multiprocessing.shared_memory` to allow workers direct access to image arrays. This is designed to mitigate the heavy serialization/deserialization overhead associated with IPC.

### 5. Producer-Consumer (`producer_consumer.py`)
A Task Parallelism approach that decouples I/O bound tasks (loading/saving images) and CPU-bound tasks (SIFT and RANSAC calculations) using synchronized thread/process queues.

### 6. MapReduce Merge Tree (`mapreduce.py`)
Restructures the dependency chain from an $O(n)$ linear fold into an $O(\log_2 n)$ merge tree. Pairs of images are merged in parallel, completely sidestepping the sequential bottleneck of continuous feature re-extraction.


## Key Experimental Findings

Detailed performance metrics and analyses are available in the [Project Report](report.pdf). Key takeaways include:

### Phase-Specific Concurrency
* **SIFT Extraction (CPU-Bound):** highly embarrassingly parallel. Running it with a `ProcessPoolExecutor` reaches a peak speedup of $\approx 2.9\times$ at 4 cores, saturating thereafter due to hardware limits.
* **Warp & Blend (Memory-Bound):** since OpenCV and NumPy operations release the GIL, running this phase with a `ThreadPoolExecutor` scales almost linearly up to 8 threads, proving that hyper-threading is highly effective for memory-bound tasks.

### The Re-Extraction Bottleneck (Amdahl's Law)
Despite massive phase-specific speedups, the end-to-end speedup for all linear-fold pipelines never exceeds $\approx 1.5\times$. 
* **The Reason:** feature re-extraction on the growing canvas accounts for **≈ 65%** of the total execution time (**≈ 14.5s**) and cannot be parallelized due to strict sequential dependencies. This is a direct, quantitative confirmation of **Amdahl's Law**.

### 🔍 IPC Overhead vs. Computational Weight
The four linear-fold candidates (`parallel`, `joblib`, `shared_memory`, `producer_consumer`) perform within a narrow $1\%$ margin of each other. At the tested resolution, the heavy operations of **SIFT Extraction and continuous Feature Re-extraction** completely dominate the execution time. This immense computational weight means that the pipeline is heavily CPU-bound, rendering advanced IPC optimizations—like shared memory—virtually useless, as they provide no measurable performance advantage over the relatively microscopic communication overhead.

### The MapReduce Trade-Off
By avoiding the sequential bottleneck of repeatedly re-extracting features from a single, ever-growing canvas, **MapReduce** achieves the lowest average total time and the highest peak per-window speedup.
* **The Risk:** since it pairs images by index rather than verified spatial adjacency, geometric mismatches can stall the pipeline. It is the only pipeline capable of falling below the sequential baseline performance, making it a **high-variance, high-reward** paradigm that depends on the spatial coherence of the input sequence.


## 📂 Repository Structure

```text
├── dataset/                    # Directory containing input images
├── output/                     # Directory where the generated panoramas are saved
├── results/                    # Raw benchmarking data and execution times
├── plots/                      # Generated performance graphs and speedup curves
├── profiling_results/          # Detailed profiling logs and performance traces
├── src/                        # Main source code directory
│   ├── main.py                 # Entry point for executing the pipelines
│   ├── benchmark.py            # Automated testing suite for collecting performance metrics
│   ├── sequential.py           # Standard sequential baseline pipeline
│   ├── parallel.py             # Standard Multiprocessing baseline (ProcessPoolExecutor)
│   ├── joblib_pipeline.py      # Joblib-backed parallel pipeline
│   ├── shared_memory_pipeline.py # Shared memory optimized pipeline to reduce IPC
│   ├── producer_consumer.py    # Task-parallel producer-consumer model
│   └── mapreduce.py            # O(log n) tree-reduction pipeline
├── utils/                      # Helper scripts and automation utilities
│   ├── download_dataset.py     # Utility to automatically download and unpack the Aukerman dataset
│   ├── window_diagnostic.py    # Diagnostic tool to analyze sliding window connectivity and keypoint counts
│   ├── reorder_windows.py      # Script to pre-process and optimize the spatial adjacency order of image sequences
│   ├── plot_benchmark_speedup.py # Script to parse benchmark outputs and generate speedup scaling curves
│   └── calculate_time.py       # Helper to aggregate, average, and format raw timing data
├── report.pdf                  # Technical report and analysis
├── requirements.txt            # Package dependencies
└── README.md                   # Project documentation
```

## Getting Started

### Prerequisites
* Python 3.8+ (Classic or Free-threaded build)
* OpenCV, NumPy, Joblib

### Setup

You can set up this project using either the modern **`uv`** package manager (recommended, especially for testing No-GIL builds) or the standard **`venv` + `pip`** workflow.

#### Option A: Setup with `uv`
[`uv`](https://github.com/astral-sh/uv) is an ultra-fast project manager that makes it incredibly easy to test the project under experimental **Free-Threaded (No-GIL)** Python builds.

```bash
# 1. Clone the repository
git clone [https://github.com/davidelomba/ParallelFinalTermProject.git](https://github.com/davidelomba/ParallelFinalTermProject.git)
cd ParallelFinalTermProject

# 2. Create a virtual environment and install dependencies in one shot
uv venv
uv pip install -r requirements.txt

# (Optional) Want to test under a Free-Threaded (No-GIL) Python 3.13 environment?
uv python install 3.13t
uv venv -p 3.13t
uv pip install -r requirements.txt
```

#### Option B: Standard Setup (Classic `venv` + `pip`)
If you don't have `uv` installed, you can use the traditional Python workflow:

```bash
# 1. Clone the repository
git clone [https://github.com/davidelomba/ParallelFinalTermProject.git](https://github.com/davidelomba/ParallelFinalTermProject.git)
cd ParallelFinalTermProject

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

# 3. Install the required dependencies
pip install -r requirements.txt
```



### Execution


Run the individual scripts from your terminal:

```bash
# Run the sequential baseline
python src/sequential.py

# Run the standard multiprocessing approach (ProcessPoolExecutor)
python src/parallel.py

# Run the Joblib-backed pipeline
python src/joblib_pipeline.py

# Run the Shared Memory optimized pipeline
python src/shared_memory_pipeline.py

# Run the Task-Parallel approach (Producer-Consumer Queue)
python src/producer_consumer.py

# Run the MapReduce merge-tree architecture
python src/mapreduce.py

``` 