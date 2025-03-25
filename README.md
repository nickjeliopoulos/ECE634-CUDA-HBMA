<div align="center">

# ECE634 Final Project - CUDA C++ Hierarchical Block Matching (HBMA)
</div>

<div align="center">

Efficient implementation of HBMA algorithm designed and optimized for an XXX GPU.
A naive kernel `hbma_v0` was first implemented, with iterative improvements `v1`, `v2`, etc. to illustrate various design decisions and their resulting speedups.

</div>

<!-- Installation Guide -->
# Installation 
```bash
python -m pip install -r requirements.txt
cd hbma/ops
python setup.py develop
```

<!-- Usage Guide -->
# Usage 
> [!CAUTION]
> To best reproduce latency measurements, we encourage users to lock the clock and/or memory rates of their device.

## benchmark.py
```bash
python benchmark.py --anchor-image-path im1k_0.jpg --target-image-path im1k_8.jpg
```

# Project Goals and Deliverables
**Main goal:** our implementation in CUDA is faster (lower latency) than a CPU implementation across various HBMA configurations.

### Design Decisions:
* How should we partition workload on the GPU (multiprocessing cores, threadblocks, and threads)?
* How should we utilize shared memory to promote better memory traffic patterns?
* How should we utilize design paradigms (warp-tiling, thread-swizzling) to promote better performance?

### Evaluations
We will implement various versions of our kernel to illustrate the effect of design decisions on performance.
* **V1:** Naive, no manual or intentional optimization
* **V2:** Redesign V1 to take advantage of shared memory, other 
* **V3:** Autotune V2 kernel parameters across GPU's:
  * NVIDIA RTX 3090Ti
  * NVIDIA AGX Orin Developer Board,
  * NVIDIA A100

**What's the baseline(s)?**
* PyTorch CPU implementation (Done in Nick's `project-1`)
* PyTorch GPU implementation (Done in Nick's `project-1`)
* (Stretch Goal) PyTorch GPU + torch.compile

**Tables + Figures:**
* Table 1: **Main latency results table.** V1-V3 latency vs Torch CPU, Torch GPU, Torch GPU + Compile across devices and HBMA parameterizations.
* Table 2: **V2 shared memory evaluation.** Quantify memory traffic improvement with profiling tools (cache hit rate, number of loads to/from global memory).
* Table 3: **V3 autotune evaluation.** Plot grid and threadblock sizes of V3 vs. latency, # instructions 
* Table 4: **Search Granualrity - Latency Equivalence.** For *similar latency* as the baselines, how much *finer* can our HBMA search be?

