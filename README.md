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
TODO
```

<!-- Usage Guide -->
# Usage 
> [!CAUTION]
> To best reproduce latency measurements, we encourage users to lock the clock and/or memory rates of their device.

## benchmark.py
```bash
TODO
```

# Project Goals and Deliverables
**Main goal:** our implementation in CUDA is faster (lower latency) than a CPU implementation on X device given Y image size

### Questions + Design Decisions:
1.  Given device X, what are the optimal thread / grid sizes for our CUDA HBMA?
2.  How can we optimize our HBMA to take advantage of (1) shared memory and (2) cache hit rate (overall memory traffic)
3.  After that, how can we optimize our HBMA to optimize ALU/TensorCore? pipeline utilization (overall compute)

### Evaluations
**Implement multiple versions of our HBMA kernel**
1.  Naive, no manual optimization for memory traffic
2.  Attempt to cache in SMEM, optimize memory traffic
3.  Fine-tune grid / block size for X GPU.
4.  ???

**What's the baseline(s)?**
1.  PyTorch CPU implementation
2.  Stretch: PyTorch GPU implementation 
3.  Mega Stretch: PyTorch GPU + torch.compile

**Tables + Figures:**
1.  Table 1: Version 1-3 latency vs Torch CPU, Torch GPU
2.  Table 2: Plot grid / threadblock sizes of Version 3 vs. Version 3 latency, # instructions
3.  Table / Figure: How many more stages can we get for equivalent latency with our HBMA to Torch CPU?
