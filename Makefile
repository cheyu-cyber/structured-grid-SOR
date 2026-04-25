CC       ?= gcc
NVCC     ?= nvcc
CFLAGS   ?= -O1 -std=gnu11
NVCCFLAGS?= -O2
LDFLAGS  ?= -lrt -lm

OMPFLAGS ?= -fopenmp
PTFLAGS  ?= -pthread

SRCDIR   := src
BINDIR   := build

BINS     := $(BINDIR)/sor2d_cpu $(BINDIR)/sor3d_cpu \
            $(BINDIR)/sor2d_omp $(BINDIR)/sor3d_omp \
            $(BINDIR)/sor2d_pth $(BINDIR)/sor2d_rb \
            $(BINDIR)/sor2d_pth_decomp $(BINDIR)/sor3d_omp_part

GPU_BINS := $(BINDIR)/sor2d_gpu $(BINDIR)/sor3d_gpu

all: $(BINS)
gpu: $(GPU_BINS)

$(BINDIR)/sor2d_cpu: $(SRCDIR)/sor2d_cpu.c | $(BINDIR)
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

$(BINDIR)/sor3d_cpu: $(SRCDIR)/sor3d_cpu.c | $(BINDIR)
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

$(BINDIR)/sor2d_omp: $(SRCDIR)/sor2d_omp.c | $(BINDIR)
	$(CC) $(CFLAGS) $(OMPFLAGS) $< -o $@ $(LDFLAGS) $(OMPFLAGS)

$(BINDIR)/sor3d_omp: $(SRCDIR)/sor3d_omp.c | $(BINDIR)
	$(CC) $(CFLAGS) $(OMPFLAGS) $< -o $@ $(LDFLAGS) $(OMPFLAGS)

$(BINDIR)/sor2d_pth: $(SRCDIR)/sor2d_pth.c | $(BINDIR)
	$(CC) $(CFLAGS) $(PTFLAGS) $< -o $@ -lpthread $(LDFLAGS)

$(BINDIR)/sor2d_rb: $(SRCDIR)/sor2d_rb.c | $(BINDIR)
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

$(BINDIR)/sor2d_pth_decomp: $(SRCDIR)/sor2d_pth_decomp.c | $(BINDIR)
	$(CC) $(CFLAGS) $(PTFLAGS) $< -o $@ -lpthread $(LDFLAGS)

$(BINDIR)/sor3d_omp_part: $(SRCDIR)/sor3d_omp_part.c | $(BINDIR)
	$(CC) $(CFLAGS) $(OMPFLAGS) $< -o $@ $(LDFLAGS) $(OMPFLAGS)

$(BINDIR)/sor2d_gpu: $(SRCDIR)/sor2d_gpu.cu | $(BINDIR)
	$(NVCC) $(NVCCFLAGS) $< -o $@

$(BINDIR)/sor3d_gpu: $(SRCDIR)/sor3d_gpu.cu | $(BINDIR)
	$(NVCC) $(NVCCFLAGS) $< -o $@

$(BINDIR):
	mkdir -p $(BINDIR)

# Quick correctness smoke test.
smoke: $(BINS)
	$(BINDIR)/sor2d_cpu 130 16 32 4
	$(BINDIR)/sor3d_cpu  34  8 16 2
	$(BINDIR)/sor2d_pth 130 16 4
	$(BINDIR)/sor2d_rb  130
	$(BINDIR)/sor2d_pth_decomp 130 16 --threads 4 --mode strip       --sched persistent
	$(BINDIR)/sor2d_pth_decomp 130 16 --threads 4 --mode interleaved --sched persistent
	$(BINDIR)/sor2d_pth_decomp 130 16 --threads 4 --mode block       --sched persistent
	$(BINDIR)/sor2d_pth_decomp 130 16 --threads 4 --mode strip       --sched spawn
	OMP_NUM_THREADS=4 $(BINDIR)/sor3d_omp_part 34 8 --mode slab
	OMP_NUM_THREADS=4 $(BINDIR)/sor3d_omp_part 34 8 --mode pencil
	OMP_NUM_THREADS=4 $(BINDIR)/sor3d_omp_part 34 8 --mode cube --block 16

# Big-grid sweep harness.  Drives every binary at multiple (N, threads, mode)
# and writes one row per run to results/results.csv.
bench: $(BINS)
	bash scripts/sweep.sh

bench-quick: $(BINS)
	bash scripts/sweep.sh --quick

# Render the four canonical charts from the sweep CSV.  Uses the venv at
# $(HOME)/.venv by default (override: PYTHON=/path/to/python make plots).
PYTHON ?= $(HOME)/.venv/bin/python
plots:
	$(PYTHON) scripts/plot.py

# Omega-sweep CSV: plot "build/omega_sweep.csv" columns 1 vs 2 for the
# classic Lab-5 U-curve of iterations to converge vs. omega.
omega: $(BINDIR)/sor2d_rb
	$(BINDIR)/sor2d_rb 128 --sweep > $(BINDIR)/omega_sweep.csv

clean:
	rm -rf $(BINDIR)

.PHONY: all gpu smoke omega bench bench-quick plots clean
