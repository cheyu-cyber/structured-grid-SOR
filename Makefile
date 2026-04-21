CC       ?= gcc
NVCC     ?= nvcc
CFLAGS   ?= -O3 -march=native -std=c11 -Wall -Wextra
NVCCFLAGS?= -O3
LDFLAGS  ?= -lm

OMPFLAGS ?= -fopenmp
PTFLAGS  ?= -pthread

SRCDIR   := src
BINDIR   := build

BINS     := $(BINDIR)/sor2d_cpu $(BINDIR)/sor3d_cpu \
            $(BINDIR)/sor2d_omp $(BINDIR)/sor3d_omp \
            $(BINDIR)/sor2d_pth $(BINDIR)/sor2d_rb

GPU_BINS := $(BINDIR)/sor2d_gpu

all: $(BINS)
gpu: $(GPU_BINS)

$(BINDIR)/sor2d_cpu: $(SRCDIR)/sor2d_cpu.c $(SRCDIR)/common.h | $(BINDIR)
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

$(BINDIR)/sor3d_cpu: $(SRCDIR)/sor3d_cpu.c $(SRCDIR)/common.h | $(BINDIR)
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

$(BINDIR)/sor2d_omp: $(SRCDIR)/sor2d_omp.c $(SRCDIR)/common.h | $(BINDIR)
	$(CC) $(CFLAGS) $(OMPFLAGS) $< -o $@ $(LDFLAGS) $(OMPFLAGS)

$(BINDIR)/sor3d_omp: $(SRCDIR)/sor3d_omp.c $(SRCDIR)/common.h | $(BINDIR)
	$(CC) $(CFLAGS) $(OMPFLAGS) $< -o $@ $(LDFLAGS) $(OMPFLAGS)

$(BINDIR)/sor2d_pth: $(SRCDIR)/sor2d_pth.c $(SRCDIR)/common.h | $(BINDIR)
	$(CC) $(CFLAGS) $(PTFLAGS) $< -o $@ $(LDFLAGS)

$(BINDIR)/sor2d_rb: $(SRCDIR)/sor2d_rb.c $(SRCDIR)/common.h | $(BINDIR)
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

$(BINDIR)/sor2d_gpu: $(SRCDIR)/sor2d_gpu.cu | $(BINDIR)
	$(NVCC) $(NVCCFLAGS) $< -o $@

$(BINDIR):
	mkdir -p $(BINDIR)

# Quick correctness smoke test.
smoke: $(BINS)
	$(BINDIR)/sor2d_cpu 130 16 32 4
	$(BINDIR)/sor3d_cpu  34  8 16 2
	$(BINDIR)/sor2d_pth 130 16 4
	$(BINDIR)/sor2d_rb  130

# Omega-sweep CSV: plot "build/omega_sweep.csv" columns 1 vs 2 for the
# classic Lab-5 U-curve of iterations to converge vs. omega.
omega: $(BINDIR)/sor2d_rb
	$(BINDIR)/sor2d_rb 128 --sweep > $(BINDIR)/omega_sweep.csv

clean:
	rm -rf $(BINDIR)

.PHONY: all gpu smoke omega clean
