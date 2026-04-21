CC       ?= gcc
NVCC     ?= nvcc
CFLAGS   ?= -O3 -march=native -std=c11 -Wall -Wextra
NVCCFLAGS?= -O3
LDFLAGS  ?= -lm

OMPFLAGS ?= -fopenmp

SRCDIR   := src
BINDIR   := build

BINS     := $(BINDIR)/sor2d_cpu $(BINDIR)/sor3d_cpu \
            $(BINDIR)/sor2d_omp $(BINDIR)/sor3d_omp

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

$(BINDIR)/sor2d_gpu: $(SRCDIR)/sor2d_gpu.cu | $(BINDIR)
	$(NVCC) $(NVCCFLAGS) $< -o $@

$(BINDIR):
	mkdir -p $(BINDIR)

# Quick correctness smoke test: small grid, small iters, small tile.
smoke: $(BINS)
	$(BINDIR)/sor2d_cpu 130 16 32 4
	$(BINDIR)/sor3d_cpu  34  8 16 2

clean:
	rm -rf $(BINDIR)

.PHONY: all smoke clean
