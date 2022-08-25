CC = nvcc

all: float float_s naive naive_s

float: ex_particle_CUDA_float_seq.cu
	$(CC) $< -o $@

float_s: ex_particle_CUDA_float_seq.cu
	$(CC) $< -o $@ -DSEPERATE

naive: ex_particle_CUDA_naive_seq.cu
	$(CC) $< -o $@

naive_s: ex_particle_CUDA_naive_seq.cu
	$(CC) $< -o $@ -DSEPERATE

.PHONY: run run_naive
run: float float_s
	-./float -x 100 -y 100 -z 100 -np 1000
	-./float_s -x 100 -y 100 -z 100 -np 1000

run_naive: naive naive_s
	-./naive -x 100 -y 100 -z 100 -np 1000
	-./naive_s -x 100 -y 100 -z 100 -np 1000