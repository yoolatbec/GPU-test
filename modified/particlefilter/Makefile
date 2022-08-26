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

.PHONY: run run_naive run_inv run_naive_inv clean
run: float float_s
	-./float -x 100 -y 100 -z 100 -np 1000
	-./float_s -x 100 -y 100 -z 100 -np 1000

run_naive: naive naive_s
	-./naive -x 100 -y 100 -z 100 -np 1000
	-./naive_s -x 100 -y 100 -z 100 -np 1000

run_inv: float float_s
	-./float_s -x 100 -y 100 -z 100 -np 1000
	-./float -x 100 -y 100 -z 100 -np 1000

run_naive_inv: naive naive_s
	-./naive_s -x 100 -y 100 -z 100 -np 1000
	-./naive -x 100 -y 100 -z 100 -np 1000

clean:
	-rm float
	-rm float_s
	-rm naive
	-rm naive_s