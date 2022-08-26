CC = nvcc
SRC = cuda_mult.cu

all: mult mult_s

mult: $(SRC)
	$(CC) $(SRC) -o $@

mult_s: $(SRC)
	$(CC) $(SRC) -o $@ -DSEPERATE

.PHONY: run run_inv clean
run: mult mult_s
	-./mult 4096
	-./mult_s 4096

run_inv: mult mult_s
	-./mult_s 4096
	-./mult 4096

clean:
	-rm mult
	-rm mult_s