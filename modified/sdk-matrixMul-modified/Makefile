CC = nvcc
SRC = cuda_mult.cu

all: mult mult_s

mult: $(SRC)
	$(CC) $(SRC) -o $@

mult_s: $(SRC)
	$(CC) $(SRC) -o $@ -DSEPERATE

.PHONY: run
run: mult mult_s
	-./mult 4096
	-./mult_s 4096