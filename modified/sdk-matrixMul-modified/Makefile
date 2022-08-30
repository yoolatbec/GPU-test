CC = nvcc
SRC = cuda_mult.cu

all: mult mult_s
debug: mult_g mult_sg

mult: $(SRC)
	$(CC) $(SRC) -o $@

mult_s: $(SRC)
	$(CC) $(SRC) -o $@ -DSEPERATE

mult_g: $(SRC)
	$(CC) $(SRC) -G -o $@

mult_sg: $(SRC)
	$(CC) $(SRC) -G -o $@ -DSEPERATE

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
	-rm mult_g
	-rm mult_sg