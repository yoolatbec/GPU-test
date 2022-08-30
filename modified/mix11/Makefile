CC = nvcc
SRC = kernel.cu

all: mix11 mix11_s
debug: mix11_g mix11_sg

mix11: $(SRC)
	$(CC) $(SRC) -o $@

mix11_s: $(SRC)
	$(CC) $(SRC) -o $@ -DSEPERATE

mix11_g: $(SRC)
	$(CC) $(SRC) -G -o $@

mix11_sg: $(SRC)
	$(CC) $(SRC) -G -o $@ -DSEPERATE

.PHONY: run run_inv clean
run: mix11 mix11_s
	-./mix11
	-./mix11_s

run_inv: mix11 mix11_s
	-./mix11_s
	-./mix11

clean:
	-rm mix11
	-rm mix11_s
	-rm mix11_g
	-rm mix11_sg