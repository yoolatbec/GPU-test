CC = nvcc
SRC = kernel.cu

all: mix11 mix11_s
mix11: $(SRC)
	$(CC) $(SRC) -o $@

mix11_s: $(SRC)
	$(CC) $(SRC) -o $@ -DSEPERATE

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