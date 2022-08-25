CC = nvcc
SRC = kernel.cu

all: mix11 mix11_s
mix11: $(SRC)
	$(CC) $(SRC) -o $@

mix11_s: $(SRC)
	$(CC) $(SRC) -o $@ -DSEPERATE

.PHONY: run
run: mix11 mix11_s
	-./mix11
	-./mix11_s
