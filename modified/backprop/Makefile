CC = nvcc
SRC = backprop_cuda.cu backprop.c facetrain.c imagenet.c

all: backprop backprop_s

backprop: $(SRC) *.h
	$(CC) $(SRC) -o $@

backprop_s: $(SRC) *.h
	$(CC) $(SRC) -o $@ -DSEPERATE

.PHONY: run run_inv clean
run: backprop backprop_s
	-./backprop 480000
	-./backprop_s 480000

run_inv: backprop backprop_s
	-./backprop_s 480000
	-./backprop 480000

clean:
	-rm backprop
	-rm backprop_s