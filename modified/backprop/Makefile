CC = nvcc
SRC = backprop_cuda.cu backprop.c facetrain.c imagenet.c

all: backprop backprop_s

backprop: $(SRC) *.h
	$(CC) $(SRC) -o $@

backprop_s: $(SRC) *.h
	$(CC) $(SRC) -o $@ -DSEPERATE

.PHONY: run
run: backprop backprop_s
	-./backprop 480000
	-./backprop_s 480000