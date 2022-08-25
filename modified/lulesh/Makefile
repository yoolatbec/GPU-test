CC = nvcc
SRC = lulesh.cu

all: lulesh lulesh_s

lulesh: $(SRC)
	$(CC) $(SRC) -o $@

lulesh_s: $(SRC)
	$(CC) $(SRC) -o $@ -DSEPERATE

.PHONY: run
run: lulesh lulesh_s
	-./lulesh
	-./lulesh_s

