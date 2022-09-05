CC = nvcc
SRC = lulesh.cu

all: lulesh lulesh_s
debug: lulesh_g lulesh_sg

lulesh: $(SRC)
	$(CC) $(SRC) -o $@

lulesh_s: $(SRC)
	$(CC) $(SRC) -o $@ -DSEPERATE

lulesh_g: $(SRC)
	$(CC) $(SRC) -G -o $@

lulesh_sg: $(SRC)
	$(CC) $(SRC) -G -o $@ -DSEPERATE

.PHONY: run run_inv clean
run: lulesh lulesh_s
	-./lulesh
	-./lulesh_s

run_inv: lulesh lulesh_s
	-./lulesh_s
	-./lulesh


clean:
	-rm lulesh
	-rm lulesh_s
	-rm lulesh_g
	-rm lulesh_sg