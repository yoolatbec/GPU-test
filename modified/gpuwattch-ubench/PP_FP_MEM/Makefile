CC = nvcc
SRC = PP_FP_MEM.cu

all: mem mem_s
debug: mem_g mem_sg

mem: $(SRC)
	$(CC) $(SRC) -o $@

mem_s: $(SRC)
	$(CC) $(SRC) -o $@ -DSEPERATE

mem_g: $(SRC)
	$(CC) $(SRC) -G -o $@

mem_sg: $(SRC)
	$(CC) $(SRC) -G -o $@ -DSEPERATE

.PHONY: run run_inv clean
run: mem mem_s
	-./mem
	-./mem_s

run_inv: mem mem_s
	-./mem_s
	-./mem

clean:
	-rm mem
	-rm mem_s
	-rm mem_g
	-rm mem_sg