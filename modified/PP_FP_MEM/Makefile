CC = nvcc
SRC = PP_FP_MEM.cu

all: mem mem_s

mem: $(SRC)
	$(CC) $(SRC) -o $@

mem_s: $(SRC)
	$(CC) $(SRC) -o $@ -DSEPERATE

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