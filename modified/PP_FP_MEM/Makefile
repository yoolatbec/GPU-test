CC = nvcc
SRC = PP_FP_MEM.cu

all: mem mem_s

mem: $(SRC)
	$(CC) $(SRC) -o $@

mem_s: $(SRC)
	$(CC) $(SRC) -o $@ -DSEPERATE

.PHONY: run
run: mem mem_s
	-./mem
	-./mem_s