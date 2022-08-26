CC = nvcc
SRC = main.cu parboil_cuda.c lbm.cu

all: lbm lbm_s

lbm: *.cu *.h *.c
	$(CC) $(SRC) -o $@

lbm_s: *.cu *.h *.c
	$(CC) $(SRC) -o $@ -DSEPERATE

.PHONY: run run_inv clean
run: lbm lbm_s
	-./lbm 200
	-./lbm_s 200

run_inv: lbm lbm_s
	-./lbm_s 200
	-./lbm 200

clean:
	-rm lbm
	-rm lbm_s