CC = nvcc
SRC = main.cu parboil_cuda.c lbm.cu

all: lbm lbm_s
debug: lbm_g lbm_sg

lbm: *.cu *.h *.c
	$(CC) $(SRC) -o $@

lbm_s: *.cu *.h *.c
	$(CC) $(SRC) -o $@ -DSEPERATE

lbm_g: *.cu *.h *.c
	$(CC) $(SRC) -G -o $@

lbm_sg: *.cu *.h *.c
	$(CC) $(SRC) -G -o $@ -DSEPERATE

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
	-rm lbm_g
	-rm lbm_sg