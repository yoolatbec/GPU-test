CC = nvcc
SRC = main.cu parboil_cuda.c lbm.cu

all: lbm lbm_s

lbm: *.cu *.h *.c
	$(CC) $(SRC) -o $@

lbm_s: *.cu *.h *.c
	$(CC) $(SRC) -o $@ -DSEPERATE

run: lbm lbm_s
	-./lbm 200
	-./lbm_s 200