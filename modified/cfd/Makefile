CC = nvcc

all: euler3d euler3d_s euler3d_double euler3d_double_s

euler3d: euler3d.cu
	$(CC) euler3d.cu -o $@

euler3d_s: euler3d.cu
	$(CC) euler3d.cu -o $@ -DSEPERATE

euler3d_double: euler3d_double.cu
	$(CC) euler3d_double.cu -o $@

euler3d_double_s: euler3d_double.cu
	$(CC) euler3d_double.cu -o $@ -DSEPERATE

.PHONY: run run_double
run: euler3d euler3d_s
	-./euler3d fvcorr.domn.097k
	-./euler3d_s fvcorr.domn.097k

run_double: euler3d_double euler3d_double_s
	-./euler3d_double fvcorr.domn.097k
	-./euler3d_double_s fvcorr.domn.097k
