CC = nvcc

all: euler3d euler3d_s euler3d_double euler3d_double_s
debug: euler3d_g euler3d_sg euler3d_double_g euler3d_double_sg

euler3d: euler3d.cu
	$(CC) euler3d.cu -o $@

euler3d_s: euler3d.cu
	$(CC) euler3d.cu -o $@ -DSEPERATE

euler3d_double: euler3d_double.cu
	$(CC) euler3d_double.cu -o $@

euler3d_double_s: euler3d_double.cu
	$(CC) euler3d_double.cu -o $@ -DSEPERATE

euler3d_g: euler3d.cu
	$(CC) $< -G -o $@

euler3d_sg: euler3d.cu
	$(CC) $< -G -o $@ -DSEPERATE

euler3d_double_g: euler3d_double.cu
	$(CC) $< -G -o $@

euler3d_double_sg: euler3d_double.cu
	$(CC) $< -G -o $@ -DSEPERATE

.PHONY: run run_double run_inv run_double_inv clean
run: euler3d euler3d_s
	-./euler3d fvcorr.domn.097k
	-./euler3d_s fvcorr.domn.097k

run_double: euler3d_double euler3d_double_s
	-./euler3d_double fvcorr.domn.097k
	-./euler3d_double_s fvcorr.domn.097k

run_inv: euler3d euler3d_s
	-./euler3d_s fvcorr.domn.097k
	-./euler3d fvcorr.domn.097k

run_double_inv: euler3d_double_s euler3d_double
	-./euler3d_double_s fvcorr.domn.097k
	-./euler3d_double fvcorr.domn.097k

clean:
	-rm euler3d euler3d_s
	-rm euler3d_double euler3d_double_s
	-rm euler3d_g euler3d_sg
	-rm euler3d_double_g euler3d_double_sg