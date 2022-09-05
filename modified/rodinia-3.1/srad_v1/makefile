CC = nvcc

all: srad srad_s
debug: srad_g srad_sg

srad: *.cu *.c *.h
	$(CC) main.cu -o $@

srad_s: *.cu *.c *.h
	$(CC) main.cu -o $@ -DSEPERATE

srad_g: *.cu *.c *.h
	$(CC) main.cu -G -o $@

srad_sg: *.cu *.c *.h
	$(CC) main.cu -G -o $@ DSEPERATE

.PHONY: run run_inv clean
run: srad srad_s
	-./srad 100 0.5 502 458
	-./srad_s 100 0.5 502 458

run_inv: srad srad_s
	-./srad_s 100 0.5 502 458
	-./srad 100 0.5 502 458

clean:
	-rm srad
	-rm srad_s
	-rm srad_g
	-rm srad_sg