CC = nvcc

all: srad srad_s

srad: *.cu *.c *.h
	$(CC) main.cu -o $@

srad_s: *.cu *.c *.h
	$(CC) main.cu -o $@ -DSEPERATE

.PHONY: run
run: srad srad_s
	-./srad 100 0.5 502 458
	-./srad_s 100 0.5 502 458