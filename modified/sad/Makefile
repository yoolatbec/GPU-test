CC = nvcc

SRC = file.c image.c largerBlocks.cu main.cu parboil_cuda.c sad4.cu

all: sad sad_s
debug: sad_g sad_sg

sad: *.cu *.h *.c
	$(CC) $(SRC) -o $@

sad_s: *.cu *.h *.c
	$(CC) $(SRC) -o $@ -DSEPERATE

sad_g: *.cu *.h *.c
	$(CC) $(SRC) -G -o $@

sad_sg: *.cu *.h *.c
	$(CC) $(SRC) -G -o $@ -DSEPERATE

.PHONY: run run_inv clean
run: sad sad_s
	-./sad -i reference.bin,frame.bin
	-./sad_s -i reference.bin,frame.bin

run_inv: sad sad_s
	-./sad_s -i reference.bin,frame.bin
	-./sad -i reference.bin,frame.bin

clean:
	-rm sad
	-rm sad_s
	-rm sad_g
	-rm sad_sg