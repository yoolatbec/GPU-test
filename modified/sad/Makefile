CC = nvcc

SRC = file.c image.c largerBlocks.cu main.cu parboil_cuda.c sad4.cu

all: sad sad_s

sad: *.cu *.h *.c
	$(CC) $(SRC) -o $@

sad_s: *.cu *.h *.c
	$(CC) $(SRC) -o $@ -DSEPERATE

.PHONY: run
run: sad sad_s
	-./sad -i reference.bin,frame.bin
	-./sad_s -i reference.bin,frame.bin