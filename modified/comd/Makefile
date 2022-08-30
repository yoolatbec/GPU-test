CC = nvcc
INC = -I. -I./prototypes -I./utils
SRC = ./utils/*.cpp *.cu *.cpp

all: comd comd_s
debug: comd_g comd_sg

comd: $(SRC) *.h ./prototypes/*.h ./utils/*.h
	$(CC) $(SRC) $(INC) -o $@

comd_s: $(SRC) *.h ./prototypes/*.h ./utils/*.h
	$(CC) $(SRC) $(INC) -o $@ -DSEPERATE

comd_g: $(SRC) *.h ./prototypes/*.h ./utils/*.h
	$(CC) $(SRC) $(INC) -G -o $@

comd_sg: $(SRC) *.h ./prototypes/*.h ./utils/*.h
	$(CC) $(SRC) $(INC) -G -o $@ -DSEPERATE

.PHONY: run run_inv clean
run: comd comd_s
	-./comd -f data/8k.inp -x 1024 -y 1024 -z 1024 -i 1000
	-./comd_s -f data/8k.inp -x 1024 -y 1024 -z 1024 -i 1000

run_inv: comd comd_s
	-./comd_s -f data/8k.inp -x 1024 -y 1024 -z 1024 -i 1000
	-./comd -f data/8k.inp -x 1024 -y 1024 -z 1024 -i 1000

clean:
	-rm comd
	-rm comd_s
	-rm comd_g
	-rm comd_sg