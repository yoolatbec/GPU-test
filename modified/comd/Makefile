CC = nvcc
INC = -I. -I./prototypes -I./utils
SRC = ./utils/*.cpp *.cu *.cpp

all: comd comd_s

comd: $(SRC) *.h ./prototypes/*.h ./utils/*.h
	$(CC) $(SRC) $(INC) -o $@

comd_s: $(SRC) *.h ./prototypes/*.h ./utils/*.h
	$(CC) $(SRC) $(INC) -o $@ -DSEPERATE

.PHONY: run clean
run:
	-./comd -f data/8k.inp -x 1024 -y 1024 -z 1024 -i 1000
	-./comd_s -f data/8k.inp -x 1024 -y 1024 -z 1024 -i 1000