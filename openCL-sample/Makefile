DEBUG = -DDEBUG -ggdb

# Linux
#CC = g++ -I../ ${DEBUG} ${openclinc} ${opencllib}  -lOpenCL
# OSX 10.6
CC = g++ -framework opencl -I../ ${DEBUG}

.SUFFIXES:      .cl .clh .cli

mtgp32-sample-jump1: mtgp32-sample-jump1.cpp ../mtgp32-fast-jump.h \
	mtgp32-jump-table.o ../mtgp32-fast.h mtgp32-jump.cl \
	mtgp32-jump.cli \
	parse_opt.o mtgp32-jump-table.o
	${CC} -o $@ mtgp32-sample-jump1.cpp parse_opt.o \
	mtgp32-jump-table.o ../mtgp32-fast-jump.o ../mtgp32-fast.o \
	../mtgp32-param-fast.o

.cl.cli:
	cpp $< $@

clean:
	rm -rf *.o *~ *.dSYM