DEBUG = -DDEBUG -ggdb

# Linux
#CC = g++ -I../ ${DEBUG} -lOpenCL
# OSX 10.6
CC = g++ -framework opencl -I../ ${DEBUG}

# uncomment next line if you installed NTL with gf2x
LIBGF2X = -lgf2x
# uncomment next line if you installed NTL with gmp
#LIBGMP = -lgmp
LINKOPT = -lntl ${LIBGF2X} ${LIBGMP}

.SUFFIXES:      .cl .clh .cli

mtgp32-sample: mtgp32-sample.cpp \
	mtgp32.cl \
	mtgp32.cli \
	../mtgp32-fast.h \
	parse_opt.o \
	../mtgp32-fast.o \
	../mtgp32-param-fast.o
	${CC} -o $@ mtgp32-sample.cpp \
	parse_opt.o \
	../mtgp32-fast.o \
	../mtgp32-param-fast.o

mtgp32-sample-jump1: mtgp32-sample-jump1.cpp ../mtgp32-fast-jump.h \
	mtgp32-jump-table.o ../mtgp32-fast.h mtgp32-jump.cl \
	mtgp32-jump.cli \
	parse_opt.o mtgp32-jump-table.o
	${CC} -o $@ mtgp32-sample-jump1.cpp parse_opt.o \
	mtgp32-jump-table.o ../mtgp32-fast-jump.o ../mtgp32-fast.o \
	../mtgp32-param-fast.o

mtgp32-sample-jump2: mtgp32-sample-jump2.cpp ../mtgp32-fast-jump.h \
	mtgp32-jump-table.o ../mtgp32-fast.h mtgp32-jump.cl \
	mtgp32-jump.cli ../mtgp32-calc-poly.hpp ../mtgp-calc-jump.hpp \
	parse_opt.o
	${CC} -o $@ mtgp32-sample-jump2.cpp parse_opt.o \
	../mtgp32-fast-jump.o ../mtgp32-fast.o \
	../mtgp32-param-fast.o ../mtgp32-calc-poly.o ${LINKOPT}

.cl.cli:
	cpp $< $@

clean:
	rm -rf *.o *~ *.dSYM