#DEBUG = -DDEBUG -ggdb

# Linux
#CC = g++ -O3 -I../ ${DEBUG} -lOpenCL
# OSX 10.6
CC = g++ -O3 -framework opencl -I../ ${DEBUG}

# uncomment next line if you installed NTL with gf2x
LIBGF2X = -lgf2x
# uncomment next line if you installed NTL with gmp
#LIBGMP = -lgmp
LINKOPT = -lntl ${LIBGF2X} ${LIBGMP}

.SUFFIXES:      .cl .clh .cli

all:
	${MAKE} -C ../ all
	${MAKE} mtgp32-sample mtgp32-sample-jump1 mtgp32-sample-jump2 \
	mtgp64-sample mtgp64-sample-jump1 mtgp64-sample-jump2

mtgp32-sample: mtgp32-sample.cpp \
		opencl_tools.hpp \
		mtgp32-sample-common.h \
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

mtgp32-sample-jump1: mtgp32-sample-jump1.cpp \
		opencl_tools.hpp \
		mtgp32-sample-common.h \
		mtgp32-jump.cl \
		mtgp32-jump.cli \
		../mtgp32-fast-jump.h \
		../mtgp32-fast.h \
		parse_opt.o \
		mtgp32-jump-table.o
	${CC} -o $@ mtgp32-sample-jump1.cpp \
		parse_opt.o \
		mtgp32-jump-table.o \
		../mtgp32-fast-jump.o \
		../mtgp32-fast.o \
		../mtgp32-param-fast.o

mtgp32-sample-jump2: mtgp32-sample-jump2.cpp \
		opencl_tools.hpp \
		mtgp32-jump.cl \
		mtgp32-jump.cli \
		../mtgp32-fast-jump.h \
		../mtgp32-fast.h \
		../mtgp32-calc-poly.hpp \
		../mtgp-calc-jump.hpp \
		parse_opt.o
	${CC} -o $@ mtgp32-sample-jump2.cpp \
		parse_opt.o \
		../mtgp32-fast-jump.o \
		../mtgp32-fast.o \
		../mtgp32-param-fast.o \
		../mtgp32-calc-poly.o \
		${LINKOPT}

mtgp64-sample: mtgp64-sample.cpp \
		opencl_tools.hpp \
		mtgp64-sample-common.h \
		mtgp64.cl \
		mtgp64.cli \
		../mtgp64-fast.h \
		parse_opt.o \
		mtgp64dc-param-11213.o \
		../mtgp64-fast.o
	${CC} -o $@ mtgp64-sample.cpp \
		parse_opt.o \
		mtgp64dc-param-11213.o \
		../mtgp64-fast.o

mtgp64-sample-jump1: mtgp64-sample-jump1.cpp \
		opencl_tools.hpp \
		mtgp64-sample-common.h \
		mtgp64-jump.cl \
		mtgp64-jump.cli \
		../mtgp64-fast-jump.h \
		../mtgp64-fast.h \
		parse_opt.o \
		mtgp64dc-param-11213.o \
		mtgp64-jump-table.o
	${CC} -o $@ mtgp64-sample-jump1.cpp \
		parse_opt.o \
		mtgp64-jump-table.o \
		mtgp64dc-param-11213.o \
		../mtgp64-fast-jump.o \
		../mtgp64-fast.o

mtgp64-sample-jump2: mtgp64-sample-jump2.cpp \
		opencl_tools.hpp \
		mtgp64-jump.cl \
		mtgp64-jump.cli \
		../mtgp64-fast-jump.h \
		../mtgp64-fast.h \
		../mtgp64-calc-poly.hpp \
		../mtgp-calc-jump.hpp \
		parse_opt.o
	${CC} -o $@ mtgp64-sample-jump2.cpp \
		parse_opt.o \
		mtgp64dc-param-11213.o \
		../mtgp64-fast-jump.o \
		../mtgp64-fast.o \
		../mtgp64-calc-poly.o \
		${LINKOPT}

.c.o:
	gcc -O3 -I../ -c -o $@ $<

.cl.cli:
	cpp $< $@

clean:
	rm -rf *.o *~ *.dSYM