# @file  Makefile
# @brief Makefile
#
# @author Mutsuo Saito (Hiroshima University)
# @author Makoto Matsumoto (Hiroshima University -> The University of Tokyo)
#
# Copyright (C) 2009, 2010 Mutsuo Saito, Makoto Matsumoto and
# Hiroshima University.
# Copyright (C) 2012 Mutsuo Saito, Makoto Matsumoto,
# Hiroshima University and The University of Tokyo.
# All rights reserved.
#
# The new BSD License is applied to this software.
# see LICENSE.txt
#
# jump programs need NTL(@see http://shoup.net/ntl/)

#DEBUG = -DDEBUG -ggdb -O0
# uncomment next line if you installed NTL with gf2x
LIBGF2X = -lgf2x
# uncomment next line if you installed NTL with gmp
#LIBGMP = -lgmp

LINKOPT = -lntl $(LIBGF2X) $(LIBGMP)

WARN = -Wmissing-prototypes -Wall
OPTI = -O3
STD = -std=c99
CC = gcc
CPP = g++
CPPFLAGS = -Wall -Wextra -O3 -msse3 $(DEBUG)
CCFLAGS = $(OPTI) $(WARN) $(STD) $(DEBUG)
OBJS = mtgp32-fast.o mtgp64-fast.o mtgp32-param-fast.o mtgp64-param-fast.o \
	mtgp-calc-jump.o mtgp64-fast-jump.o mtgp32-calc-poly.o \
	mtgp32-fast-jump.o mtgp64-calc-poly.o

mtgp:test64-ref test64-fast test32-ref test32-fast

# need NTL
jump:mtgp mtgp-calc-jump mtgp32-calc-poly mtgp64-calc-poly \
	test-jump32 test-jump64 ${OBJS}

all:mtgp jump ${OBJS}

test64-ref: mtgp64-ref.h mtgp64-ref.c mtgp64-param-ref.o
	${CC} ${CCFLAGS} -DMAIN=1 -o $@ mtgp64-ref.c mtgp64-param-ref.o

test64-fast: mtgp64-fast.h mtgp64-fast.c mtgp64-param-fast.o
	${CC} ${CCFLAGS} -DMAIN=1 -o $@ mtgp64-fast.c mtgp64-param-fast.o

test32-ref: mtgp32-ref.h mtgp32-ref.c mtgp32-param-ref.o
	${CC} ${CCFLAGS} -DMAIN=1 -o $@ mtgp32-ref.c mtgp32-param-ref.o

test32-fast: mtgp32-fast.h mtgp32-fast.c mtgp32-param-fast.o
	${CC} ${CCFLAGS} -DMAIN=1 -o $@ mtgp32-fast.c mtgp32-param-fast.o

mtgp-calc-jump: mtgp-calc-jump.cpp mtgp-calc-jump.hpp
	${CPP} ${CPPFLAGS} -o $@ mtgp-calc-jump.cpp ${LINKOPT}

mtgp32-calc-poly: mtgp32-calc-poly.cpp mtgp32-fast.h mtgp32-fast.o \
	mtgp32-param-fast.o
	${CPP} ${CPPFLAGS} -DMAIN=1 -o $@ mtgp32-calc-poly.cpp mtgp32-fast.o \
	mtgp32-param-fast.o ${LINKOPT}

test-jump32: test-jump32.cpp mtgp32-calc-poly.cpp mtgp-calc-jump.hpp \
	mtgp32-fast.h mtgp32-fast.o mtgp32-fast-jump.o mtgp32-param-fast.o
	${CPP} ${CPPFLAGS}  -o $@ test-jump32.cpp \
	mtgp32-calc-poly.cpp mtgp32-fast.o mtgp32-fast-jump.o \
	mtgp32-param-fast.o ${LINKOPT}

mtgp64-calc-poly: mtgp64-calc-poly.cpp mtgp64-fast.h mtgp64-fast.o \
	mtgp64-param-fast.o
	${CPP} ${CPPFLAGS} -DMAIN=1 -o $@ mtgp64-calc-poly.cpp mtgp64-fast.o \
	mtgp64-param-fast.o ${LINKOPT}

test-jump64: test-jump64.cpp mtgp64-calc-poly.cpp mtgp-calc-jump.hpp \
	mtgp64-fast.h mtgp64-fast.o mtgp64-fast-jump.o mtgp64-param-fast.o
	${CPP} ${CPPFLAGS}  -o $@ test-jump64.cpp \
	mtgp64-calc-poly.cpp mtgp64-fast.o mtgp64-fast-jump.o \
	mtgp64-param-fast.o ${LINKOPT}

.c.o:
	${CC} ${CCFLAGS} -c $<

clean:
	rm -rf *.o *~ *.dSYM
