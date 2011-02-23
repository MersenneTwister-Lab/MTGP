# -*- Makefile -*-
# @file  Makefile.osx
# @brief Makefile.osx
#
# @author Mutsuo Saito (Hiroshima University)
# @author Makoto Matsumoto (University of Tokyo)
#
# Copyright (C) 2009, 2010 Mutsuo Saito, Makoto Matsumoto and
# Hiroshima University. All rights reserved.
# Copyright (C) 2011 Mutsuo Saito, Makoto Matsumoto, Hiroshima
# University and University of Tokyo. All rights reserved.
#
# The new BSD License (3-clause license) is applied to this software.
# See LICENSE.txt

#DEBUG = -DDEBUG -DDEVICE_EMU -g -O0 -deviceemu
WARN = -Wmissing-prototypes -Wall #-Winline
OPTI = -O3 -finline-functions -fomit-frame-pointer -DNDEBUG \
-fno-strict-aliasing

STD = -std=c99
CC = gcc
ARCH = -arch=compute_11
#ARCH = -arch=compute_13
NVCC = nvcc -DDEBUG -D__STDC_CONSTANT_MACROS=1 -D__STDC_FORMAT_MACROS=1 \
	-I../ $(ARCH) #--compiler-options '-Wall'

CUDALINK = -lcuda
CPP = g++
CPPFLAGS = -Wall -Wextra -O3
CCFLAGS = $(OPTI) $(WARN) $(STD) $(DEBUG)

.SUFFIXES:	.cu .cubin .ptx .bin

hybridtaus-cuda: hybridtaus-cuda.cu util.cu ../mtgp32-fast.c
	${NVCC} -o $@ hybridtaus-cuda.cu util.cu ../mtgp32-fast.c \
	 ${CUTILINC} ${CUTILLIB} ${CUDALINK}

warp-cuda: warp-cuda.cu util.cu ../mtgp32-fast.c
	${NVCC} -o $@ warp-cuda.cu util.cu ../mtgp32-fast.c \
	 ${CUTILINC} ${CUTILLIB} ${CUDALINK}

mtgp32-cuda-tex: mtgp32-cuda-tex.cu util.cu \
	mtgp32dc-param-11213.c ../mtgp32-fast.c
	${NVCC} -o $@ mtgp32-cuda-tex.cu util.cu \
	mtgp32dc-param-11213.c ../mtgp32-fast.c ${CUDALINK}

warp: warp.cpp
	${CPP} -o $@ warp.cpp -lntl -lgf2x

.cu.ptx:
	${NVCC} ${CUTILINC} -ptx $<

.cu.cubin:
	${NVCC} ${CUTILINC} -cubin $<

.cu.bin:
	${NVCC} ${ARCH} ${CUTILINC} -o $@ $< ${CUTILLIB} ${CUDALINK}

clean:
	rm -f *.o *~
