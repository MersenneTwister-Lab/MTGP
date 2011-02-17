# -*- Makefile -*-
# @file  Makefile
# @brief Makefile
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
CC = gcc -Wall -Wextra -O3 $(STD)
ARCH = -arch=compute_11
#ARCH = -arch=compute_13
NVCC = nvcc -DDEBUG $(ARCH) #--compiler-options '-Wall'

# for OS X
#CUTILINC = -I"/Developer/GPU Computing/C/common/inc/" -I../
#CUTILLIB = -L"/Developer/GPU Computing/C/lib/" -lcutil_i386
# for linux
CUTILINC = -I$(HOME)/NVIDIA_GPU_Computing_SDK/C/common/inc -I../
CUTILLIB = -L$(HOME)/NVIDIA_GPU_Computing_SDK/C/lib -lcutil

CUDALINK = -lcuda
CPP = g++
CPPFLAGS = -Wall -Wextra -O3
CCFLAGS = $(OPTI) $(WARN) $(STD) $(DEBUG)

.SUFFIXES:	.cu .cubin .ptx .bin

hybridtaus-cuda: hybridtaus-cuda.cu util.c
	${NVCC} -o $@ hybridtaus-cuda.cu util.c \
	 ${CUTILINC} ${CUTILLIB} ${CUDALINK}

warp-cuda: warp-cuda.cu util.c
	${NVCC} -o $@ warp-cuda.cu util.c \
	 ${CUTILINC} ${CUTILLIB} ${CUDALINK}

warp: warp.c
	${CC} -o $@ warp.c

.cu.ptx:
	${NVCC} ${CUTILINC} -ptx $<

.cu.cubin:
	${NVCC} ${CUTILINC} -cubin $<

.cu.bin:
	${NVCC} ${ARCH} ${CUTILINC} -o $@ $< ${CUTILLIB} ${CUDALINK}

.c.o:
	${CC} ${CCFLAGS} -c $<

.cpp.o:
	${CPP} ${CCFLAGS} -c $<

clean:
	rm -f *.o *~
