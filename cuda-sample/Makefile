# @file  Makefile
# @brief Makefile
#
# @author Mutsuo Saito (Hiroshima University)
# @author Makoto Matsumoto (Hiroshima University)
#
# Copyright (C) 2009, 2010 Mutsuo Saito, Makoto Matsumoto and
# Hiroshima University. All rights reserved.
#
# The new BSD License is applied to this software.
# see LICENSE.txt
#

#DEBUG = -DDEBUG -g -O0
WARN = -Wmissing-prototypes -Wall #-Winline
OPTI = -O3 -finline-functions -fomit-frame-pointer -DNDEBUG \
-fno-strict-aliasing

STD = -std=c99
CC = gcc
NVCC = nvcc -I../
CUTILINC = -I$(HOME)/NVIDIA_GPU_Computing_SDK/C/common/inc
CUTILLIB = -L$(HOME)/NVIDIA_GPU_Computing_SDK/C/lib -lcutil
CUDALINK = -lcuda
CPP = g++
CPPFLAGS = -Wall -Wextra -O3
CCFLAGS = $(OPTI) $(WARN) $(STD) $(DEBUG)

sample32-cuda: mtgp32-cuda.cu ../mtgp32-fast.h mtgp32dc-param-11213.c \
	../mtgp32-fast.o mtgp32-cuda-common.c mtgp-cuda-common.c
	${NVCC} -o $@ mtgp32-cuda.cu ../mtgp32-fast.o \
	 ${CUTILINC} ${CUTILLIB} ${CUDALINK}

sample32-cuda-tex: mtgp32-cuda-tex.cu ../mtgp32-fast.h mtgp32dc-param-11213.c \
	../mtgp32-fast.o mtgp32-cuda-common.c mtgp-cuda-common.c
	${NVCC} -o $@ mtgp32-cuda-tex.cu ../mtgp32-fast.o \
	 ${CUTILINC} ${CUTILLIB} ${CUDALINK}

sample64-cuda: mtgp64-cuda.cu ../mtgp64-fast.h mtgp64dc-param-11213.c \
	../mtgp64-fast.o mtgp64-cuda-common.c mtgp-cuda-common.c
	${NVCC} -o $@ mtgp64-cuda.cu ../mtgp64-fast.o \
	 ${CUTILINC} ${CUTILLIB} ${CUDALINK}

sample64-cuda-tex: mtgp64-cuda-tex.cu ../mtgp64-fast.h mtgp64dc-param-11213.c \
	../mtgp64-fast.o mtgp64-cuda-common.c mtgp-cuda-common.c
	${NVCC} -o $@ mtgp64-cuda-tex.cu ../mtgp64-fast.o \
	 ${CUTILINC} ${CUTILLIB} ${CUDALINK}

.c.o:
	${CC} ${CCFLAGS} -c $<

clean:
	rm -f *.o *~
