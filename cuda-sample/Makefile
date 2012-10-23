# Makefile

#DEBUG = -DDEBUG -g -O0

#ARCH = -arch=compute_11
#ARCH = -arch=compute_13
#VERB = -Xptxas=-v
#ABI = -m32
ABI = -m64

NVCC =nvcc ${ABI} ${DEBUG} ${ARCH} -D__STDC_FORMAT_MACROS -I../ \
	 -D__STDC_CONSTANT_MACROS ${VERB}
CC = g++ -Wall ${ABI} ${DEBUG}  -D__STDC_FORMAT_MACROS -I../ \
	 -D__STDC_CONSTANT_MACROS

CUDALINK = -lcuda -lcudart

#.SUFFIXES:	.cu .cubin .ptx .bin

samples:sample32-cuda \
	sample64-cuda

sample32-cuda: sample.cpp \
	mtgp32-cuda.o \
	mtgp32dc-param-11213.o \
	mtgp32-fast.o \
	mtgp-print.o
	${CC} -o $@ sample.cpp \
	mtgp32-cuda.o \
	mtgp32dc-param-11213.o \
	mtgp32-fast.o \
	mtgp-print.o \
	${CUDALINK}

mtgp32-cuda.o: mtgp32-cuda.cu mtgp-util.cuh
	${NVCC} -o $@ -c mtgp32-cuda.cu

sample64-cuda: sample.cpp \
	mtgp64-cuda.o \
	mtgp64dc-param-11213.o \
	mtgp64-fast.o \
	mtgp-print.o
	${CC} -o $@ sample.cpp \
	mtgp64-cuda.o \
	mtgp64dc-param-11213.o \
	mtgp64-fast.o \
	mtgp-print.o \
	${CUDALINK}

mtgp64-cuda.o: mtgp64-cuda.cu mtgp-util.cuh
	${NVCC} -o $@ -c mtgp64-cuda.cu

mtgp32-fast.o:
	${CC} -c -o mtgp32-fast.o ../mtgp32-fast.c

mtgp64-fast.o:
	${CC} -c -o mtgp64-fast.o ../mtgp64-fast.c

.c.o:
	${CC} -c -o $@ $<

.cpp.o:
	${CC} -c -o $@ $<

clean:
	rm -rf *.o *~ *.cubin *.ptx *.dSYM
