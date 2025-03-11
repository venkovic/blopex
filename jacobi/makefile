# makefile using 64 bit long integer


# --------------------------------
# C sample drivers
# testing the CHOLMOD interface
MAIN=script


MKL=-D_USE_MKL_
# MKL=



ifeq ($(MKL),-D_USE_MKL_)
MKL_INCLUDE=-I$(MKLROOT)/include 
MKL_LIBPATH=-L$(MKLROOT)/lib 
MKL_LIBS=-lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core  -lpthread -liomp5
else
MKL_LIBPATH=-L .
MKL_LIBS= liblapack-32.a libblas-32.a 
endif


$(MAIN).out: $(MAIN).o jacobi.o qsort.o
	gfortran -O -fPIC -fopenmp -m64 -mcmodel=medium -o$(MAIN).out $(MAIN).o  jacobi.o qsort.o -L . $(MKL_LIBS)  -lstdc++
#	gfortran -g -fsanitize=leak,address,signed-integer-overflow -fPIC -fopenmp -m64 -mcmodel=medium -o$(MAIN).out $(MAIN).o jacobi.o qsort.o -L . $(MKL_LIBS)  -lstdc++ -lasan


.c.o:
	gcc -c -O -fPIC -fopenmp -m64 -DMKL_LP64 $(MKL) -mcmodel=medium $(MKL_INCLUDE) -I. $*.c
#	gcc -c -g -fsanitize=leak -fsanitize=address -fsanitize=signed-integer-overflow -fPIC -fopenmp -m64 -DMKL_LP64 $(MKL)  -mcmodel=medium $(MKL_INCLUDE) -I. $*.c

.f.o:
	gfortran -c -O -fPIC -fopenmp -m64 -mcmodel=medium $*.f
#	gfortran -c -g -fsanitize=leak -fsanitize=address -fsanitize=signed-integer-overflow -fPIC -fopenmp -m64 -mcmodel=medium $*.f

.cpp.o:
#	g++ -c -O -fPIC -fopenmp -m64 -DMKL_LP64  $(MKL) -mcmodel=medium $(MKL_INCLUDE) -I. $*.cpp
	g++ -c -g -fsanitize=leak -fsanitize=address -fsanitize=signed-integer-overflow -fPIC -fopenmp -m64 -DMKL_LP64  $(MKL) -mcmodel=medium  $(MKL_INCLUDE) -I. $*.cpp

clean:
	rm -f *.o *.out
