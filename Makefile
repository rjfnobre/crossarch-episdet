CXX = dpcpp
CXXFLAGS = -I gsl/include -L gsl/lib -O3 -DNUM_CPU_THREADS=10 -o 
LDFLAGS = -lgsl -lgslcblas -fiopenmp 
EXE_NAME = episdet
SOURCES = src/epistasis.dp.cpp
BINDIR = bin

all: main

main:
	[ -d $(BINDIR) ] || mkdir $(BINDIR)
	$(CXX) $(CXXFLAGS) $(BINDIR)/$(EXE_NAME) $(SOURCES) $(LDFLAGS)



run:
	qsub -l nodes=1:ppn=2 -d . run.sh

run_cpu:
	qsub -l nodes=1:ppn=2:gold6128 -d . run.sh

run_gpu:
	qsub -l nodes=1:ppn=2:iris_xe_max -d . run.sh

clean: 
	rm -rf $(BINDIR)/$(EXE_NAME)

