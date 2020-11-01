#### BUILD PARAMETERS ####

BU=build
OB=$(BU)/objects

#### COMPILATION PARAMETERS ####

CC=g++
CFLAGS=-std=gnu++11 -O3 -Wall
LDFLAGS=
MPIFLAGS=

# CUSTOM DEFINITIONS
ifneq ($(DEFINITIONS),)
	CFLAGS+=$(foreach definition, $(DEFINITIONS),-D$(definition))
endif

# TEST
ifeq ($(TEST),yes)
	EXEC=$(BU)/test
	CPP=test.cpp
	# compile with openMP
	LDFLAGS+=-fopenmp
	MPIFLAGS+=-fopenmp
else

# ABP CLONING ALGORITHM
ifeq ($(CLONING),yes)
	CPP=cloning.cpp
	# compile with openMP and libstdc++fs library
	CFLAGS+=-lstdc++fs
	LDFLAGS+=-fopenmp -lstdc++fs
	MPIFLAGS+=-fopenmp
ifeq ($(BIAS_POLARISATION),yes)
	EXEC=$(BU)/cloningP
	CFLAGS+=-DBIAS_POLARISATION
ifeq ($(CONTROLLED_DYNAMICS),yes)
	EXEC:=$(EXEC)_C
	CFLAGS+=-DCONTROLLED_DYNAMICS
endif
else
	EXEC=$(BU)/cloning
ifeq ($(CONTROLLED_DYNAMICS),1)
	EXEC:=$(EXEC)_C1
	CFLAGS+=-DCONTROLLED_DYNAMICS=1
endif
ifeq ($(CONTROLLED_DYNAMICS),2)
	EXEC:=$(EXEC)_C2
	CFLAGS+=-DCONTROLLED_DYNAMICS=2
endif
ifeq ($(CONTROLLED_DYNAMICS),3)
	EXEC:=$(EXEC)_C3
	CFLAGS+=-DCONTROLLED_DYNAMICS=3
endif
endif
ifeq ($(TORQUE_DUMP),yes)
	CFLAGS+=-DTORQUE_DUMP
endif
else

# ROTORS CLONING ALGORITHM
ifeq ($(CLONINGR),yes)
	EXEC=$(BU)/cloningR
	CPP=cloningR.cpp
	# compile with openMP and libstdc++fs library
	CFLAGS+=-lstdc++fs
	LDFLAGS+=-fopenmp -lstdc++fs
	MPIFLAGS+=-fopenmp
ifeq ($(BIAS),1)
	CFLAGS+=-DBIAS=1
	EXEC:=$(EXEC)_B1
ifeq ($(CONTROLLED_DYNAMICS),yes)
	CFLAGS+=-DCONTROLLED_DYNAMICS
	EXEC:=$(EXEC)_C
endif
else
	CFLAGS+=-DBIAS=0
	EXEC:=$(EXEC)_B0
endif
else

# ROTORS
ifeq ($(ROTORS),yes)
	EXEC=$(BU)/rotors
	CPP=mainR.cpp
else

# SIMULATIONS
ifeq ($(SIM),dat)
	CPP=main.cpp
	EXEC=$(BU)/simulation
	CFLAGS+=-DABP
else
ifeq ($(SIM),dat0)
	CPP=main0.cpp
	EXEC=$(BU)/simulation0
endif
ifeq ($(SIM),datN)
	CPP=mainN.cpp
	EXEC=$(BU)/simulationN
endif
ifeq ($(TYPE),AOUP)
# AOUPs
	CFLAGS+=-DAOUP
	EXEC:=$(EXEC)OU
endif
ifeq ($(TYPE),ABP)
# ABPs
	CFLAGS+=-DABP
endif
endif

endif
endif
endif
endif

# DEBUGGING OUTPUT
ifeq ($(DEBUG),yes)
	CFLAGS+=-DDEBUG
endif

# CELL LIST
ifeq ($(CELLLIST),yes)
	EXEC:=$(EXEC)_cell_list
	CFLAGS+=-DUSE_CELL_LIST
endif

# HEUN'S SCHEME
ifneq ($(HEUN),no)
	CFLAGS+=-DHEUN=true
endif

# NAME OF EXECTUABLE
ifneq ($(EXEC_NAME),)
	EXEC=$(BU)/$(EXEC_NAME)
endif

MAIN=main.cpp main0.cpp mainN.cpp mainR.cpp cloning.cpp cloningR.cpp test.cpp																	# files with main()
SRC=$(filter-out $(filter-out $(CPP), $(MAIN) pycpp.cpp), $(filter-out $(wildcard old*), $(wildcard *.cpp)))	# compile all files but the ones with wrong main()

OBJ=$(addprefix $(OB)/, $(SRC:.cpp=.o))

.PHONY: all memcheck massif clean mrproper

#### COMPILATION #####

all: dir $(EXEC)

dir:
	@mkdir -p $(BU)
	@mkdir -p $(OB)

$(EXEC): $(OBJ)
	$(CC) -o $@ $^ $(LDFLAGS)

#### DEPENDENCIES ####

$(OB)/dat.o: dat.cpp dat.hpp readwrite.hpp
	$(CC) -o $(OB)/dat.o -c dat.cpp $(CFLAGS)

$(OB)/env.o: env.cpp env.hpp
	$(CC) -o $(OB)/env.o -c env.cpp $(CFLAGS)

$(OB)/iteration.o: iteration.cpp iteration.hpp particle.hpp
	$(CC) -o $(OB)/iteration.o -c iteration.cpp $(CFLAGS)

$(OB)/maths.o: maths.cpp maths.hpp
	$(CC) -o $(OB)/maths.o -c maths.cpp $(CFLAGS)

$(OB)/particle.o: particle.cpp particle.hpp maths.hpp readwrite.hpp
	$(CC) -o $(OB)/particle.o -c particle.cpp $(CFLAGS)

##

$(OB)/cloning.o: cloning.cpp cloningserial.hpp env.hpp particle.hpp readwrite.hpp
	$(CC) -o $(OB)/cloning.o -c cloning.cpp $(CFLAGS) $(MPIFLAGS)

$(OB)/cloningR.o: cloningR.cpp cloningserial.hpp env.hpp particle.hpp readwrite.hpp
	$(CC) -o $(OB)/cloningR.o -c cloningR.cpp $(CFLAGS) $(MPIFLAGS)

$(OB)/main.o: main.cpp env.hpp iteration.hpp particle.hpp
	$(CC) -o $(OB)/main.o -c main.cpp $(CFLAGS)

$(OB)/main0.o: main0.cpp env.hpp fire.hpp iteration.hpp maths.hpp particle.hpp
	$(CC) -o $(OB)/main0.o -c main0.cpp $(CFLAGS)

$(OB)/mainN.o: mainN.cpp env.hpp fire.hpp iteration.hpp maths.hpp particle.hpp
	$(CC) -o $(OB)/mainN.o -c mainN.cpp $(CFLAGS)

$(OB)/mainR.o: mainR.cpp env.hpp iteration.hpp particle.hpp
	$(CC) -o $(OB)/mainR.o -c mainR.cpp $(CFLAGS)

$(OB)/test.o: test.cpp
	$(CC) -o $(OB)/test.o -c test.cpp $(CFLAGS)

##

$(OB)/pycpp.o: dir pycpp.cpp pycpp.hpp maths.hpp
	$(CC) -o $(OB)/pycpp.o -c pycpp.cpp $(CFLAGS)

_pycpp.so: CFLAGS+=-fPIC
_pycpp.so: $(OB)/pycpp.o $(OB)/maths.o
	$(CC) -o _pycpp.so -shared $(OB)/pycpp.o $(OB)/maths.o $(LDFLAGS)

#### GPROF ####

gprof: dir $(OBJ)
	$(CC) -pg -o $(EXEC) $(OBJ) $(LDFLAGS)
	GMON_OUT_PREFIX=$(BU)/gmon.out $(EXEC)
	gprof $(EXEC) `ls -t $(BU)/gmon.out.* | head -n1` > $(BU)/gprof_analysis.txt

#### VALGRIND ####

callgrind: dir $(OBJ)
	$(CC) -o $(EXEC) $(OBJ) $(LDFLAGS)
	valgrind --tool=callgrind --callgrind-out-file=$(BU)/callgrind.%p.output $(EXEC)

memcheck: dir $(OBJ)
	$(CC) -g -o $(EXEC) $(OBJ) $(LDFLAGS)
	valgrind -s --leak-check=full --show-leak-kinds=all --track-origins=yes --log-file=$(BU)/memcheck.%p.output $(EXEC)

massif: dir $(OBJ)
	$(CC) -g -o $(EXEC) $(OBJ) $(LDFLAGS)
	valgrind --tool=massif --massif-out-file=$(BU)/massif.out $(EXEC)
	ms_print $(BU)/massif.out > $(BU)/massif.output

#### CLEAN ####

clean:
	rm -rf $(OB)

mrproper: clean
	rm -rf $(BU)
