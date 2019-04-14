#
# Copyright (c) 2019 IMPACT Research Group, University of Illinois.
# All rights reserved.
#
# This file is covered by the LICENSE.txt license file in the root directory.
#

CXX=g++
CXXFLAGS=-std=c++11 -O0 -g
LD_FLAGS=
INCLUDE=-I../include

DEP=$(wildcard *.h)
SRC=$(wildcard *.cpp)
OBJ=$(SRC:.cpp=.o)
BIN=libpuma.so

$(BIN): $(OBJ)
	g++ $(CXXFLAGS) $(LD_FLAGS) -shared -o $@ $(OBJ)

%.o: %.cpp $(DEP)
	$(CXX) $(CXXFLAGS) $(INCLUDE) -c -fpic -o $@ $<

clean:
	rm -f *.o *.so

