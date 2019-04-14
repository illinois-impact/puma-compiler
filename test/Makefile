# 
#  Copyright (c) 2019 IMPACT Research Group, University of Illinois.
#  All rights reserved.
# 
#  This file is covered by the LICENSE.txt license file in the root directory.
# 

CXX=g++
CXXFLAGS=-std=c++11 -O0 -g
LD_FLAGS=
INCLUDE=-I../include
LIB=-L../src -lpuma

DEP=$(wildcard *.h)
SRC=$(wildcard *.cpp)
TEST=$(SRC:.cpp=.test)

default: $(TEST)

%.test: %.o
	g++ $(CXXFLAGS) $(LD_FLAGS) -o $@ $< $(LIB)

%.o: %.cpp $(DEP)
	g++ $(CXXFLAGS) $(INCLUDE) -c -o $@ $<

clean:
	rm -f *.o *.test *.dot *.pdf *.puma *.puma.py *.graph *.out *.weights

