CC = g++
CFLAGS = -g -O2
TARGET = $(filter-out print.cpp matrix%,$(wildcard *.cpp))
OBJ = $(TARGET:.cpp=.o) matrix.o
TARGET_EXEC = $(OBJ:.o=)
MATRIX_CFLAGS = -I/usr/include/eigen3

.DELETE_ON_ERROR:
all: $(TARGET_EXEC)

matrix: matrix.cpp
	$(CC) $(CFLAGS) $(MATRIX_CFLAGS) $< -o $@

%: %.cpp
	$(CC) $(CFLAGS) $< -o $@ 

.PHONY: clean

clean:
	rm -f *.cpp.* *.o $(TARGET_EXEC)
