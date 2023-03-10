# compile flags
CPPFLAGS=-Wno-macro-redefined
CFLAGS=-Isrc -I/usr/local/cuda/include -O3 -g
NVCCFLAGS=-Isrc -I/usr/local/cuda/include -O3 -g -allow-unsupported-compiler -Xcompiler=-Wall -Xcompiler=-Wno-macro-redefined

# link flags
LDFLAGS=-L/usr/local/cuda/lib64

# default build
all: main discrete

# build the main program
main: src/main.o src/model.o src/data.o src/optimizer.o src/init.o src/function.o
	nvcc $(NVCCFLAGS) $(LDFLAGS) -o $@ $^ -lcublas

# build the discrete program
discrete: src/discrete.o src/convolve.o src/init.o src/function.o
	nvcc $(NVCCFLAGS) $(LDFLAGS) -o $@ $^ -lcublas

# remove all build artifacts
clean:
	rm -f src/*.o main discrete

# compile each source file, with dependencies
src/main.o: src/main.c src/config.h src/init.h src/data.h src/model.h src/optimizer.h
src/model.o: src/model.c src/config.h src/model.h
src/data.o: src/data.c src/config.h src/data.h
src/optimizer.o: src/optimizer.c src/config.h src/optimizer.h
src/init.o: src/init.c src/config.h src/init.h
src/function.o: src/function.cu src/config.h src/function.h
src/discrete.o: src/discrete.c src/config.h src/init.h
src/convolve.o: src/convolve.cu src/config.h src/init.h

# pattern rule to compile an object file from a CUDA source file
%.o : %.cu
	nvcc $(NVCCFLAGS) -c -o $@ $<
