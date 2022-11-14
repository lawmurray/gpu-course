CFLAGS=-Isrc -O3 -g

main: src/main.o src/model.o src/data.o src/optimizer.o src/init.o src/function.o
	nvcc $(CFLAGS) -o $@ $^ -lcublas

%.o : %.cu
	nvcc $(CFLAGS) -Xcompiler=-Wall -c -o $@ $^

clean:
	rm -f src/*.o main
