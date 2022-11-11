CFLAGS=-Isrc -O3

main: src/main.o src/model.o src/data.o src/optimizer.o src/init.o src/function.o
	nvcc $(CFLAGS) -g -o $@ $^ -lcublas -lcurand

%.o : %.cu
	nvcc $(CFLAGS) -g -c -o $@ $^

clean:
	rm -f src/*.o main
