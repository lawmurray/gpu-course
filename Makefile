CFLAGS=-Isrc -O3

main: src/main.o src/model.o src/data.o src/init.o
	nvcc $(CFLAGS) -o $@ $^ -lcublas -lcurand

%.o : %.cu
	nvcc $(CFLAGS) -c -o $@ $^

clean:
	rm -f src/*.o main
