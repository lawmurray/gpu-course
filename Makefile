CFLAGS=-Isrc

main: src/main.o src/model.o src/data.o src/aux.o
	nvcc -O3 -lcublas -lcurand -o $@ $^

%.o : %.cu
	nvcc -O3 -c -o $@ $^

clean:
	rm -f src/*.o main
