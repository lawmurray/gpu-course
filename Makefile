NVCC=nvcc
CFLAGS=-Isrc -I/usr/local/cuda/include -O3 -g
LDFLAGS=-L/usr/local/cuda/lib64

main: src/main.o src/model.o src/data.o src/optimizer.o src/init.o src/function.o
	$(NVCC) $(CFLAGS) $(LDFLAGS) -o $@ $^ -lcublas

%.o : %.cu
	$(NVCC) $(CFLAGS) -Xcompiler=-Wall -c -o $@ $^

clean:
	rm -f src/*.o main
