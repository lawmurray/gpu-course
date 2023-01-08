NVCC=nvcc
CFLAGS=-Isrc -I/usr/local/cuda/include -O3 -g
NVCCFLAGS=$(CFLAGS) -allow-unsupported-compiler
LDFLAGS=-L/usr/local/cuda/lib64

main: src/main.o src/model.o src/data.o src/optimizer.o src/init.o src/function.o
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o $@ $^ -lcublas

%.o : %.cu
	$(NVCC) $(NVCCFLAGS) -Xcompiler=-Wall -c -o $@ $<

src/main.o: src/model.c src/config.h src/init.h src/data.h src/model.h src/optimizer.h
src/model.o: src/model.c src/config.h src/model.h
src/data.o: src/data.c src/config.h src/data.h
src/optimizer.o: src/optimizer.c src/config.h src/optimizer.h
src/init.o: src/init.c src/config.h src/init.h
src/function.o: src/function.cu src/config.h src/function.h

clean:
	rm -f src/*.o main
