train: train.h train.c train.cu
	nvcc -O3 -lcublas -o $@ $@.c $@.cu
