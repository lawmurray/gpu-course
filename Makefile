train: train.h train.c train.cu
	nvcc -O3 -lcublas -lcurand -o $@ $@.c $@.cu
