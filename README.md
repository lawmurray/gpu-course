# Foundations of GPU Computing: Example Code

This code accompanies the course [Foundations of GPU Computing](https://indii.org/gpu-course/) (coming soon). It implements a simple feed-forward neural network in C, using CUDA and cuBLAS to run on Nvidia GPUs. The code includes the forward and backward (gradient) passes, and an Adam optimizer for training.

The purpose of using C is to reinforce the foundational lessons of the course, forcing us to be explicit about each step: each memory allocation, each kernel call, each stream synchronization.

## License

This code is open source software. It is licensed under the Apache License,
Version 2.0 (the "License"); you may not use it except in compliance with the
License. You may obtain a copy of the License at
<http://www.apache.org/licenses/LICENSE-2.0>.

## Build and run

The code requires:
- Linux.
- [CUDA](https://developer.nvidia.com/cuda-downloads).
- An Nvidia GPU of the Pascal generation (circa 2016) or later. Specifically, the code makes use of unified virtual memory, and only from the Pascal generation forward can the hardware coherently handle managed memory and kernel execution concurrently.

To build, use:

    make

To run, use:

    ./main

If successful, the output will be one line per epoch, reporting test loss and elapsed time. If the build fails, it may be necessary to modify the `Makefile` for the system.

## Data

The file `bikeshare.csv` provides the data set. Each row corresponds to one hour of the year 2019, and each column one of 23 features (e.g. weather and holiday information, rescaled) and, for the last column, the label (number of trips in the hour, normalized).

The data has been prepared from the following sources:

  * [Capital Bikeshare trips](https://www.capitalbikeshare.com/system-data)
  * [Capital Bikeshare station locations](http://opendata.dc.gov/datasets/a1f7acf65795451d89f0a38565a975b3_5)
  * [U.S. Local Climatological Data (LCD)](https://www.ncei.noaa.gov/metadata/geoportal/rest/metadata/item/gov.noaa.ncdc:C00684/html)
LCD_documentation.pdf)
  * [NOAA Solar Geometry Calculator](https://www.esrl.noaa.gov/gmd/grad/antuv/SolarCalc.jsp)

The course is only concerned with GPU programming and not model performance, but this at least provides some realistic data to play with.
