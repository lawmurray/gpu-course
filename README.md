# Foundations of GPU Computing: Example Code

This code accompanies the course [Foundations of GPU Computing](https://indii.org/gpu-course/) (coming soon). It implements a simple feed-forward neural network in C, using CUDA language extensions and the cuBLAS library to run on GPU. The purpose of using C is to reinforce the foundational lessons of the course, forcing us to be explicit about each step: each memory allocation, each kernel call, each stream synchronization.

## Build and run

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
