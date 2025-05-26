# Structure of the Project

```bash
.
├── ellipse_fit.cpp  # main program
├── CMakeLists.txt   # CMake configuration
├── README.md        # this file
├── report.pdf       # report
├── build            # build directory
└── examples         # example images and results
    ├── image_1.png
    ├── image_2.png
    ├── image_3.png
    └── result_1.png
    └── result_2.png
    └── result_3.png
```

# How to Compile and Run

Operating System: MacOS M4 Chip

## Install Dependencies

```bash
brew install cmake
brew install opencv
```

## Compile Using Clangd and CMake

**Clangd** is recommended to compile the program.

First switch to the root directory of the project.

```bash
mkdir build
cd build
cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=1 ..
make -j
```

## Run the Program

The executable file is `ellipse_fit`, which is located in the `build` directory. In order to run the program, you need to provide the path to the image file and the path to the output directory.

```bash
./ellipse_fit <image_path> <output_path>
```

For example, to run the program on the first example image and save the result to the `./examples/result_1.png` file, you can use the following command:

```bash
./ellipse_fit ../examples/image_1.png ../examples/result_1.png
```

# Examples and Results

I have provided three example images in the `./examples` directory. The results are saved in the same directory. You can switch to the `./examples` directory and run the program to see the results.