# Structure of the Project

```bash
.
├── video_maker.cpp  # main program
├── CMakeLists.txt   # CMake configuration
├── README.md        # this file
├── report.pdf       # report
├── build            # build directory
└── input            # example input images/video and output video
    ├── slide_1.png
    ├── slide_2.png
    ├── slide_3.png
    ├── slide_4.png
    ├── slide_5.png
    ├── input_video.avi
    └── output_video.avi
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

The executable file is `video_maker`, which is located in the `build` directory. In order to run the program, you need to provide the path to the input directory.

```bash
./video_maker <input_path>
```

For example, to run the program on the example images and video and save the result to the `./input/output_video.avi` file, you can use the following command:

```bash
./video_maker ../input
```

# Examples and Results

I have provided an example input directory in the `./input` directory. The results are saved in the same directory. You can switch to the `./input` directory and run the program to see the results.