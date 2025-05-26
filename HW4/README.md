# Structure of the Project

```bash
.
├── calibration_birdeye.cpp  # main program
├── CMakeLists.txt   # CMake configuration
├── README.md        # this file
├── report.pdf       # report
├── build            # build directory
└── examples         # example images and results
    ├── example1.jpg    # example image 1
    ├── example2.jpg    # example image 2
    └── example1_results
        ├── detected_corners.jpg    # detected corners
        ├── undistorted_image.jpg    # undistorted image
        └── birds_eye_view.jpg    # birds-eye view
        └── camera_params.txt    # camera parameters
    └── example2_results
        ├── detected_corners.jpg    # detected corners
        ├── undistorted_image.jpg    # undistorted image
        └── birds_eye_view.jpg    # birds-eye view
        └── camera_params.txt    # camera parameters
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

**First make sure you are in the `build` directory.**

The executable file is `calibration_birdeye`, which is located in the `build` directory. In order to run the program, you need to provide the path to the image file and the path to the output directory. **Note that you need to manually set the chessboard width and height points. Otherwise it will not work.**

```bash
./calibration_birdeye <image_path> <output_path> <chessboard_width_points> <chessboard_height_points>
```

For example, to run the program on the first example image and save the result to the `./examples/example1_results` directory, you can use the following command:

```bash
./calibration_birdeye ../examples/example1.jpg 6 9
```

For the second example image, you can use the following command:

```bash
./calibration_birdeye ../examples/example2.jpg 9 6
```

# Examples and Results

I have provided 2 example images in the `./examples` directory. The results are saved in the same directory. You can switch to the `./examples` directory and run the program to see the results.