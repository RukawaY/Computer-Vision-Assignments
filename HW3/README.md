# Structure of the Project

```bash
.
├── train.cpp        # eigenface training program
├── test.cpp         # eigenface testing program
├── CMakeLists.txt   # CMake configuration for train and test
├── README.md        # this file
├── report.pdf       # report
├── build            # build directory for train and test
├── prepare_image    # prepare image for training and testing
│   ├── prepare_dataset.sh       # bash script to prepare dataset
│   ├── haarcascade_eye_tree_eyeglasses.xml # haar cascade xml file for face detection
│   ├── CMakeLists.txt           # CMake configuration for prepare_data
│   ├── build                    # build directory for prepare_data
│   └── prepare_data.cpp         # data preparation program
├── train_dataset    # training dataset
│   ├── S001
│   ├── S002
│   └── ...
├── test_dataset     # test dataset
│   ├── S001
│   ├── S002
│   └── ...
├── train_results
│   ├── checkpoint.txt
│   ├── mean_face.png
│   ├── eigenfaces_grid.png
│   ├── eigenface_0.png
│   ├── eigenface_1.png
│   └── ...
└── test_results
    ├── annotated_image.jpg
    └── closest_training_image.jpg
```

# How to Compile and Run

Operating System: MacOS M4 Chip

## Install Dependencies

```bash
brew install cmake
brew install opencv
brew install eigen
```

## Prepare Dataset

I use the dataset `cohn-kanade-jpg.zip` provided by the instructor, which can be found in DingTalk Chat.

Firstly unzip the dataset and rename the folder to `datasets`.

Then switch to the `prepare_image` directory and run the following command to prepare the dataset.

```bash
cd prepare_image
bash prepare_dataset.sh
```

This will generate the `train_dataset` and `test_dataset` in the root directory.

## Train Eigenfaces

Firstly switch to the root directory.

```bash
cd ..
```

Then run the following command to build the project.

```bash
rm -rf build
mkdir build && cd build
cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=1 ..
make -j
```

This will generate the `train` and `test` executable files in the `build` directory.

Then run the following command to train the eigenfaces:

```bash
./train <train_dataset_path> <energy_percentage> <output_path>
```

For example, to train the eigenfaces with 70% energy, you can use the following command:

```bash
./train ../train_dataset 0.7 ../train_results
```

This will generate the `train_results` directory in the root directory, which contains the mean face, eigenfaces, eigenfaces grid image and checkpoint.

## Test Eigenfaces

Firstly make sure you are in the `build` directory.

Then run the following command to test the eigenfaces:

```bash
./test <test_image_path> <checkpoint_path> <output_path>
```

For example, to test the eigenfaces on the first test image and save the result to the `../test_results` directory, you can use the following command:

```bash
./test ../test_dataset/S001/001.jpg ../train_results/checkpoint.txt ../test_results
```

This will generate the `test_results` directory in the root directory, which contains the annotated image and the closest training image.