#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <fstream>

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
    // check command line arguments
    if (argc != 4) {
        cout << "Correct usage: " << argv[0] << " <image path> <chessboard width points> <chessboard height points>" << endl;
        return -1;
    }
    
    string imagePath = argv[1];
    // ../examples/example.jpg
    string imageName = imagePath.substr(imagePath.find_last_of("/") + 1);
    imageName = imageName.substr(0, imageName.find_last_of("."));
    
    // read input image
    Mat image = imread(imagePath);
    if (image.empty()) {
        cout << "Failed to read image: " << imagePath << endl;
        return -1;
    }
    
    // create result directory
    string resultDir = "../examples/" + imageName + "_results";
    // check and create directory
    system(("mkdir -p \"" + resultDir + "\"").c_str());
    
    // chessboard size, need to be set manually
    Size boardSize(atoi(argv[2]), atoi(argv[3])); // chessboard corner points
    
    // find chessboard corners
    vector<Point2f> corners;
    bool found = findChessboardCorners(image, boardSize, corners);
    
    if (!found) {
        cout << "Failed to find chessboard corners! Please check the number of width and height points." << endl;
        return -1;
    }
    
    // improve corner accuracy
    Mat grayImage;
    cvtColor(image, grayImage, COLOR_BGR2GRAY);
    cornerSubPix(grayImage, corners, Size(11, 11), Size(-1, -1),
                TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.1));
    
    // draw corners
    Mat imageWithCorners = image.clone();
    drawChessboardCorners(imageWithCorners, boardSize, corners, found);
    namedWindow("detected corners", WINDOW_NORMAL);
    imshow("detected corners", imageWithCorners);
    imwrite("../examples/" + imageName + "_results/detected_corners.jpg", imageWithCorners);

    waitKey(0);

    // prepare camera calibration data
    vector<vector<Point2f>> imagePoints;
    imagePoints.push_back(corners);
    
    vector<vector<Point3f>> objectPoints(1);
    for (int i = 0; i < boardSize.height; i++) {
        for (int j = 0; j < boardSize.width; j++) {
            objectPoints[0].push_back(Point3f(j, i, 0.0f));
        }
    }
    
    // calibrate camera
    Mat cameraMatrix = Mat::eye(3, 3, CV_64F);
    Mat distCoeffs = Mat::zeros(8, 1, CV_64F);
    vector<Mat> rvecs, tvecs;
    
    double rms = calibrateCamera(objectPoints, imagePoints, image.size(),
                                cameraMatrix, distCoeffs, rvecs, tvecs);
    
    // print camera parameters to terminal
    cout << "-----------------------------------------" << endl;
    cout << "\033[33mcamera calibration RMS error: \033[0m" << rms << endl;
    cout << "\033[33mcamera intrinsic matrix: \033[0m" << endl << cameraMatrix << endl;
    cout << "\033[33mdistortion coefficients: \033[0m" << endl << distCoeffs << endl;
    cout << "-----------------------------------------" << endl;
    
    // print camera parameters to file
    ofstream outFile("../examples/" + imageName + "_results/camera_params.txt");
    if (outFile.is_open()) {
        outFile << "camera calibration RMS error: " << rms << endl;
        outFile << "camera intrinsic matrix: " << endl << cameraMatrix << endl;
        outFile << "distortion coefficients: " << endl << distCoeffs << endl;
        outFile.close();
    }
    
    // correct image
    Mat undistortedImage;
    undistort(image, undistortedImage, cameraMatrix, distCoeffs);
    
    namedWindow("undistorted image", WINDOW_NORMAL);
    imshow("undistorted image", undistortedImage);
    imwrite("../examples/" + imageName + "_results/undistorted_image.jpg", undistortedImage);

    waitKey(0);
    
    // bird's-eye view transformation

    // define the four corners of the chessboard in the original image
    Point2f srcQuad[4];
    // find the four corners of the chessboard
    srcQuad[0] = corners[0];                                   // top-left corner
    srcQuad[1] = corners[boardSize.width-1];                   // top-right corner
    srcQuad[2] = corners[boardSize.width*boardSize.height-1];  // bottom-right corner
    srcQuad[3] = corners[boardSize.width*(boardSize.height-1)]; // bottom-left corner
    
    // calculate the width and height of the chessboard in the original image (pixels)
    float widthChessboard = norm(srcQuad[1] - srcQuad[0]);
    float heightChessboard = norm(srcQuad[3] - srcQuad[0]);
    
    // calculate the width and height of the chessboard in the original image (pixels)
    float realWidthChessboard = (boardSize.width - 1);  // number of width points of the chessboard
    float realHeightChessboard = (boardSize.height - 1); // number of height points of the chessboard
    
    // calculate the size of each square in the chessboard in the original image (pixels)
    float squareSize = 30.0; // size of each square in the chessboard in the original image (pixels)
    
    // calculate the size of the chessboard in the target image (pixels)
    float dstWidth = squareSize * realWidthChessboard;
    float dstHeight = squareSize * realHeightChessboard;
    
    // calculate the width and height of the original image (pixels)
    float srcWidth = image.cols;
    float srcHeight = image.rows;
    
    // calculate the size of the birds-eye view (pixels)
    float scaleFactor = 1.5; // scale factor, ensure the birds-eye view is large enough
    float birdsEyeWidth = srcWidth * scaleFactor;
    float birdsEyeHeight = srcHeight * scaleFactor;
    
    // define the position of the chessboard in the birds-eye view (centered)
    Point2f dstQuad[4];
    float offsetX = (birdsEyeWidth - dstWidth) / 2;
    float offsetY = (birdsEyeHeight - dstHeight) / 2;
    dstQuad[0] = Point2f(offsetX, offsetY);                   // top-left corner
    dstQuad[1] = Point2f(offsetX + dstWidth, offsetY);        // top-right corner
    dstQuad[2] = Point2f(offsetX + dstWidth, offsetY + dstHeight);  // bottom-right corner
    dstQuad[3] = Point2f(offsetX, offsetY + dstHeight);       // bottom-left corner
    
    // calculate the perspective transformation matrix
    Mat perspectiveMatrix = getPerspectiveTransform(srcQuad, dstQuad);
    
    // apply perspective transformation to create a birds-eye view
    Mat birdsEyeView;
    warpPerspective(image, birdsEyeView, perspectiveMatrix, Size(birdsEyeWidth, birdsEyeHeight));
    
    // show birds-eye view
    namedWindow("birds-eye view", WINDOW_NORMAL);
    imshow("birds-eye view", birdsEyeView);
    
    // save birds-eye view
    imwrite("../examples/" + imageName + "_results/birds_eye_view.jpg", birdsEyeView);
    cout << "Results saved to ../examples/" + imageName + "_results" << endl;
    
    waitKey(0);

    return 0;
}