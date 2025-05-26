#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
    // missing image path
    if (argc != 3) {
        cout << "Usage: ./ellipse_fit <image_path> <output_path>" << endl;
        return -1;
    }

    // read input image
    Mat src = imread(argv[1], IMREAD_COLOR);
    if (src.empty()) {
        cout << "Could not open or find the image!" << endl;
        return -1;
    }

    // convert to gray image
    Mat gray;
    cvtColor(src, gray, COLOR_BGR2GRAY);

    // binary processing
    Mat binary;
    threshold(gray, binary, 127, 255, THRESH_BINARY);

    // find contours
    vector<vector<Point>> contours;
    findContours(binary, contours, RETR_LIST, CHAIN_APPROX_NONE);

    // draw all contours on the original image (green)
    drawContours(src, contours, -1, Scalar(0, 255, 0), 2);

    // fit ellipse for each contour
    for (size_t i = 0; i < contours.size(); i++) {
        // skip contours with less than 5 points
        if (contours[i].size() < 5) {
            continue;
        }

        // fit ellipse
        RotatedRect fitted_ellipse = fitEllipse(contours[i]);

        // draw fitted ellipse (red)
        ellipse(src, fitted_ellipse, Scalar(0, 0, 255), 2);

        // output ellipse parameters
        cout << "-----------------------------------------------" << endl;
        cout << "Contour " << i << " ellipse parameters:" << endl;
        cout << "    Center: " << fitted_ellipse.center << endl;
        cout << "    Size: " << fitted_ellipse.size << endl;
        cout << "    Angle: " << fitted_ellipse.angle << " degrees" << endl;
        cout << "-----------------------------------------------" << endl;
    }

    // show results
    imshow("Ellipse Fit", src);
    waitKey(0);

    // save figure
    string output_path = argv[2];
    imwrite(output_path, src);

    return 0;
}