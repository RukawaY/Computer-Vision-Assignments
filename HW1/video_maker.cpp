#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <boost/filesystem.hpp>

using namespace cv;
using namespace std;
namespace fs = boost::filesystem;

// global configuration
const Size OUTPUT_SIZE(720, 540); // output video size
const int FPS = 24;              // output video frame rate
const int SLIDE_DURATION = 3;    // duration of each slide
const int TITLE_DURATION = 3;    // duration of title
const string STUDENT_INFO = "Student ID: 3230103043  Name: Xia Ziyuan"; // student information

// add text to the center of the image
void addTextCentered(Mat& frame, const string& text, double fontScale = 1.0, Scalar color = Scalar(255, 255, 255)) {
    int baseline = 0;
    Size textSize = getTextSize(text, FONT_HERSHEY_SIMPLEX, fontScale, 2, &baseline);
    Point textOrg((frame.cols - textSize.width) / 2, (frame.rows + textSize.height) / 2);
    putText(frame, text, textOrg, FONT_HERSHEY_SIMPLEX, fontScale, color, 2);
}

// add bottom text
void addBottomText(Mat& frame, const string& text) {
    int baseline = 0;
    Size textSize = getTextSize(text, FONT_HERSHEY_SIMPLEX, 0.9, 2, &baseline);
    Point textOrg((frame.cols - textSize.width) / 2, frame.rows - 10);
    putText(frame, text, textOrg, FONT_HERSHEY_SIMPLEX, 0.9, Scalar(255, 255, 255), 2);
}

// cross fade transition effect
void crossFade(const Mat& src1, const Mat& src2, Mat& dst, double alpha) {
    addWeighted(src1, 1.0 - alpha, src2, alpha, 0.0, dst);
}

// main function
int main(int argc, char** argv) {
    // check command line arguments
    if (argc != 2) {
        cout << "Usage: " << argv[0] << " <folder_path>" << endl;
        return -1;
    }

    string folderPath = argv[1];
    vector<string> imagePaths;
    string videoPath;

    // read folder content
    try {
        for (const auto& entry : fs::directory_iterator(folderPath)) {
            string path = entry.path().string();
            string ext = entry.path().extension().string();
            
            transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            
            if (ext == ".jpg" || ext == ".jpeg" || ext == ".png") {
                imagePaths.push_back(path);
            } else if (ext == ".avi" || ext == ".mp4") {
                videoPath = path;
            }
        }
    } catch (const fs::filesystem_error& e) {
        cerr << "Error reading directory: " << e.what() << endl;
        return -1;
    }

    // check file number
    if (imagePaths.size() < 5) {
        cerr << "Need at least 5 images in the folder" << endl;
        return -1;
    }
    if (videoPath.empty()) {
        cerr << "No video file found in the folder" << endl;
        return -1;
    }

    // create video writer
    string outputPath = folderPath + "/output_video.avi";
    VideoWriter writer(outputPath, VideoWriter::fourcc('M', 'J', 'P', 'G'), FPS, OUTPUT_SIZE);
    if (!writer.isOpened()) {
        cerr << "Could not open the output video for write" << endl;
        return -1;
    }

    // 1. generate title
    Mat titleFrame(OUTPUT_SIZE, CV_8UC3, Scalar(50, 50, 150));
    addTextCentered(titleFrame, "Xia Ziyuan's Generated Video", 1.4);
    addBottomText(titleFrame, STUDENT_INFO);
    
    for (int i = 0; i < TITLE_DURATION * FPS; ++i) {
        writer.write(titleFrame);
    }

    // 2. slide show
    vector<Mat> images;
    for (const auto& path : imagePaths) {
        Mat img = imread(path);
        if (img.empty()) {
            cerr << "Could not read image: " << path << endl;
            continue;
        }
        
        // adjust image size
        Mat resizedImg;
        resize(img, resizedImg, OUTPUT_SIZE);
        images.push_back(resizedImg);
    }

    // add transition effect
    for (size_t i = 0; i < images.size(); ++i) {
        // show current image
        for (int j = 0; j < SLIDE_DURATION * FPS * 0.8; ++j) {
            Mat frame = images[i].clone();
            addBottomText(frame, STUDENT_INFO);
            writer.write(frame);
        }

        // if not the last image, add transition effect
        if (i < images.size() - 1) {
            for (int j = 0; j < SLIDE_DURATION * FPS * 0.2; ++j) {
                double alpha = static_cast<double>(j) / (SLIDE_DURATION * FPS * 0.2);
                Mat frame;
                crossFade(images[i], images[i+1], frame, alpha);
                addBottomText(frame, STUDENT_INFO);
                writer.write(frame);
            }
        }
    }

    // 3. play original video
    VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        cerr << "Could not open video: " << videoPath << endl;
        return -1;
    }

    Mat frame;
    while (cap.read(frame)) {
        Mat resizedFrame;
        resize(frame, resizedFrame, OUTPUT_SIZE);
        addBottomText(resizedFrame, STUDENT_INFO);
        writer.write(resizedFrame);
    }

    // release resources
    cap.release();
    writer.release();

    cout << "Video created successfully: " << outputPath << endl;
    return 0;
}