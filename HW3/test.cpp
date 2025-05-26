#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <boost/filesystem.hpp>
#include <map>
#include <limits> // Required for std::numeric_limits

#include <Eigen/Dense>

#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp> // For cv::cv2eigen

namespace fs = boost::filesystem;

// Helper function to extract student ID from path like ../train_dataset/SXXX/YYY.jpg -> SXXX
std::string extractStudentID(const std::string& path) {
    size_t lastSlash = path.find_last_of('/');
    if (lastSlash == std::string::npos) {
        return "Unknown"; // Or handle error appropriately
    }
    std::string dirOnly = path.substr(0, lastSlash);
    size_t secondLastSlash = dirOnly.find_last_of('/');
    if (secondLastSlash == std::string::npos) {
        return "Unknown"; // Or handle error appropriately
    }
    return dirOnly.substr(secondLastSlash + 1);
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <test_image_path> <checkpoint_path> <output_path>" << std::endl;
        return 1;
    }

    std::string testImagePath = argv[1];
    std::string checkpointPath = argv[2];
    std::string outputPath = argv[3];

    if (fs::exists(outputPath)) {
        fs::remove_all(outputPath);
    }
    fs::create_directory(outputPath);
    /*
    format of checkpoint.txt:
        num_of_eigenfaces: K
        mean_face: D double values separated by spaces
        eigenfaces: K lines, each line contains D double values separated by spaces
        num_of_training_images: N
        paths and weights: N lines, each line contains an image path and N double values separated by spaces, format like:
            ../train_dataset/S038/002.jpg -9998.52164415 -6094.17717782 1216.31550027 -3232.36227185 -3286.95100986 682.36149513 -2429.55918185 1239.15988511
    */

    // 1. get width, height, D from test image
    cv::Mat testImage = cv::imread(testImagePath, cv::IMREAD_GRAYSCALE);
    if (testImage.empty()) {
        std::cerr << "Error: Could not open test image " << testImagePath << " for reading." << std::endl;
        return 1;
    }
    int width = testImage.cols;
    int height = testImage.rows;
    int D = width * height;

    // 2. Load data from checkpoint.txt
    std::cout << "Loading data from checkpoint: " << checkpointPath << std::endl;
    
    std::ifstream checkpointFile(checkpointPath);
    if (!checkpointFile.is_open()) {
        std::cerr << "Error: Could not open checkpoint file " << checkpointPath << " for reading." << std::endl;
        return 1;
    }

    // Read K (number of eigenfaces)
    int K;
    checkpointFile >> K;
    if (K <= 0) {
        std::cerr << "Error: K=" << K << " from checkpoint. Cannot perform recognition with non-positive K." << std::endl;
        return 1;
    }
    std::cout << "  K (Number of eigenfaces): " << K << std::endl;

    // Read Mean Face (1 x D)
    Eigen::RowVectorXd meanFaceEigen(D);
    for (int i = 0; i < D; ++i) {
        checkpointFile >> meanFaceEigen(i);
    }
    // std::cout << "  Mean face loaded." << std::endl;

    // Read Eigenfaces (D x K)
    Eigen::MatrixXd eigenfaces(D, K);
    for (int k_idx = 0; k_idx < K; ++k_idx) { // K eigenfaces (columns)
        for (int i = 0; i < D; ++i) {     // D pixels (rows)
            checkpointFile >> eigenfaces(i, k_idx);
        }
    }
    std::cout << "  Eigenfaces loaded: " << eigenfaces.rows() << "x" << eigenfaces.cols() << std::endl;

    // Read number of training images and their weights
    int numTrainingImages;
    checkpointFile >> numTrainingImages;
    std::string line;
    std::getline(checkpointFile, line); // Consume the rest of the numTrainingImages line

    std::vector<std::pair<std::string, Eigen::VectorXd>> trainingSamples; // <image_path, weights_Kx1>
    std::map<std::string, std::vector<Eigen::VectorXd>> studentTrainingWeights; // <studentID, list_of_weights_Kx1>

    // std::cout << "  Loading " << numTrainingImages << " training image weights..." << std::endl;
    for (int i = 0; i < numTrainingImages; ++i) {
        std::getline(checkpointFile, line);
        std::stringstream ss(line);
        std::string imagePath;
        ss >> imagePath; // First part is path

        Eigen::VectorXd currentTrainingWeights(K);
        for (int k_idx = 0; k_idx < K; ++k_idx) {
            ss >> currentTrainingWeights(k_idx);
        }
        trainingSamples.push_back({imagePath, currentTrainingWeights});

        std::string studentID = extractStudentID(imagePath);
        studentTrainingWeights[studentID].push_back(currentTrainingWeights);
    }
    checkpointFile.close();
    std::cout << "  Checkpoint data loaded successfully." << std::endl;

    // 3. Process Test Image: Flatten, convert to double, subtract mean, project to get weights
    // Flatten testImage (CV_8U HxW) to (CV_64F Dx1)
    cv::Mat flatTestImageCV_8U = testImage.reshape(1, D); // D rows, 1 channel (becomes Dx1)
    cv::Mat flatTestImageCV_64F;
    flatTestImageCV_8U.convertTo(flatTestImageCV_64F, CV_64F);

    Eigen::VectorXd flatTestImageEigen(D);
    cv::cv2eigen(flatTestImageCV_64F, flatTestImageEigen);

    // Subtract mean face (meanFaceEigen is 1xD, needs transpose to become Dx1)
    Eigen::VectorXd diffImageEigen = flatTestImageEigen - meanFaceEigen.transpose();

    // Project onto eigenfaces to get weights for the test image
    // Weights_test (Kx1) = Eigenfaces^T (KxD) * DiffImage (Dx1)
    Eigen::VectorXd testImageWeights = eigenfaces.transpose() * diffImageEigen;
    // std::cout << "Test image weights computed: " << testImageWeights.rows() << "x" << testImageWeights.cols() << std::endl;

    // 4. Find the closest training image
    if (trainingSamples.empty()) {
        std::cerr << "Error: No training samples loaded from checkpoint. Cannot find closest image." << std::endl;
        return 1;
    }

    double minDistanceImage = std::numeric_limits<double>::max();
    std::string closestImagePath = "N/A";

    for (const auto& sample : trainingSamples) {
        double dist = (testImageWeights - sample.second).norm(); // Euclidean distance
        if (dist < minDistanceImage) {
            minDistanceImage = dist;
            closestImagePath = sample.first;
        }
    }
    std::cout << "Closest training image: " << closestImagePath << " (Distance: " << minDistanceImage << ")" << std::endl;

    cv::imwrite(outputPath + "/closest_training_image.jpg", cv::imread(closestImagePath, cv::IMREAD_GRAYSCALE));

    // 5. Identify the best match student
    if (studentTrainingWeights.empty()) {
        std::cerr << "Error: No student training weights available. Cannot identify student." << std::endl;
        return 1;
    }
    
    double minAvgStudentDistance = std::numeric_limits<double>::max();
    std::string bestMatchStudentID = "N/A";

    for (const auto& studentEntry : studentTrainingWeights) {
        const std::string& studentID = studentEntry.first;
        const std::vector<Eigen::VectorXd>& weightsList = studentEntry.second;
        
        if (weightsList.empty()) continue;

        double currentStudentTotalDistance = 0;
        for (const auto& w : weightsList) {
            currentStudentTotalDistance += (testImageWeights - w).norm();
        }
        double avgDist = currentStudentTotalDistance / weightsList.size();

        if (avgDist < minAvgStudentDistance) {
            minAvgStudentDistance = avgDist;
            bestMatchStudentID = studentID;
        }
    }
    std::cout << "Best match student ID: " << bestMatchStudentID << " (Avg Distance: " << minAvgStudentDistance << ")" << std::endl;

    // 6. Annotate the original test image and save it
    cv::Mat displayImage = cv::imread(testImagePath, cv::IMREAD_COLOR);
    if (displayImage.empty()) { // Fallback to grayscale if color load fails
        cv::cvtColor(testImage, displayImage, cv::COLOR_GRAY2BGR);
    }

    std::string textToDisplay = "Test Result: " + bestMatchStudentID;
    cv::Point textOrg(10, 30); // Coordinates of the bottom-left corner of the text string in the image
    int fontFace = cv::FONT_HERSHEY_SIMPLEX;
    double fontScale = 1.0;
    cv::Scalar textColor(0, 255, 0); // Green
    int thickness = 2;
    int lineType = cv::LINE_AA;

    cv::putText(displayImage, textToDisplay, textOrg, fontFace, fontScale, textColor, thickness, lineType);

    if (cv::imwrite(outputPath + "/annotated_image.jpg", displayImage)) {
        std::cout << "Annotated image saved to: " << outputPath << std::endl;
    } else {
        std::cerr << "Error: Could not save annotated image to " << outputPath << std::endl;
        return 1;
    }

    return 0;
}