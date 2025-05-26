#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <boost/filesystem.hpp>
#include <algorithm> // For std::sort
#include <iomanip> // For std::setprecision
#include <stdexcept> // For std::runtime_error, std::invalid_argument

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp> // For cv2eigen and eigen2cv

namespace fs = boost::filesystem;

// Function declarations
cv::Mat flattenImage(const cv::Mat& img);
void loadImages(const std::string& datasetPath, std::vector<cv::Mat>& images, std::vector<std::string>& imagePaths, int& imgRows, int& imgCols);
void saveCheckpoint(const std::string& filename, int K, const Eigen::MatrixXd& eigenfaces, const Eigen::RowVectorXd& meanFace, const std::vector<std::string>& imagePaths, const Eigen::MatrixXd& weights);


int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <dataset_path> <energy_percentage> <save_path>" << std::endl;
        return 1;
    }

    double energyPercentage = 0.0;
    try {
        energyPercentage = std::stod(argv[2]);
        if (energyPercentage <= 0.0 || energyPercentage > 1.0) {
            throw std::invalid_argument("Energy percentage must be a positive value greater than 0.0 and less than or equal to 1.0.");
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: Invalid value for energy_percentage. " << e.what() << std::endl;
        return 1;
    }

    std::cout << "Starting Eigenface training with energy percentage = " << energyPercentage * 100 << "%" << std::endl;

    std::string datasetPath = argv[1];
    std::string savePath = argv[3];

    if (fs::exists(savePath)) {
        fs::remove_all(savePath);
    }
    fs::create_directory(savePath);

    std::vector<cv::Mat> images;
    std::vector<std::string> imagePaths;
    int imgRows = 0, imgCols = 0;

    // 1. Load images
    std::cout << "Loading images from " << datasetPath << "..." << std::endl;
    try {
        loadImages(datasetPath, images, imagePaths, imgRows, imgCols);
    } catch (const std::runtime_error& e) {
        std::cerr << "Error loading images: " << e.what() << std::endl;
        return 1;
    }


    if (images.empty()) {
        std::cerr << "No images loaded. Please check the dataset path and image files. Exiting." << std::endl;
        return 1;
    }
    std::cout << "Loaded " << images.size() << " images." << std::endl;
    std::cout << "Image dimensions: " << imgRows << "x" << imgCols << std::endl;

    int numImages = images.size();
    int D = imgRows * imgCols; // Dimensionality of each image vector




    // 2. Create data matrix (each row is a flattened image)
    Eigen::MatrixXd dataMatrix(numImages, D);
    for (int i = 0; i < numImages; ++i) {
        try {
            cv::Mat flat = flattenImage(images[i]); // Returns D x 1, CV_64F
            Eigen::VectorXd eigenVec;
            cv::cv2eigen(flat, eigenVec); // eigenVec is D x 1
            dataMatrix.row(i) = eigenVec.transpose(); // Store as 1 x D row
        } catch (const std::runtime_error& e) {
            std::cerr << "Error processing image " << imagePaths[i] << ": " << e.what() << std::endl;
            return 1;
        }
    }
    std::cout << "Data matrix created: " << dataMatrix.rows() << "x" << dataMatrix.cols() << std::endl;

    // 3. Calculate mean face
    Eigen::RowVectorXd meanFace = dataMatrix.colwise().mean(); // Results in a 1 x D row vector
    std::cout << "Mean face calculated." << std::endl;

    // convert meanFace to cv::Mat and reshape to original image size
    cv::Mat meanFaceToShow = cv::Mat(1, D, CV_64F);
    Eigen::Map<Eigen::MatrixXd>(meanFaceToShow.ptr<double>(), 1, D) = meanFace;
    meanFaceToShow = meanFaceToShow.reshape(1, imgRows);
    cv::imwrite(savePath + "/mean_face.png", meanFaceToShow);

    // 4. Calculate difference images (phi matrix, where each row is a difference image)
    Eigen::MatrixXd phiMatrix(numImages, D);
    for (int i = 0; i < numImages; ++i) {
        phiMatrix.row(i) = dataMatrix.row(i) - meanFace;
    }
    std::cout << "Difference images (phi matrix) calculated: " << phiMatrix.rows() << "x" << phiMatrix.cols() << std::endl;

    // 5. Build L = A^T * A (where A has difference images as columns)
    // Here, phiMatrix has rows as difference images (phi_i^T).
    // So, A = phiMatrix.transpose() (D x numImages).
    // L = A.transpose() * A = (phiMatrix.transpose()).transpose() * phiMatrix.transpose()
    // L = phiMatrix * phiMatrix.transpose() (numImages x numImages)
    Eigen::MatrixXd L = phiMatrix * phiMatrix.transpose();
    std::cout << "L = phiMatrix * phiMatrix.transpose() calculated: " << L.rows() << "x" << L.cols() << std::endl;


    // 6. Calculate eigenvalues and eigenvectors of L
    Eigen::EigenSolver<Eigen::MatrixXd> eigenSolver(L);
    // Correctly type eigenvaluesL_complex to store complex eigenvalues first
    Eigen::VectorXcd eigenvaluesL_complex = eigenSolver.eigenvalues(); 
    Eigen::MatrixXcd eigenvectorsL_complex = eigenSolver.eigenvectors();

    // Eigenvalues might be complex, take the real part. For symmetric matrix L, they should be real.
    Eigen::VectorXd eigenvaluesL = eigenvaluesL_complex.real();
    Eigen::MatrixXd eigenvectorsL = eigenvectorsL_complex.real(); // Columns are eigenvectors v_i

    std::cout << "Eigenvalues and eigenvectors of L calculated." << std::endl;

    // Sort eigenvalues and corresponding eigenvectors in descending order
    std::vector<std::pair<double, Eigen::VectorXd>> eigenPairs(numImages);
    for (int i = 0; i < numImages; ++i) {
        eigenPairs[i] = {eigenvaluesL(i), eigenvectorsL.col(i)};
    }

    std::sort(eigenPairs.begin(), eigenPairs.end(), [](const auto& a, const auto& b) {
        return a.first > b.first;
    });
    std::cout << "Eigenpairs sorted." << std::endl;

    // Determine K_actual based on energy percentage
    double totalEnergy = 0;
    for (const auto& pair : eigenPairs) {
        if (pair.first > 1e-9) { // Consider only significant positive eigenvalues
            totalEnergy += pair.first;
        }
    }

    double cumulativeEnergy = 0;
    int K_for_energy = 0;
    if (totalEnergy > 1e-9) { // Ensure there's some energy to begin with
        for (int i = 0; i < eigenPairs.size(); ++i) {
            if (eigenPairs[i].first < 1e-9) break; // Stop if eigenvalue is too small
            cumulativeEnergy += eigenPairs[i].first;
            K_for_energy++;
            if (cumulativeEnergy / totalEnergy >= energyPercentage) {
                break;
            }
        }
    }
    if (K_for_energy == 0 && numImages > 0) {
        std::cout << "Warning: No eigenvalues contribute significantly to energy, or total energy is zero. Defaulting to 1 eigenface if possible." << std::endl;
        K_for_energy = (numImages > 0) ? 1 : 0; // Use at least one if available, and if training images exist
    }
    if (K_for_energy > numImages) K_for_energy = numImages; // Cap at numImages

    int K_actual = K_for_energy;
    std::cout << "Selected K_actual = " << K_actual << " to capture at least " << energyPercentage * 100 << "% of total energy (" << (totalEnergy > 0 ? (cumulativeEnergy / totalEnergy)*100 : 0) << "% captured)." << std::endl;

    // 7. Select top K_actual eigenfaces (u_i = A * v_i)
    // A = phiMatrix.transpose()
    // Initialize eigenfaces matrix with the determined K_actual
    if (K_actual == 0 && numImages > 0) {
        std::cerr << "Error: K_actual is 0 even after energy calculation, but there are images. Cannot proceed." << std::endl;
        return 1;
    } else if (K_actual == 0 && numImages == 0) {
        // This is a valid case if no images were loaded, K_actual will be 0.
        // The program should handle this gracefully, e.g. by not proceeding with weights etc.
        std::cout << "No images loaded, so K_actual is 0. Eigenface calculation will be skipped." << std::endl;
    }


    Eigen::MatrixXd eigenfaces; // Declare first
    if (K_actual > 0) {
        eigenfaces.resize(D, K_actual); // Resize with determined K_actual
        for (int i = 0; i < K_actual; ++i) {
            // We already checked for significant eigenvalues when determining K_actual based on energy.
            // However, eigenPairs[i] should still be valid up to K_actual.
            Eigen::VectorXd v_i = eigenPairs[i].second; // numImages x 1
            Eigen::VectorXd u_i = phiMatrix.transpose() * v_i; // (D x numImages) * (numImages x 1) = D x 1
            eigenfaces.col(i) = u_i.normalized();
        }
    } else {
        // If K_actual is 0, create an empty eigenfaces matrix (0 columns)
        // This might be D x 0 or 0 x 0 depending on Eigen's behavior for 0 columns.
        // Let's make it D x 0 explicitly for clarity if D is known and positive.
        eigenfaces.resize(D, 0);
        std::cout << "K_actual is 0, so no eigenfaces will be computed or stored." << std::endl;
    }

    // The old loop for K_param should be replaced by the logic above.
    // The old K_actual incrementing and K_param break logic is now handled by energy calculation.
    // The conservativeResize after the loop is also not needed if initialized/resized correctly before.

    // save eigenfaces to savePath
    for (int i = 0; i < K_actual; ++i) {
        cv::Mat eigenfaceToShowRaw = cv::Mat(1, D, CV_64F); 
        Eigen::Map<Eigen::MatrixXd>(eigenfaceToShowRaw.ptr<double>(), 1, D) = eigenfaces.col(i).transpose();
        eigenfaceToShowRaw = eigenfaceToShowRaw.reshape(1, imgRows); 

        cv::Mat eigenfaceToShowNormalized;
        // normalize to 0-255
        cv::normalize(eigenfaceToShowRaw, eigenfaceToShowNormalized, 0, 255, cv::NORM_MINMAX, CV_8U);
        
        // save normalized eigenface
        cv::imwrite(savePath + "/eigenface_" + std::to_string(i) + ".png", eigenfaceToShowNormalized);
    }

    // put the top 9 eigenfaces into an image (3x3 grid, total size = 1080x900) and save
    if (K_actual > 0) {
        cv::Mat gridImage = cv::Mat::zeros(imgRows * 3, imgCols * 3, CV_8U);
        
        // iterate over the top 9 eigenfaces
        for (int i = 0; i < 9 && i < K_actual; ++i) {
            // calculate the position of the eigenface in the grid
            int row = i / 3;  // 0, 0, 0, 1, 1, 1, 2, 2, 2
            int col = i % 3;  // 0, 1, 2, 0, 1, 2, 0, 1, 2
            
            // convert the eigenface to an image
            cv::Mat eigenfaceToShowRaw = cv::Mat(1, D, CV_64F);
            Eigen::Map<Eigen::MatrixXd>(eigenfaceToShowRaw.ptr<double>(), 1, D) = eigenfaces.col(i).transpose();
            eigenfaceToShowRaw = eigenfaceToShowRaw.reshape(1, imgRows);
            
            // normalize to 0-255
            cv::Mat eigenfaceNormalized;
            cv::normalize(eigenfaceToShowRaw, eigenfaceNormalized, 0, 255, cv::NORM_MINMAX, CV_8U);
            
            // copy the eigenface to the grid image
            cv::Rect roi(col * imgCols, row * imgRows, imgCols, imgRows);
            eigenfaceNormalized.copyTo(gridImage(roi));
        }
        
        // save the grid image
        cv::imwrite(savePath + "/eigenfaces_grid.png", gridImage);
    }

    std::cout << "Top " << K_actual << " eigenfaces calculated and normalized." << std::endl;
    std::cout << "Eigenfaces matrix U: " << eigenfaces.rows() << "x" << eigenfaces.cols() << std::endl;

    // 8. Project training images onto eigenfaces to get weights
    // Weights W_ik = phi_i^T * u_k.
    // Row i of W is phi_i^T * U
    // phiMatrix.row(i) is (1 x D)
    // eigenfaces is U (D x K_actual)
    // weights.row(i) = phiMatrix.row(i) * eigenfaces ( (1xD) * (DxK_actual) = 1xK_actual )
    Eigen::MatrixXd weights(numImages, K_actual);
    if (K_actual > 0) { // Only compute weights if there are eigenfaces
        for (int i = 0; i < numImages; ++i) {
            weights.row(i) = phiMatrix.row(i) * eigenfaces;
        }
        std::cout << "Weights for training images calculated: " << weights.rows() << "x" << weights.cols() << std::endl;
    } else {
        std::cout << "No eigenfaces to project onto, weights matrix will be empty or not used." << std::endl;
        // weights matrix is already numImages x 0 if K_actual is 0
    }
    

    // 9. Save K_actual, meanFace, eigenfaces, and weights to checkpoint.txt
    std::cout << "Saving checkpoint to " << savePath + "/checkpoint.txt" << "..." << std::endl;
    saveCheckpoint(savePath + "/checkpoint.txt", K_actual, eigenfaces, meanFace, imagePaths, weights);
    std::cout << "Training complete. Checkpoint saved with K_actual = " << K_actual << "." << std::endl;

    return 0;
}

// Function to flatten an image (grayscale) into a 1D column vector of doubles
cv::Mat flattenImage(const cv::Mat& img) {
    if (img.empty()) {
        throw std::runtime_error("Cannot flatten empty image.");
    }
    // Ensure it's a single channel image (already done in loadImages, but good check)
    cv::Mat grayImg;
    if (img.channels() != 1) {
         //This case should ideally not be hit if loadImages enforces grayscale
        cv::cvtColor(img, grayImg, cv::COLOR_BGR2GRAY); // Or other appropriate conversion
    } else {
        grayImg = img.clone();
    }

    // Convert to CV_64F (double) for precision with Eigen
    cv::Mat floatImg;
    grayImg.convertTo(floatImg, CV_64F);

    // Reshape to a single column vector (D x 1)
    // Mat::reshape(int cn, int rows) where cn is new number of channels, rows is new number of rows
    // To get a D x 1 vector from an H x W image (D = H*W):
    // reshape(1, D) means 1 channel, D rows.
    return floatImg.reshape(1, floatImg.total()); // floatImg.total() is D = rows*cols
}


// Function to load images from the dataset directory
// Assumes structure: datasetPath/S*/image_files
void loadImages(const std::string& datasetPath,
                std::vector<cv::Mat>& images,
                std::vector<std::string>& imagePaths,
                int& imgRows, int& imgCols) {
    images.clear();
    imagePaths.clear();
    bool firstImage = true;
    int firstImgRows = 0, firstImgCols = 0;

    if (!fs::exists(datasetPath) || !fs::is_directory(datasetPath)) {
        throw std::runtime_error("Dataset path does not exist or is not a directory: " + datasetPath);
    }

    for (const auto& subjectEntry : fs::directory_iterator(datasetPath)) {
        if (subjectEntry.is_directory()) {
            std::string subjectName = subjectEntry.path().filename().string();
            if (subjectName.rfind("S", 0) == 0) { // Check if directory starts with 'S'
                std::cout << "  Processing subject directory: " << subjectEntry.path().string() << std::endl;
                int imageCountInSubject = 0;
                for (const auto& imageEntry : fs::directory_iterator(subjectEntry.path())) {
                    if (imageEntry.is_regular_file()) {
                        std::string pathStr = imageEntry.path().string();
                        // Load as grayscale directly
                        cv::Mat img = cv::imread(pathStr, cv::IMREAD_GRAYSCALE);

                        if (img.empty()) {
                            std::cerr << "    Warning: Could not load image " << pathStr << ". Skipping." << std::endl;
                            continue;
                        }

                        if (firstImage) {
                            firstImgRows = img.rows;
                            firstImgCols = img.cols;
                            imgRows = firstImgRows; // Set global imgRows/Cols
                            imgCols = firstImgCols;
                            firstImage = false;
                            std::cout << "    First image loaded: " << pathStr << ", dimensions: " << imgRows << "x" << imgCols << std::endl;
                        } else {
                            if (img.rows != firstImgRows || img.cols != firstImgCols) {
                                std::cerr << "    Error: Image " << pathStr << " has different dimensions ("
                                          << img.rows << "x" << img.cols
                                          << ") than expected (" << firstImgRows << "x" << firstImgCols
                                          << "). All images must have the same dimensions." << std::endl;
                                throw std::runtime_error("Images have inconsistent dimensions. Please ensure all training images are of the same size.");
                            }
                        }
                        images.push_back(img);
                        imagePaths.push_back(pathStr);
                        imageCountInSubject++;
                    }
                }
                 if(imageCountInSubject == 0){
                    std::cout << "    Warning: No images found in subject directory: " << subjectEntry.path().string() << std::endl;
                }
            }
        }
    }

    if (firstImage && images.empty()) { // No images found at all
        imgRows = 0; imgCols = 0; // Reset if no valid images found
        throw std::runtime_error("No images found in the dataset directory that match the S*/ criteria or are loadable.");
    }
}

// Function to save K, meanFace, eigenfaces, image paths and weights to checkpoint.txt
void saveCheckpoint(const std::string& filename, int K_actual,
                    const Eigen::MatrixXd& eigenfaces, // D x K_actual, each col is an eigenface
                    const Eigen::RowVectorXd& meanFace, // 1 x D
                    const std::vector<std::string>& imagePaths,
                    const Eigen::MatrixXd& weights) { // numImages x K_actual
    std::ofstream outFile(filename);
    if (!outFile.is_open()) {
        std::cerr << "Error: Could not open checkpoint file " << filename << " for writing." << std::endl;
        return;
    }

    outFile << std::fixed << std::setprecision(8); // Set precision for floating point numbers

    // 1. Save K_actual (number of eigenfaces actually used)
    outFile << K_actual << std::endl;

    // 2. Save Mean Face (1 x D)
    if (meanFace.size() > 0) {
        for (int i = 0; i < meanFace.cols(); ++i) {
            outFile << meanFace(i) << (i == meanFace.cols() - 1 ? "" : " ");
        }
    }
    outFile << std::endl;


    // 3. Save Eigenfaces (D x K_actual)
    // Each eigenface is a column vector of size D. Store it as a row in the file.
    if (K_actual > 0) {
        for (int k = 0; k < eigenfaces.cols(); ++k) { // Iterate through K_actual eigenfaces
            for (int i = 0; i < eigenfaces.rows(); ++i) { // Iterate through D pixels
                outFile << eigenfaces(i, k) << (i == eigenfaces.rows() - 1 ? "" : " ");
            }
            outFile << std::endl;
        }
    }


    // 4. Save number of training images for which weights are stored
    outFile << imagePaths.size() << std::endl;

    // 5. Save weights for each training image (numImages x K_actual)
    // Also save corresponding image path for reference
    if (K_actual > 0) { // Only save weights if there are eigenfaces
        for (int i = 0; i < weights.rows(); ++i) {
            outFile << imagePaths[i]; // Save image path
            for (int k = 0; k < weights.cols(); ++k) {
                outFile << " " << weights(i, k);
            }
            outFile << std::endl;
        }
    } else if (!imagePaths.empty()) { // K_actual is 0, but there are images
        // Save image paths even if no weights
        for (const auto& path : imagePaths) {
            outFile << path << std::endl; // Path followed by an empty line or just path
        }
    }


    outFile.close();
    if (outFile.fail()) {
        std::cerr << "Error writing to checkpoint file " << filename << ". Disk full or permissions issue?" << std::endl;
    }
}
