#include <iostream>
#include <vector>
#include <string>
#include <algorithm> // For std::sort, std::transform
#include <boost/filesystem.hpp> // For C++17 filesystem operations
#include <cctype>     // For std::tolower

#include <opencv2/opencv.hpp> // Main OpenCV header
#include <opencv2/imgproc.hpp>  // Image processing
#include <opencv2/highgui.hpp>  // High-level GUI (imread, imwrite)
#include <opencv2/objdetect.hpp> // Object detection (CascadeClassifier)

namespace fs = boost::filesystem;
using namespace fs;

// 函数：处理单张图像
void process_image(const std::string& image_path_str, 
                   const std::string& output_path_str, 
                   cv::CascadeClassifier& eye_cascade) {
    // 加载图像
    cv::Mat img = cv::imread(image_path_str);
    if (img.empty()) {
        std::cerr << "错误：无法读取图像 " << image_path_str << std::endl;
        return;
    }

    // 转换为灰度图并进行直方图均衡化以提高检测对比度
    cv::Mat gray_img_for_detection;
    cv::cvtColor(img, gray_img_for_detection, cv::COLOR_BGR2GRAY);
    cv::equalizeHist(gray_img_for_detection, gray_img_for_detection);

    // 检测眼睛
    std::vector<cv::Rect> eyes;
    // detectMultiScale 参数:
    //   image: 输入图像 (CV_8U 类型)
    //   objects: 检测到的对象矩形框
    //   scaleFactor: 每次图像缩小的比例
    //   minNeighbors: 每个候选矩形至少应有多少邻居才被保留 (值越高，检测越严格，漏检可能增加，但误检减少)
    //   flags: 旧版级联分类器的标志，新版未使用
    //   minSize: 最小检测对象尺寸
    //   maxSize: 最大检测对象尺寸 (可选)
    eye_cascade.detectMultiScale(gray_img_for_detection, eyes, 1.1, 5, 0, cv::Size(20, 10));

    if (eyes.size() == 2) {
        // 获取眼睛中心点
        std::vector<cv::Point2f> eye_centers(2);
        for (size_t i = 0; i < 2; ++i) {
            eye_centers[i] = cv::Point2f(static_cast<float>(eyes[i].x + eyes[i].width * 0.5), 
                                         static_cast<float>(eyes[i].y + eyes[i].height * 0.5));
        }

        // 根据x坐标确保眼睛顺序（左眼在前）
        if (eye_centers[0].x > eye_centers[1].x) {
            std::swap(eye_centers[0], eye_centers[1]);
        }

        // 源点（检测到的眼睛中心）和目标点（期望的眼睛位置）
        std::vector<cv::Point2f> src_pts = {eye_centers[0], eye_centers[1]};
        std::vector<cv::Point2f> dst_pts = {cv::Point2f(100.0f, 160.0f), 
                                            cv::Point2f(200.0f, 160.0f)};
        
        // 估算仿射变换矩阵（相似变换：平移、旋转、均匀缩放）
        // cv::estimateAffinePartial2D 至少需要两个点对
        cv::Mat M = cv::estimateAffinePartial2D(src_pts, dst_pts);
        
        if (M.empty()) {
            std::cerr << "警告：无法为图像 " << image_path_str << " 估算仿射变换矩阵。跳过此图像。" << std::endl;
            return;
        }

        // 应用仿射变换到原始彩色图像
        cv::Mat final_gray_img;
        // 目标图像大小为 300x360
        // INTER_LINEAR: 双线性插值
        // BORDER_CONSTANT: 边界外填充常数
        // cv::Scalar(0,0,0): 填充黑色
        cv::warpAffine(img, final_gray_img, M, cv::Size(300, 360), 
                       cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0,0,0));

        // 归一化灰度图像的值到 0-255 范围
        cv::Mat normalized_img;
        cv::normalize(final_gray_img, normalized_img, 0, 255, cv::NORM_MINMAX, CV_8UC1);

        // 保存处理后的图像
        if (!cv::imwrite(output_path_str, normalized_img)) {
            std::cerr << "错误：无法将处理后的图像保存到 " << output_path_str << std::endl;
        } else {
            std::cout << "已处理并保存: " << output_path_str << std::endl;
        }

    } else {
        std::cout << "信息：在图像 " << image_path_str << " 中检测到 " << eyes.size() << " 只眼睛。预期为2只。跳过此图像。" << std::endl;
    }
}


int main() {
    std::string base_path_str = "../../datasets/";
    std::string eye_cascade_path = "../haarcascade_eye_tree_eyeglasses.xml"; 

    cv::CascadeClassifier eye_cascade;
    if (!eye_cascade.load(eye_cascade_path)) {
        std::cerr << "错误：无法从 " << eye_cascade_path << " 加载眼睛检测级联分类器。" << std::endl;
        std::cerr << "请确保XML文件 (" << eye_cascade_path << ") 与可执行文件在同一目录，或提供其完整路径。" << std::endl;
        return -1;
    }

    fs::path base_fs_path(base_path_str);
    if (!fs::exists(base_fs_path) || !fs::is_directory(base_fs_path)) {
        std::cerr << "错误：数据集路径 " << base_path_str << " 不存在或不是一个目录。" << std::endl;
        return -1;
    }
    
    std::cout << "开始处理图像..." << std::endl;

    try {
        // 遍历 ./datasets/ 下的每个子目录 (S001, S002, ...)
        for (const auto& student_entry : fs::directory_iterator(base_fs_path)) {
            if (student_entry.is_directory()) {
                fs::path student_dir_fs_path = student_entry.path();
                // if (student_dir_fs_path.filename().string().rfind("S", 0) == 0)
                std::cout << "正在处理目录: " << student_dir_fs_path.string() << std::endl;

                // 遍历学生目录下的每个文件
                for (const auto& img_entry : fs::directory_iterator(student_dir_fs_path)) {
                    if (img_entry.is_regular_file()) {
                        fs::path original_img_fs_path = img_entry.path();
                        std::string original_img_path_str = original_img_fs_path.string();
                        
                        std::string stem = original_img_fs_path.stem().string(); // 文件名（不含扩展名）
                        std::string extension = original_img_fs_path.extension().string(); // 文件扩展名 (例如 .jpg)

                        // 将扩展名转为小写以进行不区分大小写的比较
                        std::string ext_lower = extension;
                        std::transform(ext_lower.begin(), ext_lower.end(), ext_lower.begin(),
                                       [](unsigned char c){ return std::tolower(c); });

                        // 检查是否为常见的图像文件扩展名
                        const std::vector<std::string> image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"};
                        bool is_image = false;
                        for (const auto& ext_cmp : image_extensions) {
                            if (ext_lower == ext_cmp) {
                                is_image = true;
                                break;
                            }
                        }
                        if (!is_image) {
                            // std::cout << "跳过非图像文件: " << original_img_path_str << std::endl;
                            continue; // 不是支持的图像文件，跳过
                        }

                        // 跳过已经处理过的文件 (文件名以 "_cropped" 结尾)
                        if (stem.length() >= 8 && stem.substr(stem.length() - 8) == "_cropped") {
                            // std::cout << "跳过已处理文件: " << original_img_path_str << std::endl;
                            continue;
                        }
                        
                        // 构建输出文件名和路径
                        std::string output_file_name = stem + "_cropped" + extension;
                        fs::path output_file_fs_path = student_dir_fs_path / output_file_name; // 存放在原图同一路径下
                        
                        process_image(original_img_path_str, output_file_fs_path.string(), eye_cascade);
                    }
                }
            }
        }
    } catch (const fs::filesystem_error& e) {
        std::cerr << "文件系统错误: " << e.what() << std::endl;
        return -1;
    } catch (const cv::Exception& e) { // 捕获 OpenCV 异常
        std::cerr << "OpenCV 错误: " << e.what() << std::endl;
        return -1;
    } catch (const std::exception& e) { // 捕获其他标准异常
        std::cerr << "标准异常: " << e.what() << std::endl;
        return -1;
    } catch (...) { // 捕获所有其他类型的未知异常
        std::cerr << "发生未知错误。" << std::endl;
        return -1;
    }

    std::cout << "图像处理完成。" << std::endl;
    return 0;
}
