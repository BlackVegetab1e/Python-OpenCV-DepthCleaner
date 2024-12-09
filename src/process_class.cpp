// Libraries
#include <opencv2/opencv.hpp>   // Include OpenCV API
#include <opencv2/rgbd.hpp>     // OpenCV RGBD Contrib package
#include <opencv2/highgui/highgui_c.h> // OpenCV High-level GUI

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>


// STD
#include <iostream>



namespace py = pybind11;

class DepthProcess {
public:
    const int w;
    const int h;

    std::unique_ptr<cv::rgbd::DepthCleaner> depthc;

    DepthProcess(int pic_w, int pic_h)
        : w(pic_w), h(pic_h),
          depthc(std::make_unique<cv::rgbd::DepthCleaner>(CV_16U, 7, cv::rgbd::DepthCleaner::DEPTH_CLEANER_NIL)) {
        std::cout << "DepthProcess construction complete, Depth " << w << "x" << h << std::endl;
    }

    py::array_t<uint16_t> process(py::array_t<uint16_t> rs_depth);

    cv::Mat numpy_to_mat(py::array_t<uint16_t>& input);

    py::array_t<uint16_t> mat_to_numpy(const cv::Mat& mat);
};

// 将 numpy 数组转换为 OpenCV Mat
cv::Mat DepthProcess::numpy_to_mat(py::array_t<uint16_t>& input) {
    py::buffer_info buf = input.request();
    int rows = buf.shape[0];
    int cols = buf.shape[1];

    cv::Mat mat(rows, cols, CV_16U, static_cast<uint16_t*>(buf.ptr));
    return mat.clone();  // 克隆以防止数据共享问题
}

// 将 OpenCV Mat 转换为 numpy 数组
py::array_t<uint16_t> DepthProcess::mat_to_numpy(const cv::Mat& mat) {
    py::buffer_info buf(
        mat.data,
        sizeof(uint16_t),
        py::format_descriptor<uint16_t>::format(),
        2,
        { mat.rows, mat.cols },
        { static_cast<size_t>(mat.step[0]), static_cast<size_t>(mat.elemSize()) }
    );

    return py::array_t<uint16_t>(buf);
}

// 实现 process 函数
py::array_t<uint16_t> DepthProcess::process(py::array_t<uint16_t> rs_depth) {
    // 将 numpy 数组转换为 OpenCV Mat
    cv::Mat rawDepthMat = numpy_to_mat(rs_depth);

    // 创建用于存储输出的 OpenCV Mat
    cv::Mat cleanedDepth(cv::Size(w, h), CV_16U);

    // 使用 DepthCleaner 清理深度图像
    (*depthc)(rawDepthMat, cleanedDepth);

    const unsigned char noDepth = 0;

    cv::Mat temp, temp2;

    // 对未知像素进行修复
    cv::inpaint(cleanedDepth, (cleanedDepth == noDepth), temp, 5.0, cv::INPAINT_TELEA);
    // Upscale to original size and replace inpainted regions in original depth image
    resize(temp, temp2, cleanedDepth.size());
    temp2.copyTo(cleanedDepth, (cleanedDepth == noDepth));  // add to the original signal

    // 返回结果到 numpy 数组
    return mat_to_numpy(cleanedDepth);
}




PYBIND11_MODULE(PyDepthInpaint, m) {
    py::class_<DepthProcess>(m, "DepthProcess")
        // 绑定构造函数
        .def(py::init<int, int>(), py::arg("pic_w"), py::arg("pic_h"))

        // 绑定 process 函数
        .def("process", &DepthProcess::process, py::arg("rs_depth"),
             "Process the depth image to clean and inpaint unknown values.");
}