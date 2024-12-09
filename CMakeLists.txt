cmake_minimum_required(VERSION 3.12)
project(RealsenseOpenCVPybind)


find_package(OpenCV REQUIRED)
find_package(pybind11 REQUIRED)

# 创建一个Python扩展模块
# create pybind11 module
pybind11_add_module(PyDepthInpaint src/process_class.cpp)

# 连接依赖的库
# create opencv module
target_link_libraries(PyDepthInpaint PRIVATE 
    ${OpenCV_LIBS} 
)

# 添加头文件目录
# add opencv head file
target_include_directories(PyDepthInpaint PRIVATE 
    ${OpenCV_INCLUDE_DIRS}
)



