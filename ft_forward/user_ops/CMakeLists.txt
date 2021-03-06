# University Tuebingen, 2018
cmake_minimum_required( VERSION 2.8 )

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}../../cmake/modules)

project(TFCustomOp)

find_package(TensorFlow REQUIRED)
find_package(CUDA 9 REQUIRED)

# set necessary flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SSE_FLAGS} -march=native -fopenmp -O3 -D_GLIBCXX_USE_CXX11_ABI=${TensorFlow_ABI}")
set(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} -fPIC --shared -D_GLIBCXX_USE_CXX11_ABI=${TensorFlow_ABI}" )
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11  --expt-relaxed-constexpr -D GOOGLE_CUDA=1 --gpu-architecture=compute_60 -D_GLIBCXX_USE_CXX11_ABI=${TensorFlow_ABI}" )

include_directories(SYSTEM "${CUDA_INCLUDE_DIRS}/../../")
include_directories(SYSTEM ${TensorFlow_INCLUDE_DIR})
include_directories(SYSTEM "kernels")

# quick fix for drone-ci
include_directories(SYSTEM "/usr/local/")


add_tensorflow_gpu_operation("ft_pool")
