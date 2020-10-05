// 2018, Patrick Wieschollek <mail@patwie.com>
/*
#if 1 //GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "ft_pool_op.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

namespace tensorflow {
namespace {

using CudaLaunchConfig = ::tensorflow::CudaLaunchConfig;

template <typename T>
__global__ void forward(const Tensor& input, Tensor* output) {
  // for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x
  // * gridDim.x) {
  //for (int i : CudaGridRangeX(cfg.virtual_thread_count)) {
  //  Z[i] = X[i] + Y[i] + (T)bias;
  const auto in_tensor = input.tensor<T, 4>();
  auto out_tensor = output->tensor<T, 4>();
  
  }
}

template <typename T>
__global__ void backward(CudaLaunchConfig cfg, const T* __restrict__ top_diff,
                         const int N, T* __restrict__ grad_matrixA,
                         T* __restrict__ grad_matrixB) {
  // for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x
  // * gridDim.x) {
  for (int i : CudaGridRangeX(cfg.virtual_thread_count)) {
    grad_matrixA[i] = top_diff[i];
    grad_matrixB[i] = top_diff[i];
  }
}

}  // anonymous namespace

namespace functor {

template <typename Dtype>
struct MatrixAddFunctor<GPUDevice, Dtype> {
  static void launch(::tensorflow::OpKernelContext* context, const Tensor& input,
                     Tensor* output, Dtype bias) {
    //const int N = mA_.NumElements();
    const int N = input.dim_size(0);
    const int H = input.dim_size(1);
    const int W = input.dim_size(2);
    const GPUDevice& d = context->eigen_gpu_device();

    ::tensorflow::Cuda3DLaunchConfig cfg =
        ::tensorflow::GetCuda3DLaunchConfig(W, H, N, d);

    forward<Dtype><<<cfg.block_count, cfg.thread_per_block, 0, d.stream()>>>(input, output);

    if (!d.ok()) {
      context->SetStatus(
          tensorflow::errors::Internal("Failed launching MatrixAdd on GPU"));
    }
  }
};

template struct MatrixAddFunctor<GPUDevice, int>;
template struct MatrixAddFunctor<GPUDevice, float>;
template struct MatrixAddFunctor<GPUDevice, double>;

template <typename Dtype>
struct MatrixAddGrad<GPUDevice, Dtype> {
  static void launch(::tensorflow::OpKernelContext* context,
                     const Tensor& topdiff_, Tensor* grad_mA_,
                     Tensor* grad_mB_) {
    const int N = topdiff_.NumElements();
    const GPUDevice& d = context->eigen_gpu_device();

    ::tensorflow::CudaLaunchConfig cfg =
        ::tensorflow::GetCudaLaunchConfig(N, d);

    // // optional reset gradients before running a kernel
    // cudaMemset(grad_mA_->flat<Dtype>().data(), 0, N * sizeof(Dtype));
    // cudaMemset(grad_mB_->flat<Dtype>().data(), 0, N * sizeof(Dtype));

     backward<Dtype>
     <<< cfg.block_count, cfg.thread_per_block, 0,
     context->eigen_gpu_device().stream() >>> (
       cfg,
       topdiff_.flat<Dtype>().data(),
       topdiff_.NumElements(),
       grad_mA_->flat<Dtype>().data(),
       grad_mB_->flat<Dtype>().data());

    // faster alternative to custom kernel (above)
    //cudaMemcpy(grad_mA_->flat<Dtype>().data(), topdiff_.flat<Dtype>().data(),
    //           N * sizeof(Dtype), cudaMemcpyDeviceToDevice);
    //cudaMemcpy(grad_mB_->flat<Dtype>().data(), topdiff_.flat<Dtype>().data(),
    //           N * sizeof(Dtype), cudaMemcpyDeviceToDevice);

    if (!d.ok()) {
      context->SetStatus(tensorflow::errors::Internal(
          "Failed launching MatrixAddGrad on GPU"));
    }
  }
};

template struct MatrixAddGrad<GPUDevice, float>;
template struct MatrixAddGrad<GPUDevice, double>;

}  // namespace functor
}  // namespace tensorflow

#endif  // GOOGLE_CUDA

*/