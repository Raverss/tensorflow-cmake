#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include <cstring>
#include "ft_inverse_op.h"
#include "tensorflow/core/framework/op.h"
#include <omp.h>
#include <cmath>
#include <math.h>
#include "tensorflow/core/util/cuda_kernel_helper.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

namespace tensorflow {
namespace {
using CudaLaunchConfig = ::tensorflow::CudaLaunchConfig;

__device__ float absf(float a){return a<0 ? -a : a;}

template <typename T>
__global__ void forward(CudaLaunchConfig cfg,
                         const T* __restrict__  grad_in,
                         T* __restrict__ grad_out,
                         const float stride0,
                         const float stride1,
                         const float pool_size0,
                         const float pool_size1,
                         const int Nx,
                         const int H,
                         const int W,
                         const int C,
                         const int mem_init
                        ) {
    //printf("--- Starting forward \n");
    double w = (double)(blockIdx.x * blockDim.x + threadIdx.x) * stride1;
    double h = (double)(blockIdx.y * blockDim.y + threadIdx.y) * stride0;
    printf("Launch forward inverse");

    if (w < W && h < H) {
        double bf_sum = 0, power_x, power_y, bf_value;
        double width_h  = static_cast<double>(pool_size1) / 2.0;
        double height_h = static_cast<double>(pool_size0) / 2.0;
        const int bf_arr_w = ceil(2*width_h), bf_arr_h = ceil(2*height_h);
        float* bf_values = new float[mem_init];
        int py, px;
        bf_sum = 0;
        for (double y=ceil(h-height_h); y<=h+height_h; y++){
            py = y<h ? ceil(y) : floor(y);
            if (py>=H || py<0)  continue;
            power_y = sin( 3.14*static_cast<double>(1.0 - static_cast<double>(absf(py-h) / height_h)) / 2 );
            for (double x=ceil(w-width_h); x<=w+width_h; x++){
                px = x<w ? ceil(x) : floor(x);
                if (px>=W || px<0) continue;
                power_x = sin( 3.14*static_cast<double>(1.0 - static_cast<double>(absf(px-w) / width_h)) / 2 );
                bf_value = power_x * power_y;
                bf_sum  += bf_value;
                int bf_index = static_cast<int>(py-ceil(h-height_h))*bf_arr_w + (px-ceil(w-width_h));
                bf_values[bf_index] = bf_value;
            }
        }
        for (int n = 0; n < Nx; n++){
            for (int c = 0; c < C; c++){
                int grad_in_index = static_cast<int>(n * (int)round(H/stride0) * (int)round(W/stride1) * C + (int)round(h/stride0) * (int)round(W/stride1) * C + (int)round(w/stride1) * C + c);
                for (double y=ceil(h-height_h); y<=h+height_h; y++){
                    py = y<h ? ceil(y) : floor(y);
                    if (py>=H || py<0)  continue;
                    for (double x=ceil(w-width_h); x<=w+width_h; x++){
                        px = x<w ? ceil(x) : floor(x);
                        if (px>=W || px<0) continue;
                        int bf_index = static_cast<int>(py-ceil(h-height_h))*bf_arr_w + (px-ceil(w-width_h));
                        int grad_out_index = static_cast<int>(n * H * W * C + py * W * C + px * C +c );
                        grad_out[grad_out_index] += bf_values[bf_index] * grad_in[grad_in_index];
                    }
                }
            }
        }
    }
}

template <typename T>
__global__ void backward(CudaLaunchConfig cfg,
                        const T* __restrict__ in_tensor,
                        T* __restrict__ out_tensor,
                        const float stride0,
                        const float stride1,
                        const float pool_size0,
                        const float pool_size1,
                        const int Nx,
                        const int H,
                        const int W,
                        const int C,
                        const int mem_init
                         ) {
    printf("Launch backward inverse");
    double w = (double)(blockIdx.x * blockDim.x + threadIdx.x) * stride1;
    double h = (double)(blockIdx.y * blockDim.y + threadIdx.y) * stride0;

    if (w < W && h < H) {
        double bf_sum = 0, power_x = 0, power_y = 0, bf_value;
        double width_h  = static_cast<double>(pool_size1) / 2.0;
        double height_h = static_cast<double>(pool_size0) / 2.0;
        double bf_arr_w = ceil(2*width_h), bf_arr_h = ceil(2*height_h);
        float* bf_values = new float[mem_init];
        int py, px;
        bf_sum = 0;
        for (double y=ceil(h-height_h); y<=h+height_h; y++){
            py = y<h ? ceil(y) : floor(y);
            if (py>=H || py<0)  continue;
            power_y = sin( 3.14*static_cast<double>(1.0 - static_cast<double>(absf(py-h) / height_h)) / 2 );
            for (double x=ceil(w-width_h); x<=w+width_h; x++){
                px = x<w ? ceil(x) : floor(x);
                if (px>=W || px<0) continue;
                power_x = sin( 3.14*static_cast<double>(1.0 - static_cast<double>(absf(px-w) / width_h)) / 2 );
                bf_value = power_x * power_y;
                bf_sum  += bf_value;
                int bf_index = static_cast<int>(py - ceil(h-height_h))*bf_arr_w + (px - ceil(w-width_h));
                bf_values[bf_index] = bf_value;
            }
        }
        for (int n = 0; n < Nx; n++){
            for (int c = 0; c < C; c++){
                int out_tensor_index = static_cast<int>(n * (int)round(H/stride0) * (int)round(W/stride1) * C + (int)round(h/stride0) * (int)round(W/stride1) * C + (int)round(w/stride1) * C + c);
                for (double y=ceil(h-height_h); y<=h+height_h; y++){
                    py = y<h ? ceil(y) : floor(y);
                    if (py>=H || py<0)  continue;
                    for (double x=ceil(w-width_h); x<=w+width_h; x++){
                        px = x<w ? ceil(x) : floor(x);
                        if (px>=W || px<0) continue;
                        int bf_index = static_cast<int>(py - ceil(h-height_h))*bf_arr_w + (px - ceil(w-width_h));
                        int in_tensor_index = static_cast<int>(n * H * W * C + py * W * C + px * C +c );
                        out_tensor[out_tensor_index] += in_tensor[in_tensor_index] * bf_values[bf_index];
                    }
                }
                out_tensor[out_tensor_index] /= bf_sum;
            }
        }
        delete[] bf_values;
    }
}
}//anonymous namespace

namespace functor {
float absf(float a){return a<0 ? -a : a;}

template <typename Dtype>
struct FtInverseFunctor<GPUDevice, Dtype> {
  static void launch(::tensorflow::OpKernelContext* ctx,
                     const Tensor& input,
                     Tensor* output,
                     std::vector<float> stride,
                     std::vector<float> pool_size) {
    const int Nx = input.dim_size(0), H = input.dim_size(1), W = input.dim_size(2), C = input.dim_size(3);
    const int Nxoutput = output->dim_size(0), Houtput = output->dim_size(1), Woutput = output->dim_size(2), Coutput = output->dim_size(3);

    const float stride0 = stride[0];
    const float stride1 = stride[1];
    const float pool_size0 = pool_size[0];
    const float pool_size1 = pool_size[1];

    float pool_w, pool_h;
    pool_h = (H / stride[0]);
    pool_w = (W / stride[1]);

    const GPUDevice& d = ctx->eigen_gpu_device();
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);

    ::tensorflow::CudaLaunchConfig cfg =
        ::tensorflow::GetCudaLaunchConfig(8, d);

    int const BC_S = 6;
    int block = floor(sqrt(props.maxThreadsPerBlock)) - BC_S;
    int mem_init = floor(props.regsPerBlock / pow(block, 2));

    dim3 dimBlock(block, block);
    dim3 dimGrid((int)ceil(pool_w / dimBlock.x), (int)ceil(pool_h / dimBlock.y));

    forward<Dtype><<<dimGrid, dimBlock, 0, d.stream()>>>(
        cfg,
        input.flat<Dtype>().data(),
        output->flat<Dtype>().data(),
        stride0,
        stride1,
        pool_size0,
        pool_size1,
        Nx,
        H,
        W,
        C,
        mem_init
        );

    if (!d.ok()) {
      ctx->SetStatus(
          tensorflow::errors::Internal("Failed launching FtInversePool on GPU"));
    }
  }
};

template struct FtInverseFunctor<GPUDevice, float>;
template struct FtInverseFunctor<GPUDevice, double>;

template <typename Dtype>
struct FtInverseGrad<GPUDevice, Dtype> {
  static void launch(::tensorflow::OpKernelContext* ctx,
                     const Tensor& grad_in,
                     Tensor* grad_out,
                     std::vector<float> stride,
                     std::vector<float> pool_size) {
    const int Nx = grad_out->dim_size(0), H = grad_out->dim_size(1), W = grad_out->dim_size(2), C = grad_out->dim_size(3);

    const float stride0 = stride[0];
    const float stride1 = stride[1];
    const float pool_size0 = pool_size[0];
    const float pool_size1 = pool_size[1];

    int pool_w, pool_h;
    pool_h = (int)(H / stride[0]);
    pool_w = (int)(W / stride[1]);

    // sum_tensor is incialized to shape [H,W]
    Tensor sum(DT_FLOAT, TensorShape({1, grad_out->dim_size(1), grad_out->dim_size(2), 1}));
    auto sum_tensor = sum.tensor<Dtype, 4>();
    sum_tensor.setZero();
    auto grad_in_tensor = grad_in.tensor<Dtype, 4>();

    const GPUDevice& d = ctx->eigen_gpu_device();

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);

    ::tensorflow::CudaLaunchConfig cfg =
        ::tensorflow::GetCudaLaunchConfig(8, d);

    int const BC_S = 6;
    int block = floor(sqrt(props.maxThreadsPerBlock)) - BC_S;
    int mem_init = floor(props.regsPerBlock / pow(block, 2));

    dim3 dimBlock(block, block);
    dim3 dimGrid((int)ceil(pool_w / dimBlock.x), (int)ceil(pool_h / dimBlock.y));

    backward<Dtype><<<dimGrid, dimBlock, 0, ctx->eigen_gpu_device().stream() >>> (
      cfg,
      grad_in.flat<Dtype>().data(),
      grad_out->flat<Dtype>().data(),
      stride0,
      stride1,
      pool_size0,
      pool_size1,
      Nx,
      H,
      W,
      C,
      mem_init
      );

    if (!d.ok()) {
      ctx->SetStatus(tensorflow::errors::Internal(
          "Failed launching FtInverseGrad on GPU"));
    }

  }
};

template struct FtInverseGrad<GPUDevice, float>;
template struct FtInverseGrad<GPUDevice, double>;

}  // namespace functor
}  // namespace tensorflow

#endif  // GOOGLE_CUDA