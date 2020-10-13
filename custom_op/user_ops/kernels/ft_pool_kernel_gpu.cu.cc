#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include <cstring>

#include "ft_pool_op.h"
#include "tensorflow/core/framework/op.h"
#include <omp.h>
#include <cmath>
#include "tensorflow/core/util/cuda_kernel_helper.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

namespace tensorflow {
namespace {
using CudaLaunchConfig = ::tensorflow::CudaLaunchConfig;

__device__ float absf(float a){return a<0 ? -a : a;}

template <typename T>
__global__ void forward(CudaLaunchConfig cfg,
                        const T* __restrict__ in_tensor,
                        T* __restrict__ out_tensor,
                        const float stride0,
                        const float stride1,
                        const float pool_size0,
                        const float pool_size1,
                        const int num_of_threads,
                        const int Nx,
                        const int H,
                        const int W,
                        const int C,
                        int *bf_values
                        ) {


//  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < cfg.virtual_thread_count; i += blockDim.x* gridDim.x) {
  // naalokuju si sdilenou memory
  // __shared__ int s[64];
  // a pomoci __syncthreads(); syncnu thready tam kde se bude aktualizovat

 // for (int i : CudaGridRangeX(cfg.virtual_thread_count)) {
//    const auto in_tensor = input.tensor<T, 4>();
//    auto out_tensor = output->tensor<T, 4>();
//    out_tensor.setZero();
    // [batch_size, height, width, channels]
//    const int N = input.dim_size(0), H = input.dim_size(1), W = input.dim_size(2), C = input.dim_size(3);

    const int min_H = (int)(H / stride0);
    const int min_W = (int)(W / stride1);

    double bf_sum = 0, comp = 0, power_x  = 0, power_y  = 0, bf_value, hh, ww;
    double width_h  = static_cast<double>(pool_size1) / 2.0;
    double height_h = static_cast<double>(pool_size0) / 2.0;
    double bf_arr_w = ceil(2*width_h), bf_arr_h = ceil(2*height_h);
    //std::array<float, 100> bf_values;


    int py, px;
    //int current_W = i / W;
    //int current_H = i % W;
    //int max_W = current_W + min_W;
    //int max_H = current_H + min_H;

    double w = (double)(blockIdx.x * blockDim.x + threadIdx.x);
    double h = (double)(blockIdx.y * blockDim.y + threadIdx.y);

    //printf("starting W %f, H %f \n", w, h);
    //printf("blockIdx.y=%d blockDim.y=%d  threadIdx.y=%d H=%f \n", blockIdx.y, blockDim.y, threadIdx.y, h);



    //printf("blockIdx %d", blockIdx.x);
    //printf("blockIdx %d", blockIdx.y);

    //printf("blockDim %d", blockDim.x);
    //printf("blockDim %d", blockDim.y);

    //printf("threadIdx %d", threadIdx.x);
    //printf("threadIdx %d", threadIdx.y);


    if (w < W && h < H) {
    //for (double h = blockIdx.x * blockDim.x + threadIdx.x; h < H; h += stride0){
        //if (h/stride[0] >= output->dim_size(1)) continue;
    //    for (double w = blockIdx.y * blockDim.y + threadIdx.y; w < W; w += stride1){
            //if (w/stride[1] >= output->dim_size(2)) continue;
            bf_sum = 0;
            for (double y=ceil(h-height_h); y<=h+height_h; y++){
                py = y<h ? ceil(y) : floor(y);
                if (py>=H || py<0)  continue;
                //power_y = sin( 3.14*static_cast<double>(1.0 - static_cast<double>(absf(py-h) / height_h)) / 2 );
                power_y = 1.0 - static_cast<double>(absf(py-h) / height_h);
                for (double x=ceil(w-width_h); x<=w+width_h; x++){
                    px = x<w ? ceil(x) : floor(x);
                    if (px>=W || px<0) continue;
                    //power_x = sin( 3.14*static_cast<double>(1.0 - static_cast<double>(absf(px-w) / width_h)) / 2 );
                    power_x = 1.0 - static_cast<double>(absf(px-w) / width_h);
                    bf_value = power_x * power_y;
                    bf_sum  += bf_value;
                    int bf_index = static_cast<int>(py - ceil(h-height_h))*bf_arr_w + (px - ceil(w-width_h));
                    bf_values[bf_index] = bf_value;
                    //
                    //bf_values[(py-ceil(h-height_h))*bf_arr_w + (px-ceil(w-width_h))] = bf_value;
                    //std::cout<<py<<"  "<<py-ceil(h-height_h)<<"  "<<px<<" "<<px-ceil(w-width_h)<<"\t"<<h<<"**"<<w<<"\t"<<y<<"/"<<x<<"\t"<<power_y<<" "<<power_x<<"\t"<<bf_value<<"\n";
                }
            }
            //std::cout<<"xxxxxxxxxxxxxxxx\n";
            for (int n = 0; n < Nx; n++){
                for (int c = 0; c < C; c++){
                    for (double y=ceil(h-height_h); y<=h+height_h; y++){
                        py = y<h ? ceil(y) : floor(y);
                        if (py>=H || py<0)  continue;
                        for (double x=ceil(w-width_h); x<=w+width_h; x++){
                            px = x<w ? ceil(x) : floor(x);
                            if (px>=W || px<0) continue;
                            int bf_index = static_cast<int>(py - ceil(h-height_h))*bf_arr_w + (px - ceil(w-width_h));
                            int out_tensor_index = static_cast<int>(n+(((int)round(h/stride0))*(int)round(w/stride0)) + c);
                            int in_tensor_index = static_cast<int>(n + (py * px) + c);
                            __syncthreads();
                            out_tensor[out_tensor_index] += bf_values[bf_index] * in_tensor[in_tensor_index];
                            //out_tensor(n, (int)round(h/stride0), (int)round(w/stride0), c) += bf_values[(py-ceil(h-height_h))*bf_arr_w + (px-ceil(w-width_h))] * in_tensor(n, py, px, c);
                        }
                    }
                    int out_tensor_index = static_cast<int>(n+(((int)round(h/stride0))*(int)round(w/stride1)) + c);
                    __syncthreads();
                    //out_tensor(n, (int)round(h/stride0), (int)round(w/stride1), c) /= bf_sum;
                    out_tensor[out_tensor_index] /= bf_sum;
                }
            }
        //}

        //printf("END W %f, H %f \n", w, h);
    }
  //}
}

template <typename T>
__global__ void backward(CudaLaunchConfig cfg,
                         const T* __restrict__  grad_in,
                         T* __restrict__ grad_out,
                         const float stride0,
                         const float stride1,
                         const float pool_size0,
                         const float pool_size1,
                         const int num_of_threads,
                         const int Nx,
                         const int H,
                         const int W,
                         const int C,
                         int *bf_values
                         ) {


  //for (int i : CudaGridRangeX(cfg.virtual_thread_count)) {

    //auto grad_out_tensor = grad_out->tensor<T, 4>();
    //grad_out_tensor.setZero();
    // sum_tensor is incialized to shape [H,W]
    //Tensor sum(DT_FLOAT, TensorShape({1, grad_out->dim_size(1), grad_out->dim_size(2), 1}));
    //auto sum_tensor = sum.tensor<T, 4>();
    //sum_tensor.setZero();

    //auto grad_in_tensor = grad_in.tensor<T, 4>();

    //const int N = grad_out->dim_size(0), H = grad_out->dim_size(1), W = grad_out->dim_size(2), C = grad_out->dim_size(3);
    double bf_sum = 0, power_x, power_y, bf_value, hh, ww;
    double width_h  = static_cast<double>(pool_size1) / 2.0;
    double height_h = static_cast<double>(pool_size0) / 2.0;
    const int bf_arr_w = ceil(2*width_h), bf_arr_h = ceil(2*height_h);
    //std::array<float, 100> bf_values;


    int py, px;

    double w = (double)(blockIdx.x * blockDim.x + threadIdx.x);
    double h = (double)(blockIdx.y * blockDim.y + threadIdx.y);

    if (w < W && h < H) {
    //for (double h = 0; h < H; h += stride0){
        //if (h/stride[0] >= grad_in.dim_size(1)) continue; //check if round here is necessary
        //for (double w = 0; w < W; w += stride1){
            //if (w/stride[1] >= grad_in.dim_size(2)) continue;
            bf_sum = 0;
            for (double y=ceil(h-height_h); y<=h+height_h; y++){
                py = y<h ? ceil(y) : floor(y);
                if (py>=H || py<0)  continue;
                //power_y = sin( 3.14*static_cast<double>(1.0 - static_cast<double>(absf(py-h) / height_h)) / 2 );
                power_y = 1.0 - static_cast<double>(absf(py-h) / height_h);
                for (double x=ceil(w-width_h); x<=w+width_h; x++){
                    px = x<w ? ceil(x) : floor(x);
                    if (px>=W || px<0) continue;
                    //power_x = sin( 3.14*static_cast<double>(1.0 - static_cast<double>(absf(px-w) / width_h)) / 2 );
                    power_x = 1.0 - static_cast<double>(absf(px-w) / width_h);
                    bf_value = power_x * power_y;
                    bf_sum  += bf_value;
                    int bf_index = static_cast<int>(py-ceil(h-height_h))*bf_arr_w + (px-ceil(w-width_h));
                    bf_values[bf_index] = bf_value;
                }
            }
            for (int n = 0; n < Nx; n++){
                for (int c = 0; c < C; c++){
                    for (double y=ceil(h-height_h); y<=h+height_h; y++){
                        py = y<h ? ceil(y) : floor(y);
                        if (py>=H || py<0)  continue;
                        for (double x=ceil(w-width_h); x<=w+width_h; x++){
                            px = x<w ? ceil(x) : floor(x);
                            if (px>=W || px<0) continue;
                            int bf_index = static_cast<int>(py-ceil(h-height_h))*bf_arr_w + (px-ceil(w-width_h));
                            int grad_out_index = static_cast<int>(n);
                            int grad_in_index = static_cast<int>(n);
                            grad_out[grad_out_index] += bf_values[bf_index] * grad_in[grad_in_index];
                            //grad_out_tensor(n, py, px, c) += bf_values[(py-ceil(h-height_h))*bf_arr_w + (px-ceil(w-width_h))] * grad_in_tensor(n, (int)round(h/stride0), (int)round(w/stride1), c);
                        }
                    }
                }
            }
        //}
    //}
    }
  //}
}
}//anonymous namespace

namespace functor {
float absf(float a){return a<0 ? -a : a;}

template <typename Dtype>
struct FtPoolFunctor<GPUDevice, Dtype> {
  static void launch(::tensorflow::OpKernelContext* ctx,
                     const Tensor& input,
                     Tensor* output,
                     std::vector<float> stride,
                     std::vector<float> pool_size) {

    // spocitat H / stride[0] pocet ft poolingovych oken na obrazku na vysku
    // W / stride[1] pocet ft poolingovych oken na obrazku na sirku

    // podle toho zjistit kolik potrebuju oken = kolik potrebuju threadu
    const int Nx = input.dim_size(0), H = input.dim_size(1), W = input.dim_size(2), C = input.dim_size(3);
    const float stride0 = stride[0];
    const float stride1 = stride[1];
    const float pool_size0 = pool_size[0];
    const float pool_size1 = pool_size[1];

    int pool_w, pool_h;
    pool_h = (int)(H / stride[0]);
    pool_w = (int)(W / stride[1]);

    //const int N = input.NumElements(); // (H/stride) * (W/stride)
    const int N = (int)(pool_h / pool_w);
    const GPUDevice& d = ctx->eigen_gpu_device();

    int *d_a;
    cudaMalloc((void **)&d_a, 100 * N);

    ::tensorflow::CudaLaunchConfig cfg =
        ::tensorflow::GetCudaLaunchConfig(8, d);

    //cuda_kernelC<<<grid_size,block_size,0,stream>>>(data2);
    //dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    //dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
    //forward<Dtype><<<(int)(pool_h / pool_w), 1, 0, d.stream()>>>(
    dim3 dimBlock(10, 10);
    dim3 dimGrid((int)(pool_w / dimBlock.x), (int)(pool_h / dimBlock.y));

    //printf("END pool_h %d, pool_w %d \n", pool_h, pool_w);
    //printf("END dimGrid.x %d, dimGrid.y %d , dimGrid.z %d\n", dimGrid.x, dimGrid.y, dimGrid.z);


    forward<Dtype><<<dimGrid, dimBlock, 0, d.stream()>>>(
        cfg,
        input.flat<Dtype>().data(),
        output->flat<Dtype>().data(),
        stride0,
        stride1,
        pool_size0,
        pool_size1,
        N,
        Nx,
        H,
        W,
        C,
        d_a
        );

    //cudaDeviceSynchronize();
    if (!d.ok()) {
      ctx->SetStatus(
          tensorflow::errors::Internal("Failed launching FtPool on GPU"));
    }
  }
};

template struct FtPoolFunctor<GPUDevice, int32>;
template struct FtPoolFunctor<GPUDevice, uint32>;
template struct FtPoolFunctor<GPUDevice, float>;
template struct FtPoolFunctor<GPUDevice, double>;

template <typename Dtype>
struct FtPoolGrad<GPUDevice, Dtype> {
  static void launch(::tensorflow::OpKernelContext* ctx,
                     const Tensor& grad_in,
                     Tensor* grad_out,
                     std::vector<float> stride,
                     std::vector<float> pool_size) {

    auto grad_out_tensor = grad_out->tensor<Dtype, 4>();
    grad_out_tensor.setZero();
    // sum_tensor is incialized to shape [H,W]
    Tensor sum(DT_FLOAT, TensorShape({1, grad_out->dim_size(1), grad_out->dim_size(2), 1}));
    auto sum_tensor = sum.tensor<Dtype, 4>();
    sum_tensor.setZero();

    auto grad_in_tensor = grad_in.tensor<Dtype, 4>();

    const int Nx = grad_out->dim_size(0), H = grad_out->dim_size(1), W = grad_out->dim_size(2), C = grad_out->dim_size(3);

    const float stride0 = stride[0];
    const float stride1 = stride[1];
    const float pool_size0 = pool_size[0];
    const float pool_size1 = pool_size[1];

    int *d_a;
    cudaMalloc((void **)&d_a, 100 * Nx);

    int pool_w, pool_h;
    pool_h = (int)(H / stride[0]);
    pool_w = (int)(W / stride[1]);

    const int N = (int)(pool_h / pool_w);
    const GPUDevice& d = ctx->eigen_gpu_device();

    ::tensorflow::CudaLaunchConfig cfg =
        ::tensorflow::GetCudaLaunchConfig(8, d);

    dim3 dimBlock(5, 5);
    dim3 dimGrid((int)(pool_w / dimBlock.x), (int)(pool_h / dimBlock.y));

    backward<Dtype><<<dimGrid, dimBlock, 0, ctx->eigen_gpu_device().stream() >>> (
      cfg,
      grad_in.flat<Dtype>().data(),
      grad_out->flat<Dtype>().data(),
      stride0,
      stride1,
      pool_size0,
      pool_size1,
      N,
      Nx,
      H,
      W,
      C,
      d_a
      );

    //cudaDeviceSynchronize();
    if (!d.ok()) {
      ctx->SetStatus(tensorflow::errors::Internal(
          "Failed launching FtPoolGrad on GPU"));
    }

  }
};

template struct FtPoolGrad<GPUDevice, float>;
template struct FtPoolGrad<GPUDevice, double>;

}  // namespace functor
}  // namespace tensorflow

#endif  // GOOGLE_CUDA