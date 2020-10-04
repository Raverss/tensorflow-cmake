#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include <cstring>

#include "ft_pool_op.h"
#include "tensorflow/core/framework/op.h"
#include <omp.h>
#include <cmath>
#include "tensorflow/core/util/cuda_kernel_helper.h"

namespace tensorflow {
namespace {
using CudaLaunchConfig = ::tensorflow::CudaLaunchConfig;

float absf(float a){return a<0 ? -a : a;}

template <typename T>
__global__ void forward(CudaLaunchConfig cfg,
                        const Tensor& input,
                        Tensor* output,
                        //T*  out_tensor,
                        //const T&  in_tensor,
                        std::vector<float> stride,
                        std::vector<float> pool_size) {

  const auto in_tensor = input.tensor<T, 4>();
  auto out_tensor = output->tensor<T, 4>();
  out_tensor.setZero();

  for (int i : CudaGridRangeX(cfg.virtual_thread_count)) {
    // [batch_size, height, width, channels]
    const int N = input.dim_size(0), H = input.dim_size(1), W = input.dim_size(2), C = input.dim_size(3);

    double bf_sum = 0, comp = 0, power_x  = 0, power_y  = 0, bf_value, hh, ww;
    double width_h  = static_cast<double>(pool_size[1]) / 2.0;
    double height_h = static_cast<double>(pool_size[0]) / 2.0;
    double bf_arr_w = ceil(2*width_h), bf_arr_h = ceil(2*height_h);
    std::array<float, 100> bf_values;

    int py, px;

    for (double h = 0; h < H; h += stride[0]){
        if (h/stride[0] >= output->dim_size(1)) continue;
        for (double w = 0; w < W; w += stride[1]){
            if (w/stride[1] >= output->dim_size(2)) continue;
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
                    bf_values[(py-ceil(h-height_h))*bf_arr_w + (px-ceil(w-width_h))] = bf_value;
                    //std::cout<<py<<"  "<<py-ceil(h-height_h)<<"  "<<px<<" "<<px-ceil(w-width_h)<<"\t"<<h<<"**"<<w<<"\t"<<y<<"/"<<x<<"\t"<<power_y<<" "<<power_x<<"\t"<<bf_value<<"\n";
                }
            }
            //std::cout<<"xxxxxxxxxxxxxxxx\n";
            for (int n = 0; n < N; n++){
                for (int c = 0; c < C; c++){
                    for (double y=ceil(h-height_h); y<=h+height_h; y++){
                        py = y<h ? ceil(y) : floor(y);
                        if (py>=H || py<0)  continue;
                        for (double x=ceil(w-width_h); x<=w+width_h; x++){
                            px = x<w ? ceil(x) : floor(x);
                            if (px>=W || px<0) continue;
                            out_tensor(n, round(h/stride[0]), round(w/stride[1]), c) += bf_values[(py-ceil(h-height_h))*bf_arr_w + (px-ceil(w-width_h))] * in_tensor(n, py, px, c);
                        }
                    }
                    out_tensor(n, round(h/stride[0]), round(w/stride[1]), c) /= bf_sum;
                }
            }
        }
    }
  }
}

template <typename T>
__global__ void backward(CudaLaunchConfig cfg,
                         const Tensor&  grad_in,
                         Tensor* grad_out,
                         std::vector<float> stride,
                         std::vector<float> pool_size) {

  for (int i : CudaGridRangeX(cfg.virtual_thread_count)) {
    auto grad_out_tensor = grad_out->tensor<T, 4>();
    grad_out_tensor.setZero();
    // sum_tensor is incialized to shape [H,W]
    Tensor sum(DT_FLOAT, TensorShape({1, grad_out->dim_size(1), grad_out->dim_size(2), 1}));
    auto sum_tensor = sum.tensor<T, 4>();
    sum_tensor.setZero();

    auto grad_in_tensor = grad_in.tensor<T, 4>();

    const int N = grad_out->dim_size(0), H = grad_out->dim_size(1), W = grad_out->dim_size(2), C = grad_out->dim_size(3);
    double bf_sum = 0, power_x, power_y, bf_value, hh, ww;
    double width_h  = static_cast<double>(pool_size[1]) / 2.0;
    double height_h = static_cast<double>(pool_size[0]) / 2.0;
    const int bf_arr_w = ceil(2*width_h), bf_arr_h = ceil(2*height_h);
    std::array<float, 100> bf_values;


    int py, px;

    for (double h = 0; h < H; h += stride[0]){
        if (h/stride[0] >= grad_in.dim_size(1)) continue; //check if round here is necessary
        for (double w = 0; w < W; w += stride[1]){
            if (w/stride[1] >= grad_in.dim_size(2)) continue;
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
                    bf_values[(py-ceil(h-height_h))*bf_arr_w + (px-ceil(w-width_h))] = bf_value;
                }
            }
            for (int n = 0; n < N; n++){
                for (int c = 0; c < C; c++){
                    for (double y=ceil(h-height_h); y<=h+height_h; y++){
                        py = y<h ? ceil(y) : floor(y);
                        if (py>=H || py<0)  continue;
                        for (double x=ceil(w-width_h); x<=w+width_h; x++){
                            px = x<w ? ceil(x) : floor(x);
                            if (px>=W || px<0) continue;
                            grad_out_tensor(n, py, px, c) += bf_values[(py-ceil(h-height_h))*bf_arr_w + (px-ceil(w-width_h))] * grad_in_tensor(n, round(h/stride[0]), round(w/stride[1]), c);
                        }
                    }
                }
            }
        }
    }
  }
}
}//anonymous namespace

namespace functor {
float absf(float a){return a<0 ? -a : a;}

template <typename Dtype>
struct FtPoolFunctor<GPUDevice, Dtype> {
  static void launch(::tensorflow::OpKernelContext* ctx, const Tensor& input,
                     Tensor* output, std::vector<float> stride, std::vector<float> pool_size) {
    const int N = input.NumElements();
    const GPUDevice& d = ctx->eigen_gpu_device();

    ::tensorflow::CudaLaunchConfig cfg =
        ::tensorflow::GetCudaLaunchConfig(N, d);


    forward<Dtype><<<cfg.block_count, cfg.thread_per_block, 0, d.stream()>>>(
        cfg,
        input,
        output,
        stride,
        pool_size);

    if (!d.ok()) {
      ctx->SetStatus(
          tensorflow::errors::Internal("Failed launching FtPool on GPU"));
    }
  }
};

//template struct FtPoolFunctor<GPUDevice, int32>;
//template struct FtPoolFunctor<GPUDevice, uint32>;
//template struct FtPoolFunctor<GPUDevice, float>;
//template struct FtPoolFunctor<GPUDevice, double>;

template <typename Dtype>
struct FtPoolGrad<GPUDevice, Dtype> {
  static void launch(::tensorflow::OpKernelContext* ctx,
                     const Tensor& grad_in,
                     Tensor* grad_out,
                     std::vector<float> stride,
                     std::vector<float> pool_size) {
    const int N = grad_in.NumElements();
    const GPUDevice& d = ctx->eigen_gpu_device();

    ::tensorflow::CudaLaunchConfig cfg =
        ::tensorflow::GetCudaLaunchConfig(N, d);

    backward<Dtype><<< cfg.block_count, cfg.thread_per_block, 0, ctx->eigen_gpu_device().stream() >>> (
      cfg,
      grad_in,
      grad_out,
      stride,
      pool_size);

    if (!d.ok()) {
      ctx->SetStatus(tensorflow::errors::Internal(
          "Failed launching FtPoolGrad on GPU"));
    }
  }
};

//template struct FtPoolGrad<GPUDevice, float>;
//template struct FtPoolGrad<GPUDevice, double>;

}  // namespace functor
}  // namespace tensorflow

#endif  // GOOGLE_CUDA