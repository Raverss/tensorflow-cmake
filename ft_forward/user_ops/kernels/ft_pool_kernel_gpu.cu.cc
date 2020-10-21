//#if GOOGLE_CUDA

#define EIGEN_USE_GPU
#define DEBUG_PRINT false

#include <cstring>
#include "ft_pool_op.h"
#include "tensorflow/core/framework/op.h"
#include <omp.h>
#include <cmath>
#include <math.h>
#include "tensorflow/core/util/cuda_kernel_helper.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
//for prining purposes
#include <iostream>
#include <fstream>
#include <iomanip>

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
                        const int Nx,
                        const int H,
                        const int W,
                        const int C,
                        const int mem_init
                        ) {
    double w = (double)(blockIdx.x * blockDim.x + threadIdx.x) * stride1;
    double h = (double)(blockIdx.y * blockDim.y + threadIdx.y) * stride0;
    int ww = 2, hh = 2;
    if (w < W && h < H) {
        double bf_sum = 0, power_x = 0, power_y = 0, bf_value;
        double width_h  = static_cast<double>(pool_size1) / 2.0;
        double height_h = static_cast<double>(pool_size0) / 2.0;
        double bf_arr_w = ceil(2*width_h), bf_arr_h = ceil(2*height_h);
        float bf_values[25]; //mem_init
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
                out_tensor[out_tensor_index] = 0;
                //printf("w: %f, h: %f, out_tensor_index: %d, H/stride0: %d, W/stride1 %d \n", w, h, out_tensor_index, (int)round(h/stride0), (int)round(w/stride1));
                for (double y=ceil(h-height_h); y<=h+height_h; y++){
                    py = y<h ? ceil(y) : floor(y);
                    if (py>=H || py<0)  continue;
                    for (double x=ceil(w-width_h); x<=w+width_h; x++){
                        px = x<w ? ceil(x) : floor(x);
                        if (px>=W || px<0) continue;
                        int bf_index = static_cast<int>(py - ceil(h-height_h))*bf_arr_w + (px - ceil(w-width_h));
                        int in_tensor_index = static_cast<int>(n * H * W * C + py * W * C + px * C +c );
                        //if (w == 4.5 && h == 4.5) printf("py: %d px: %d in_tensor[in_tensor_index]: %f bf_values[bf_index]: %f , in_tensor_index: %d \n", py, px, in_tensor[in_tensor_index], bf_values[bf_index], in_tensor_index);
                        if(w/stride1 == ww && h/stride0 == hh && DEBUG_PRINT) printf("[y:%d,x:%d] %3.3f * %3.3f\t", py, px, bf_values[bf_index], in_tensor[in_tensor_index]);
                        out_tensor[out_tensor_index] += in_tensor[in_tensor_index] * bf_values[bf_index];
                    }
                    //if((int)round(w/stride1) == ww && (int)round(h/stride0) == hh && DEBUG_PRINT) printf("\n");
                }
                //if(isnan(out_tensor[out_tensor_index])) printf("nan pred!\n");
                out_tensor[out_tensor_index] /= bf_sum;
                //if(isnan(out_tensor[out_tensor_index])) printf("nan po!\n");
                //if((int)round(w/stride1) == ww && (int)round(h/stride0) == hh && DEBUG_PRINT) printf("comp pos in img:[%3.2f,%3.2f]: %3.3f\n",h,w, out_tensor[out_tensor_index]);
            }
        }
        delete[] bf_values;
    }
}

template <typename T>
__global__ void clear(CudaLaunchConfig cfg, T* __restrict__ grad_out, const int pom) {
      int i = (blockIdx.x * blockDim.x + threadIdx.x);
      if (i < pom) {
        grad_out[i] = 0;
      }
}


template <typename T>
__global__ void backward(CudaLaunchConfig cfg,
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
    int ww = 2, hh = 2;
    double w = (double)(blockIdx.x * blockDim.x + threadIdx.x) * stride1;
    double h = (double)(blockIdx.y * blockDim.y + threadIdx.y) * stride0;
    int pocitadlo = 0;
    //printf("w=%f h=%f \n",w,h);

    if (w < W && h < H) {
        if((int)round(w/stride1) == ww && (int)round(h/stride0) == hh && DEBUG_PRINT) printf("grad pos in img:[%3.2f,%3.2f]\n",h,w);
        double bf_sum = 0, power_x, power_y, bf_value;
        double width_h  = static_cast<double>(pool_size1) / 2.0;
        double height_h = static_cast<double>(pool_size0) / 2.0;
        const int bf_arr_w = ceil(2*width_h), bf_arr_h = ceil(2*height_h);
        double bf_values[25]; //mem_init
        int py, px, bf_index, grad_out_index;
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
        /* deleni sumou */
        for (double y=ceil(h-height_h); y<=h+height_h; y++){
            py = y<h ? ceil(y) : floor(y);
            if (py>=H || py<0)  continue;
            for (double x=ceil(w-width_h); x<=w+width_h; x++){
                px = x<w ? ceil(x) : floor(x);
                if (px>=W || px<0) continue;
                bf_index = static_cast<int>(py-ceil(h-height_h))*bf_arr_w + (px-ceil(w-width_h));
                bf_values[bf_index] /= bf_sum;
            }
        }
        
        //printf("2--- Starting backward \n");
        for (int n = 0; n < Nx; n++){
            for (int c = 0; c < C; c++){
                int grad_in_index = static_cast<int>(n * (int)round(H/stride0) * (int)round(W/stride1) * C + (int)round(h/stride0) * (int)round(W/stride1) * C + (int)round(w/stride1) * C + c);
                for (double y=ceil(h-height_h); y<=h+height_h; y++){
                    py = y<h ? ceil(y) : floor(y);
                    if (py>=H || py<0)  continue;
                    for (double x=ceil(w-width_h); x<=w+width_h; x++){
                        px = x<w ? ceil(x) : floor(x);
                        if (px>=W || px<0) continue;
                        bf_index = static_cast<int>(py-ceil(h-height_h))*bf_arr_w + (px-ceil(w-width_h));
                        grad_out_index = static_cast<int>(n * H * W * C + py * W * C + px * C + c);
                        atomicAdd(&grad_out[grad_out_index], bf_values[bf_index] * grad_in[grad_in_index]);
                        /*
                        if(
                            (isnan(grad_out[grad_out_index]) || isinf(grad_out[grad_out_index])) && 
                            (!isnan(grad_in[grad_in_index]) && !isinf(grad_in[grad_in_index]))
                        ){
                            printf("grad_out:%f grad_in:%f bf_values:%f bf_sum:%f calc result:%f result is nan:%d, result is inf:%d\n",grad_out[grad_out_index],
                             grad_in[grad_in_index], bf_values[bf_index], bf_sum, bf_values[bf_index] / bf_sum * grad_in[grad_in_index],
                             isnan(bf_values[bf_index] / bf_sum * grad_in[grad_in_index]), isinf(bf_values[bf_index] / bf_sum * grad_in[grad_in_index]));
                        }
                        if(
                            (isnan(grad_out[grad_out_index]) || isinf(grad_out[grad_out_index])) &&
                            (!isnan(grad_in[grad_in_index]) && !isinf(grad_in[grad_in_index])) &&
                            (isnan(bf_values[bf_index] / bf_sum * grad_in[grad_in_index]) || isinf(bf_values[bf_index] / bf_sum * grad_in[grad_in_index]))
                            ){
                                printf("nasel jsem to\n");
                            }*/
                        //if((int)round(w/stride1) == ww && (int)round(h/stride0) == hh && DEBUG_PRINT) printf("[y:%d,x:%d] %3.3f * %3.3f\t", py, px, bf_values[bf_index], grad_in[grad_in_index]);                        
                    }
                    //if((int)round(w/stride1) == ww && (int)round(h/stride0) == hh && DEBUG_PRINT) printf("\n");
                }
            }
        }
        delete[] bf_values;
    }
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

    int const BC_S = 30;
    int block = floor(sqrt(props.maxThreadsPerBlock)) - BC_S;
    int mem_init = 42;
    dim3 dimBlock(block, block);
    dim3 dimGrid((int)ceil(pool_w / dimBlock.x), (int)ceil(pool_h / dimBlock.y));
    //printf("forward dimBlock [%d, %d], dimGrid [%d, %d] \n", dimBlock.x, dimBlock.y, dimGrid.x, dimGrid.y);
    if(DEBUG_PRINT) printf("GPU Forward Start\n");
    if(DEBUG_PRINT) std::cout << typeid(Dtype).name() << std::endl;
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
    cudaDeviceSynchronize();
    if (!d.ok()) {
      ctx->SetStatus(
          tensorflow::errors::Internal("Failed launching FtPool on GPU"));
    }
    if(DEBUG_PRINT) printf("GPU Forward End\n");
    
    //copy output from device to host and print that shit out    
    /*
    if(DEBUG_PRINT){
    size_t size = Nxoutput * Houtput * Woutput * Coutput * sizeof(Dtype);
    printf("Number of components is %d\n", Nxoutput * Houtput * Woutput * Coutput);
    Dtype *p;
    cudaError_t e;
    e = cudaMallocHost(&p, size);
    printf("cudaMallocHost status: %d\n", e);
    e = cudaMemcpy(p, output->flat<Dtype>().data(), size, cudaMemcpyKind::cudaMemcpyDeviceToHost);
    printf("cudaMemcpy status: %d\n", e);
    std::ofstream myLog ("gpu_components.txt");
    for(int n=0; n<Nxoutput; n++){
        for(int y=0; y<Houtput; y++){
            for(int x=0; x<Woutput; x++){
                for(int c=0; c<Coutput; c++){
                    myLog << std::fixed << p[n * Houtput * Woutput * Coutput + y * Woutput * Coutput + x * Coutput + c] << "\t";
                }
            }
            myLog << std::endl;
        } 
    }
    e = cudaFreeHost(p);
    printf("cudaFreeHost status: %d\n", e);
    }*/
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
    const int Nx = grad_out->dim_size(0), H = grad_out->dim_size(1), W = grad_out->dim_size(2), C = grad_out->dim_size(3);

    const float stride0 = stride[0];
    const float stride1 = stride[1];
    const float pool_size0 = pool_size[0];
    const float pool_size1 = pool_size[1];

    float pool_w, pool_h;
    pool_h = H / stride[0];
    pool_w = W / stride[1];

    const GPUDevice& d = ctx->eigen_gpu_device();

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);

    ::tensorflow::CudaLaunchConfig cfg =
        ::tensorflow::GetCudaLaunchConfig(8, d);

    int const BC_S = 30;
    int block = floor(sqrt(props.maxThreadsPerBlock)) - BC_S;
    int mem_init = 42;

    dim3 dimBlock(block, block);
    dim3 dimGrid((int)ceil(pool_w / dimBlock.x), (int)ceil(pool_h / dimBlock.y));

    int pom_s = Nx * H * W * C;
    int gr = (int)ceil((double)(Nx * H * W * C) / (block * block));

    clear<Dtype><<<gr, block*block, 0, d.stream()>>> (
      cfg,
      grad_out->flat<Dtype>().data(),
      pom_s
    );
    if(DEBUG_PRINT) printf("GPU Backward Start\n");
    if(DEBUG_PRINT) std::cout << typeid(Dtype).name() << std::endl;
    backward<Dtype><<<dimGrid, dimBlock, 0, d.stream()>>> (
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
    if(DEBUG_PRINT) printf("GPU Backward End\n");
    cudaDeviceSynchronize();

    if (!d.ok()) {
    printf("NOK");
      ctx->SetStatus(tensorflow::errors::Internal(
          "Failed launching FtPoolGrad on GPU"));
    }
    //copy output from device to host and print that shit out   
   /* if(DEBUG_PRINT){ 
    size_t size = Nx * H * W * C * sizeof(Dtype);
    printf("Number of gradients is %d\n", Nx * H * W * C);
    printf("gpu grad dims[%d, %d, %d, %d]\n", Nx, H, W, C);
    Dtype *p;
    cudaError_t e;
    e = cudaMallocHost(&p, size);
    printf("cudaMallocHost status: %d\n", e);
    e = cudaMemcpy(p, grad_out->flat<Dtype>().data(), size, cudaMemcpyKind::cudaMemcpyDeviceToHost);
    printf("cudaMemcpy status: %d\n", e);    
    std::ofstream myLog ("gpu_gradients.txt");
    for(int n=0; n<Nx; n++){
        for(int y=0; y<H; y++){
            for(int x=0; x<W; x++){
                for(int c=0; c<C; c++){
                    myLog << std::fixed << p[n * H * W * C + y * W * C + x * C + c] << "\t";
                }
            }
            myLog << std::endl;
        } 
    }
    e = cudaFreeHost(p);
    printf("cudaFreeHost status: %d\n", e);
    }*/
  }
};

template struct FtPoolGrad<GPUDevice, float>;
template struct FtPoolGrad<GPUDevice, double>;

}  // namespace functor
}  // namespace tensorflow

//#endif  // GOOGLE_CUDA