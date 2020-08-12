// 2018, Patrick Wieschollek <mail@patwie.com>

#include <cstring>

#include "ft_pool_op.h"
#include "tensorflow/core/framework/op.h"
#include <omp.h>
#include <cmath>

namespace tensorflow {
namespace functor {

template <typename Dtype>
struct FtPoolFunctor<CPUDevice, Dtype> {
    static void launch(::tensorflow::OpKernelContext* ctx, const Tensor& input, Tensor* output, std::vector<float> stride, std::vector<float> pool_size) {
        //std::cout.precision(2);
        //std::cout << "dtype is: " << typeid(Dtype).name() << std::endl;
        const auto in_tensor = input.tensor<Dtype, 4>();
        auto out_tensor = output->tensor<Dtype, 4>();
        out_tensor.setZero();
        // [batch_size, height, width, channels]
        const int N = input.dim_size(0), H = input.dim_size(1), W = input.dim_size(2), C = input.dim_size(3);

        double bf_sum = 0, comp = 0, power_x  = 0, power_y  = 0, bf_value, hh, ww;
        double width_h  = static_cast<double>(pool_size[1]) / 2.0;
        double height_h = static_cast<double>(pool_size[0]) / 2.0;
        double bf_arr_w = floor(2*width_h), bf_arr_h = floor(2*height_h);
        std::array<float, 100> bf_values;

        //std::cout << "****************Forward pass****************" << std::endl;
        //std::cout << "****************input tensor****************" << std::endl;
        for (int y = 0; y < H; y += 1){
            for (int x = 0; x < W; x += 1){
                //std::cout << in_tensor(0,y,x,0) << "\t";
            }
            //std::cout << std::endl;
        }
        //std::cout << "****************input tensor****************" << std::endl;
        for (double h = 0; h < H; h += stride[0]){
            hh = ceil(h-height_h);
            for (double w = 0; w < W; w += stride[1]){
                ww = ceil(w-width_h);
                /************ compute basic fc sum ***********/
                //std::cout << "****************basic fnc[" << h << "," << w << "]****************" << std::endl;
                if (round(w/stride[1]) < output->dim_size(2) && round(h/stride[0]) < output->dim_size(1)){
                    bf_sum = 0;
                    for (double y=hh; y<=floor(h+height_h); y++){
                        if (round(y) >= H) break;
                        if (round(y) < 0) continue;
                        for (double x=ww; x<=floor(w+width_h); x++){
                            if (round(x) >= W) break;
                            if (round(x) < 0) continue;
                            power_x = sin( 3.14*static_cast<double>(1.0 - static_cast<double>(std::abs(x-w) / width_h)) / 2 );
                            power_y = sin( 3.14*static_cast<double>(1.0 - static_cast<double>(std::abs(y-h) / height_h)) / 2 );
                            bf_value = power_x * power_y;
                            bf_sum  += bf_value;
                            bf_values[(y-hh)*bf_arr_w + (x-ww)] = bf_value;
                            //std::cout << bf_values[(y-hh)*bf_arr_w + (x-ww)] << "\t";
                        }
                        //std::cout << std::endl;
                    }
                    //std::cout << "****************basic fnc****************" << std::endl;
                    //std::cout << "****************components****************" << std::endl;
                /************ compute component at x,y for all channels and data samples in batch ***********/
                    for (int n = 0; n < N; n++){
                        for (int c = 0; c < C; c++){
                            if (!bf_sum){
                                out_tensor(n, round(h/stride[0]), round(w/stride[1]), c) = 0;
                            }else{
                            comp = 0;
                            for (double y=hh; y<=floor(h+height_h); y++){
                                if (round(y) >= H) break;
                                if (round(y) < 0) continue;
                                for (double x=ceil(w-width_h); x<=floor(w+width_h); x++){
                                    if (round(x) >= W) break;
                                    if (round(x) < 0) continue;
                                    //power_x = sin( 3.14*static_cast<double>(1.0 - static_cast<double>(std::abs(x-w) / width_h)) / 2 );
                                    //power_y = sin( 3.14*static_cast<double>(1.0 - static_cast<double>(std::abs(y-h) / height_h)) / 2 );
                                    //bf_value = power_x * power_y;
                                    comp += in_tensor(n, static_cast<int>(y), static_cast<int>(x), c) * bf_values[(y-hh)*bf_arr_w + (x-ww)];
                                }
                            }
                                out_tensor(n, round(h/stride[0]), round(w/stride[1]), c) = comp/bf_sum;
                                //std::cout << out_tensor(n, round(h/stride[0]), round(w/stride[1]), c) << "(" << comp/bf_sum << ")\t";
                            }
                        }
                        //std::cout << std::endl;
                    }
                    //std::cout << "****************components****************" << std::endl;
                }
            }
        }
        //std::cout << "****************Forward pass****************" << std::endl;
    }
};

template struct FtPoolFunctor<CPUDevice, int32>;
template struct FtPoolFunctor<CPUDevice, uint32>;
template struct FtPoolFunctor<CPUDevice, float>;
template struct FtPoolFunctor<CPUDevice, double>;

template <typename Dtype>
struct FtPoolGrad<CPUDevice, Dtype> {
    static void launch(::tensorflow::OpKernelContext* ctx, const Tensor& grad_in, Tensor* grad_out, std::vector<float> stride, std::vector<float> pool_size) {
        auto grad_out_tensor = grad_out->tensor<Dtype, 4>();
        grad_out_tensor.setZero();
        // sum_tensor is incialized to shape [H,W]
        //Tensor sum(DT_FLOAT, TensorShape({1, grad_out->dim_size(1), grad_out->dim_size(2), 1}));
        //auto sum_tensor = sum.tensor<Dtype, 4>();
        //sum_tensor.setZero();
        
        auto grad_in_tensor = grad_in.tensor<Dtype, 4>();

        const int N = grad_out->dim_size(0), H = grad_out->dim_size(1), W = grad_out->dim_size(2), C = grad_out->dim_size(3);
        double bf_sum = 0, power_x, power_y, bf_value, hh, ww;
        double width_h  = static_cast<double>(pool_size[1]) / 2.0;
        double height_h = static_cast<double>(pool_size[0]) / 2.0;
        const int bf_arr_w = floor(2*width_h), bf_arr_h = floor(2*height_h);
        std::array<float, 100> bf_values;
        //std::cout << "****************Backward pass****************" << std::endl;
        //std::cout << "****************grad in****************" << std::endl;
        for (int y = 0; y < grad_in.dim_size(1); y += 1){
            for (int x = 0; x < grad_in.dim_size(2); x += 1){
                //std::cout << grad_in_tensor(0,y,x,0) << "\t";
            }
            //std::cout << std::endl;
        }
        //std::cout << "****************grad in****************" << std::endl;
        //std::cout << "****************grad out****************" << std::endl;
        for (int y = 0; y < grad_out->dim_size(1); y += 1){
            for (int x = 0; x < grad_out->dim_size(2); x += 1){
                //std::cout << grad_out_tensor(0,y,x,0) << "\t";
            }
            //std::cout << std::endl;
        }
        //std::cout << "****************grad out****************" << std::endl;
        //std::cout << "****************grad in distribution****************" << std::endl;
        for (double h = 0; h < H; h += stride[0]){
            hh = ceil(h-height_h);
            for (double w = 0; w < W; w += stride[1]){
                ww = ceil(w-width_h);
                /************ compute basic fc sum ***********/
                if (round(w/stride[1]) < grad_in.dim_size(2) && round(h/stride[0]) < grad_in.dim_size(1)){
                    bf_sum = 0;
                    for (double y=hh; y<=floor(h+height_h); y++){
                        if (round(y) >= H) break;
                        if (round(y) < 0) continue;
                        for (double x=ww; x<=floor(w+width_h); x++){
                            if (round(x) >= W) break;
                            if (round(x) < 0) continue;
                            power_x = sin( 3.14*static_cast<double>(1.0 - static_cast<double>(std::abs(x-w) / width_h)) / 2 );
                            power_y = sin( 3.14*static_cast<double>(1.0 - static_cast<double>(std::abs(y-h) / height_h)) / 2 );
                            bf_value = power_x * power_y;
                            bf_values[(y-hh)*bf_arr_w + (x-ww)] = bf_value;
                            bf_sum  += bf_value;
                            //sum_tensor(0,y,x,0) += bf_value;
                        }
                    }
                    if (!bf_sum) bf_sum = 1.0;
                    /************ distribute gradient ***********/
                    //std::cout << "****************grad in[" << h << "," << w << "] distribution****************" << std::endl;
                    for (int n = 0; n < N; n++){
                        for(int c = 0; c < C; c++){
                            for (double y=hh; y<=floor(h+height_h); y++){
                                if (round(y) >= H) break;
                                if (round(y) < 0) continue;
                                for (double x=ww; x<=floor(w+width_h); x++){
                                    if (round(x) >= W) break;
                                    if (round(x) < 0) continue;
                                    //power_x = sin( 3.14*static_cast<double>(1 - static_cast<double>(std::abs(x-(w)) / width_h)) / 2 );
                                    //power_y = sin( 3.14*static_cast<double>(1 - static_cast<double>(std::abs(y-(h)) / height_h)) / 2 );
                                    //if (!power_x || !power_y) continue;
                                    //bf_value = power_x * power_y;
                                    //std::cout << grad_out_tensor(n, y, x, c) << "+";
                                    grad_out_tensor(n, y, x, c) += (grad_in_tensor(n, round(h/stride[0]), round(w/stride[1]), c) * bf_values[(y-hh)*bf_arr_w + (x-ww)]);
                                    //std::cout << grad_in_tensor(n, round(h/stride[0]), round(w/stride[1]), c) << "*" << bf_values[(y-hh)*bf_arr_w + (x-ww)] << "=";
                                    //std::cout << grad_out_tensor(n, y, x, c) << "\t";
                                        //bf_values[yy + (x-ww)]) / bf_sum;
                                }
                                //std::cout << std::endl;
                            }
                        }
                    }
                    //std::cout << "****************grad in[" << h << "," << w << "] distribution****************" << std::endl;
                }
            }
        }/*
        //std::cout << "****************grad out****************" << std::endl;
        for (int n=0; n<N; n++){
            for (int h=0; h<H; h++){
                for (int w=0; w<W; w++){
                    for (int c=0; c<C; c++){
                        //std::cout << grad_out_tensor(n, h, w, c) << "/" << sum_tensor(n, h, w, c) << "\t";
                        if (!sum_tensor(0,h, w,0)) sum_tensor(0,h, w,0) = 1.0;
                            grad_out_tensor(n, h, w, c) /= sum_tensor(0,h, w,0);
                    }
                }
                //std::cout << std::endl;
            }
        }
        //std::cout << "****************grad out****************" << std::endl;
        */
    }
};

template struct FtPoolGrad<CPUDevice, float>;
template struct FtPoolGrad<CPUDevice, double>;

}  // namespace functor
}  // namespace tensorflow
