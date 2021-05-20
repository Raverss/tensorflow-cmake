// 2018, Patrick Wieschollek <mail@patwie.com>

#include <cstring>

#include "ft_inverse_op.h"
#include "tensorflow/core/framework/op.h"
#include <omp.h>
#include <cmath>

namespace tensorflow {
namespace functor {


float absf(float a){return a<0 ? -a : a;}

template <typename Dtype>
struct FtInverseFunctor<CPUDevice, Dtype> {
    static void launch(::tensorflow::OpKernelContext* ctx, const Tensor& input, Tensor* output, std::vector<float> stride, std::vector<float> pool_size) {
        auto output_tensor = output->tensor<Dtype, 4>();
        output_tensor.setZero();
        // sum_tensor is incialized to shape [H,W]
        //Tensor sum(DT_FLOAT, TensorShape({1, grad_out->dim_size(1), grad_out->dim_size(2), 1}));
        //auto sum_tensor = sum.tensor<Dtype, 4>();
        //sum_tensor.setZero();
        
        auto input_tensor = input.tensor<Dtype, 4>();

        const int N = output->dim_size(0), H = output->dim_size(1), W = output->dim_size(2), C = output->dim_size(3);
        double bf_sum = 0, power_x, power_y, bf_value, hh, ww;
        double width_h  = static_cast<double>(pool_size[1]) / 2.0;
        double height_h = static_cast<double>(pool_size[0]) / 2.0;
        const int bf_arr_w = ceil(2*width_h), bf_arr_h = ceil(2*height_h);
        std::array<float, 100> bf_values;

        //std::cout << "\n\n\n\n\n\n\n" << N << " " << H << " " << W << " " << C << std::endl;
        int py, px;

        for (double h = 0; h < H; h += stride[0]){
            if (h/stride[0] >= input.dim_size(1)) continue; //check if round here is necessary
            for (double w = 0; w < W; w += stride[1]){
                if (w/stride[1] >= input.dim_size(2)) continue;
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
                                output_tensor(n, py, px, c) += bf_values[(py-ceil(h-height_h))*bf_arr_w + (px-ceil(w-width_h))] * input_tensor(n, round(h/stride[0]), round(w/stride[1]), c);
                            }
                        }
                    }
                }
            }
        }
    }
};

template struct FtInverseFunctor<CPUDevice, int32>;
template struct FtInverseFunctor<CPUDevice, uint32>;
template struct FtInverseFunctor<CPUDevice, float>;
template struct FtInverseFunctor<CPUDevice, double>;

template <typename Dtype>
struct FtInverseGrad<CPUDevice, Dtype> {
    static void launch(::tensorflow::OpKernelContext* ctx, const Tensor& grad_in, Tensor* grad_out, std::vector<float> stride, std::vector<float> pool_size) {
        const auto grad_in_tensor = grad_in.tensor<Dtype, 4>();
        auto grad_out_tensor = grad_out->tensor<Dtype, 4>();
        grad_out_tensor.setZero();
        // [batch_size, height, width, channels]
        const int N = grad_in.dim_size(0), H = grad_in.dim_size(1), W = grad_in.dim_size(2), C = grad_in.dim_size(3);

        double bf_sum = 0, comp = 0, power_x  = 0, power_y  = 0, bf_value, hh, ww;
        double width_h  = static_cast<double>(pool_size[1]) / 2.0;
        double height_h = static_cast<double>(pool_size[0]) / 2.0;
        double bf_arr_w = ceil(2*width_h), bf_arr_h = ceil(2*height_h);
        std::array<float, 100> bf_values;

        int py, px;

        for (double h = 0; h < H; h += stride[0]){
            if (h/stride[0] >= grad_out->dim_size(1)) continue;
            for (double w = 0; w < W; w += stride[1]){
                if (w/stride[1] >= grad_out->dim_size(2)) continue;
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
                                grad_out_tensor(n, round(h/stride[0]), round(w/stride[1]), c) += bf_values[(py-ceil(h-height_h))*bf_arr_w + (px-ceil(w-width_h))] * grad_in_tensor(n, py, px, c);
                            }
                        }
                        //grad_out_tensor(n, round(h/stride[0]), round(w/stride[1]), c) /= bf_sum;
                    }
                }
            }
        }
    }
};
 
template struct FtInverseGrad<CPUDevice, float>;
template struct FtInverseGrad<CPUDevice, double>;

}  // namespace functor
}  // namespace tensorflow
