// 2018, Patrick Wieschollek <mail@patwie.com>

#include <cstring>

#include "ft_pool_op.h"
#include "tensorflow/core/framework/op.h"

namespace tensorflow {
namespace functor {

static float compute_component(int start_x, int start_y, float width, float height, int source_w, int source_h,
                                Eigen::TensorMap<Eigen::Tensor<const int, 4, 1, long int>, 16, Eigen::MakePointer> const X,
                                 int n, int c){
    double sum      = 0;
    double comp     = 0;
    double power_x  = 0;
    double power_y  = 0;
    double width_h  = static_cast<double>((width-1))/2.0;
    double height_h = static_cast<double>((height-1))/2.0;
    double w = 0;
    for (double y=start_y-height_h; y<=start_y+height_h; y++){
        if (round(y) >= source_h) continue;
        if (round(y) < 0) continue;
        for (double x=start_x-width_h; x<=start_x+width_h; x++){
            if (round(x) >= source_w) continue;
            if (round(x) < 0) continue;
            power_x = sin( 3.14*static_cast<double>(1 - static_cast<double>(abs(x-(start_x)) / width_h)) / 2 );
            power_y = sin( 3.14*static_cast<double>(1 - static_cast<double>(abs(y-(start_y)) / height_h)) / 2 );
            if (!power_x || !power_y) continue;
            w = power_x * power_y;
            /*comp += X(n, static_cast<int>(round(y)), static_cast<int>(round(w)), c) * w;
            sum  += w;*/
            comp += X(n, static_cast<int>(round(y)), static_cast<int>(round(w)), c);
            sum += 1;
        }
    }
    if (!sum) return 0;
    comp /=sum;
//    comp = comp < 0 ? 0 : comp > 255 ? 255 : comp;
    return comp;
}

template <typename Dtype>
struct FtPoolFunctor<CPUDevice, Dtype> {
    static void launch(::tensorflow::OpKernelContext* ctx, const Tensor& input, Tensor* output, std::vector<float> stride) {
        const auto in_tensor = input.tensor<Dtype, 4>();
        auto out_tensor = output->tensor<Dtype, 4>();

        out_tensor.setZero();

        /*const int N = output->dim_size(0);
        const int H = output->dim_size(1);
        const int W = output->dim_size(2);
        const int C = output->dim_size(3);*/
        const int N = input.dim_size(0);
        const int H = input.dim_size(1);
        const int W = input.dim_size(2);
        const int C = input.dim_size(3);
        //float sum;
        //float val;
        /**************** compute components ******************/
        //static float compute_component(int start_x, int start_y, float width, float height, int source_w, int source_h,
                                //Eigen::TensorMap<Eigen::Tensor<const int, 4, 1, long int>, 16, Eigen::MakePointer> const X,
                                 //int n, int c){
        double bf_sum   = 0;
        double comp     = 0;
        double power_x  = 0;
        double power_y  = 0;
        double width_h  = static_cast<double>((stride[1]*2-1))/2.0;
        double height_h = static_cast<double>((stride[0]*2-1))/2.0;
        double bf_value = 0;
        /**************** compute components ******************/
        for (int n = 0; n < N; ++n){
            for (int c = 0; c < C; ++c){
                //for (int h = 0; h < H; ++h){
                    //for (int w = 0; w < W; ++w){
                for (float h = 0; h < H; h += stride[0]){
                    for (float w = 0; w < W; w += stride[1]){
                        /*sum = 0;
                        sum += in_tensor(n, h*2, w*2, c);
                        sum += in_tensor(n, h*2+1, w*2, c);
                        sum += in_tensor(n, h*2, w*2+1, c);
                        sum += in_tensor(n, h*2+1, w*2+1, c);
                        out_tensor(n, h, w, c) = sum/4.0f;*/
                        if (round(w/stride[1]) < output->dim_size(2) && 
                            round(h/stride[0]) < output->dim_size(1)){
                        /**/
                        bf_sum = 0; comp = 0; power_x = 0; power_y = 0; bf_value = 0;
                        for (double y=h-height_h; y<=h+height_h; y++){
                                if (round(y) >= H) continue;
                                if (round(y) < 0) continue;
                                for (double x=w-width_h; x<=w+width_h; x++){
                                    if (round(x) >= W) continue;
                                    if (round(x) < 0) continue;
                                    power_x = sin( 3.14*static_cast<double>(1 - static_cast<double>(abs(x-(w)) / width_h)) / 2 );
                                    power_y = sin( 3.14*static_cast<double>(1 - static_cast<double>(abs(y-(h)) / height_h)) / 2 );
                                    if (!power_x || !power_y) continue;
                                    bf_value = power_x * power_y;
                                    comp += in_tensor(n, static_cast<int>(round(y)), static_cast<int>(round(x)), c) * bf_value;
                                    bf_sum  += bf_value;
                                }
                            }
                            
                            if (!bf_sum) comp = 0;
                            comp /=bf_sum;
                            out_tensor(n, round(h/stride[0]), round(w/stride[1]), c) = comp;
                        //    comp = comp < 0 ? 0 : comp > 255 ? 255 : comp;
                        
                        /**/
                        //val = compute_component(w, h, stride[1]*2, stride[0]*2, W, H, in_tensor, n, c);
                        //std::cout << "Z("<< n << "," << round(h/stride_[0]) << "," << round(w/stride_[1]) << "," << c << ") = " << val << std::endl;
                        //Z(n, static_cast<int>(round(h/stride_[0])), static_cast<int>(round(w/stride_[1])), c) = val;
                        }
                    }
                }
            }
        }
    }
};

template struct FtPoolFunctor<CPUDevice, int32>;
template struct FtPoolFunctor<CPUDevice, uint32>;
template struct FtPoolFunctor<CPUDevice, float>;
template struct FtPoolFunctor<CPUDevice, double>;

template <typename Dtype>
struct FtPoolGrad<CPUDevice, Dtype> {
    static void launch(::tensorflow::OpKernelContext* ctx, const Tensor& grad_in, Tensor* grad_out) {
        grad_out->flat<Dtype>().setZero();
        auto grad_out_tensor = grad_out->tensor<Dtype, 4>();
        auto grad_in_tensor = grad_in.tensor<Dtype, 4>();
        /*const int N = grad_in.dim_size(0);
        const int H = grad_in.dim_size(1);
        const int W = grad_in.dim_size(2);
        const int C = grad_in.dim_size(3);*/
        const int N = grad_out->dim_size(0);
        const int H = grad_out->dim_size(1);
        const int W = grad_out->dim_size(2);
        const int C = grad_out->dim_size(3);
        std::vector<float> stride = {(float)H/grad_in.dim_size(1), (float)W/grad_in.dim_size(2)}; 

        double bf_sum   = 0;
        double power_x;
        double power_y;
        double width_h  = static_cast<double>((stride[1]*2-1))/2.0;
        double height_h = static_cast<double>((stride[0]*2-1))/2.0;
        double bf_value = 0;
        //float p = 0.25;
        for (int n = 0; n < N; n++){
            for(int c = 0; c < C; c++){
                /*for (int h = 0; h < H; h++){
                    for (int w = 0; w < W; w++){
                        grad_out_tensor(n, h*2, w*2, c) = p * grad_in_tensor(n, h, w, c);
                        grad_out_tensor(n, h*2+1, w*2, c) = p * grad_in_tensor(n, h, w, c);
                        grad_out_tensor(n, h*2, w*2+1, c) = p * grad_in_tensor(n, h, w, c);
                        grad_out_tensor(n, h*2+1, w*2+1, c) = p * grad_in_tensor(n, h, w, c);
                    }
                }*/
                for (float h = 0; h < H; h += stride[0]){
                    for (float w = 0; w < W; w += stride[1]){
                        /*sum = 0;
                        sum += in_tensor(n, h*2, w*2, c);
                        sum += in_tensor(n, h*2+1, w*2, c);
                        sum += in_tensor(n, h*2, w*2+1, c);
                        sum += in_tensor(n, h*2+1, w*2+1, c);
                        out_tensor(n, h, w, c) = sum/4.0f;*/
                        if (round(w/stride[1]) < grad_in.dim_size(2) && 
                            round(h/stride[0]) < grad_in.dim_size(1)){
                        /**/
                        bf_sum = 0; bf_value = 0;
                        for (double y=h-height_h; y<=h+height_h; y++){
                                if (round(y) >= H) continue;
                                if (round(y) < 0) continue;
                                for (double x=w-width_h; x<=w+width_h; x++){
                                    if (round(x) >= W) continue;
                                    if (round(x) < 0) continue;
                                    power_x = sin( 3.14*static_cast<double>(1 - static_cast<double>(abs(x-(w)) / width_h)) / 2 );
                                    power_y = sin( 3.14*static_cast<double>(1 - static_cast<double>(abs(y-(h)) / height_h)) / 2 );
                                    if (!power_x || !power_y) continue;
                                    bf_value = power_x * power_y;
                                    bf_sum  += bf_value;
                                }
                            }
                        if (!bf_sum) bf_sum = 1.0;
                        for (double y=h-height_h; y<=h+height_h; y++){
                                if (round(y) >= H) continue;
                                if (round(y) < 0) continue;
                                for (double x=w-width_h; x<=w+width_h; x++){
                                    if (round(x) >= W) continue;
                                    if (round(x) < 0) continue;
                                    power_x = sin( 3.14*static_cast<double>(1 - static_cast<double>(abs(x-(w)) / width_h)) / 2 );
                                    power_y = sin( 3.14*static_cast<double>(1 - static_cast<double>(abs(y-(h)) / height_h)) / 2 );
                                    if (!power_x || !power_y) continue;
                                    bf_value = power_x * power_y;
                                    grad_out_tensor(n, y, x, c) = grad_in_tensor(n, round(h/stride[0]), round(w/stride[1]), c) * bf_value / bf_sum;
                                }
                            }
                        //    comp = comp < 0 ? 0 : comp > 255 ? 255 : comp;
                        
                        /**/
                        //val = compute_component(w, h, stride[1]*2, stride[0]*2, W, H, in_tensor, n, c);
                        //std::cout << "Z("<< n << "," << round(h/stride_[0]) << "," << round(w/stride_[1]) << "," << c << ") = " << val << std::endl;
                        //Z(n, static_cast<int>(round(h/stride_[0])), static_cast<int>(round(w/stride_[1])), c) = val;
                        }
                    }
                }
            }
        }
    }
};

template struct FtPoolGrad<CPUDevice, float>;
template struct FtPoolGrad<CPUDevice, double>;

}  // namespace functor
}  // namespace tensorflow
