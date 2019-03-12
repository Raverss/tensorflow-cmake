// 2018, Patrick Wieschollek <mail@patwie.com>

#include <cstring>

#include "ft_pool_op.h"
#include "tensorflow/core/framework/op.h"

namespace tensorflow {
namespace functor {

template <typename Dtype>
struct FtPoolFunctor<CPUDevice, Dtype> {
    static void launch(::tensorflow::OpKernelContext* ctx, const Tensor& input, Tensor* output, std::vector<float> stride) {
        const auto in_tensor = input.tensor<Dtype, 4>();
        auto out_tensor = output->tensor<Dtype, 4>();

        out_tensor.setZero();

        const int N = output->dim_size(0);
        const int H = output->dim_size(1);
        const int W = output->dim_size(2);
        const int C = output->dim_size(3);

        float sum;
        for (int n = 0; n < N; ++n){
            for (int c = 0; c < C; ++c){
                for (int h = 0; h < H; ++h){
                    for (int w = 0; w < W; ++w){
                        sum = 0;
                        sum += in_tensor(n, h*2, w*2, c);
                        sum += in_tensor(n, (h+1)*2, w*2, c);
                        sum += in_tensor(n, h*2, (w+1)*2, c);
                        sum += in_tensor(n, (h+1)*2, (w+1)*2, c);
                        out_tensor(n, h, w, c) = sum/4.0f;
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
        const int N = grad_in.dim_size(0);
        const int H = grad_in.dim_size(1);
        const int W = grad_in.dim_size(2);
        const int C = grad_in.dim_size(3);

        float p = 0.25;
        for (int n = 0; n < N; n++){
            for(int c = 0; c < C; c++){
                for (int h = 0; h < H; h++){
                    for (int w = 0; w < W; w++){
                        grad_out_tensor(n, h*2, w*2, c) = p * grad_in_tensor(n, h, w, c);
                        grad_out_tensor(n, h*2+1, w*2, c) = p * grad_in_tensor(n, h, w, c);
                        grad_out_tensor(n, h*2, w*2+1, c) = p * grad_in_tensor(n, h, w, c);
                        grad_out_tensor(n, h*2+1, w*2+1, c) = p * grad_in_tensor(n, h, w, c);
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
