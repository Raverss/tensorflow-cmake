// 2018, Patrick Wieschollek <mail@patwie.com>

#include <cstring>

#include "matrix_add_op.h"
#include "tensorflow/core/framework/op.h"

namespace tensorflow {
namespace functor {

template <typename Dtype>
struct MatrixAddFunctor<CPUDevice, Dtype> {
  static void launch(::tensorflow::OpKernelContext* ctx, const Tensor& Xt, Tensor* Zt, std::vector<float> stride_) {
    const auto X = Xt.tensor<Dtype, 4>();
    auto Z = Zt->tensor<Dtype, 4>();

    Z.setZero();

/*    std::cout << "printing value of stride size: (";
    for (auto i = stride_.begin(); i != stride_.end(); ++i)
        std::cout << *i << ',';
    std::cout << ")" << std::endl;*/

    // get dimensions
/*    const int N = Xt.dim_size(0);
    const int H = Xt.dim_size(1);
    const int W = Xt.dim_size(2);
    const int C = Xt.dim_size(3);*/

//    std::cout << "Zt size is " << Zt->dim_size(0) << " " << Zt->dim_size(1) << " " << Zt->dim_size(2) << " " << Zt->dim_size(3) << std::endl;

    /*float val;
    for (int n = 0; n < N; ++n){
        for (int c = 0; c < C; ++c){
            for (float h = 0; h < H; h += stride_[0]){
                for (float w = 0; w < W; w += stride_[1]){
                    if (round(w/stride_[1]) < Zt->dim_size(2) && round(h/stride_[0]) < Zt->dim_size(1)){
                        val = compute_component(w, h, stride_[1]*2, stride_[0]*2, W, H, X, n, c);
//                        std::cout << "Z("<< n << "," << round(h/stride_[0]) << "," << round(w/stride_[1]) << "," << c << ") = " << val << std::endl;
                        //Z(n, static_cast<int>(round(h/stride_[0])), static_cast<int>(round(w/stride_[1])), c) = val;

                    }
                }
            }
        }
    }*/

    // the computation (easy to read)
    const int N = Zt->dim_size(0);
    const int H = Zt->dim_size(1);
    const int W = Zt->dim_size(2);
    const int C = Zt->dim_size(3);
    float sum;
    for (int n = 0; n < N; ++n){
        for (int c = 0; c < C; ++c){
            for (int h = 0; h < H; ++h){
                sum = 0;
                for (int w = 0; w < W; ++w){
                    sum += X(n, h*2, w*2, c);
                    sum += X(n, (h+1)*2, w*2, c);
                    sum += X(n, h*2, (w+1)*2, c);
                    sum += X(n, (h+1)*2, (w+1)*2, c);
                    Z(n, h, w, c) = sum/4.0f;
                }
            }
        }
    }
  }

  static float compute_component(int start_x, int start_y, float width, float height, int source_w, int source_h,
                                 Eigen::TensorMap<Eigen::Tensor<const Dtype, 4, 1, long int>, 16, Eigen::MakePointer> X,
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
//      comp = comp < 0 ? 0 : comp > 255 ? 255 : comp;
      return comp;
    }
};



template struct MatrixAddFunctor<CPUDevice, int32>;
template struct MatrixAddFunctor<CPUDevice, uint32>;
template struct MatrixAddFunctor<CPUDevice, float>;
template struct MatrixAddFunctor<CPUDevice, double>;

template <typename Dtype>
struct MatrixAddGrad<CPUDevice, Dtype> {
  static void launch(::tensorflow::OpKernelContext* ctx, const Tensor& topdiff_,
                     Tensor* grad_mA_) {
    //const int W = topdiff_.NumElements();
//    std::cout << "grad_mA_ shape is " << grad_mA_->shape() << std::endl; //grad_mA_ shape is [32,26,26,2]
//    std::cout << "topdiff_ shape is " << topdiff_.shape() << std::endl; //topdiff_ shape is [32,13,13,2]
    grad_mA_->flat<Dtype>().setZero();   
    //Dtype* grad_X = grad_mA_->flat<Dtype>().data();
    auto grad_X = grad_mA_->tensor<Dtype, 4>();
    //const Dtype* topdiff = topdiff_.flat<Dtype>().data();
    auto topdiff = topdiff_.tensor<Dtype, 4>();
    const int N = topdiff_.dim_size(0);
    const int H = topdiff_.dim_size(1);
    const int W = topdiff_.dim_size(2);
    const int C = topdiff_.dim_size(3);
//    std::cout << "N " << N << " H " << H << " W " << W << " C " << C << std::endl; //N 32 H 13 W 13 C 2
    float p = 0.25;
    for (int n = 0; n < N; n++){
        for(int c = 0; c < C; c++){
            for (int h = 0; h < H; h++){
                for (int w = 0; w < W; w++){
                    /*grad_X[n * N + h*2 * H + w*2 * W + c] = p * topdiff[n * N + h * H + w * W + c];
                    grad_X[n * N + (h*2 + 1) * H + w*2 * W + c] = p * topdiff[n * N + h * H + w * W + c];
                    grad_X[n * N + h*2 * H + (w*2 + 1) * W + c] = p * topdiff[n * N + h * H + w * W + c];
                    grad_X[n * N + (h*2 + 1) * H + (w*2 + 1) * W + c] = p * topdiff[n * N + h * H + w * W + c];*/
                    grad_X(n, h*2, w*2, c) = p * topdiff(n, h, w, c);
                    grad_X(n, h*2+1, w*2, c) = p * topdiff(n, h, w, c);
                    grad_X(n, h*2, w*2+1, c) = p * topdiff(n, h, w, c);
                    grad_X(n, h*2+1, w*2+1, c) = p * topdiff(n, h, w, c);
                }
            }
        }
    }
/*    grad_mA_->flat<Dtype>().setZero();
    grad_mB_->flat<Dtype>().setZero();

    const Dtype* topdiff = topdiff_.flat<Dtype>().data();
    Dtype* grad_X = grad_mA_->flat<Dtype>().data();
    Dtype* grad_Y = grad_mB_->flat<Dtype>().data();

    std::memcpy(grad_X, topdiff, W * sizeof(Dtype));
    std::memcpy(grad_Y, topdiff, W * sizeof(Dtype));
    // for (int i = 0; i < W; ++i) {
    //   grad_X[i] = topdiff[i];
    //   grad_Y[i] = topdiff[i];
    // }*/
  }
};

template struct MatrixAddGrad<CPUDevice, float>;
template struct MatrixAddGrad<CPUDevice, double>;

}  // namespace functor
}  // namespace tensorflow
