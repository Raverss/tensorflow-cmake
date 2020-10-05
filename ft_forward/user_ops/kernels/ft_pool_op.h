// 2018, Patrick Wieschollek <mail@patwie.com>

#ifndef FT_POOL_KERNELS_FT_POOL_OP_H_
#define FT_POOL_KERNELS_FT_POOL_OP_H_

#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace functor {

template <typename Device, typename Dtype>
struct FtPoolFunctor {
    static void launch(::tensorflow::OpKernelContext* ctx, const Tensor& input, Tensor* output, std::vector<float> stride_, std::vector<float> pool_size_);

};

template <typename Device, typename Dtype>
struct FtPoolGrad {
  static void launch(::tensorflow::OpKernelContext* ctx, const Tensor& gradients, Tensor* input, std::vector<float> stride_, std::vector<float> pool_size_);
};

}  // namespace functor
}  // namespace tensorflow

#endif  // FT_POOL_KERNELS_FT_POOL_OP_H_
