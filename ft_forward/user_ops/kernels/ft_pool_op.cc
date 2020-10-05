// 2018, Patrick Wieschollek <mail@patwie.com>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

#include "ft_pool_op.h"

namespace tensorflow {

// Forward-Pass (CPU, GPU)
// --------------------------------------------------
template <typename Device, typename Dtype>
class FtPoolOp : public OpKernel {
 public:
  explicit FtPoolOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("stride", &stride));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("pool_size", &pool_size));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input = ctx->input(0);

    if (!ctx->status().ok()) {
      return;
    }

    const int N = input.dim_size(0);
    const int H = input.dim_size(1);
    const int W = input.dim_size(2);
    const int C = input.dim_size(3);

    TensorShape output_shape({N, static_cast<int>(round(H/stride[0])), static_cast<int>(round(W/stride[1])), C});

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));
    ::tensorflow::functor::FtPoolFunctor<Device, Dtype>::launch(ctx, input, output, stride, pool_size);
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(FtPoolOp);
  std::vector<float> stride;
  std::vector<float> pool_size;
};

// Backward-Pass (CPU, GPU)
// --------------------------------------------------
template <typename Device, typename Dtype>
class FtPoolGradOp : public OpKernel {
 public:
  explicit FtPoolGradOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("stride", &stride));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("pool_size", &pool_size));
    }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input = ctx->input(0);
    const Tensor& grad_in = ctx->input(1);

    if (!ctx->status().ok()) {
      return;
    }

    Tensor* grad_out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input.shape(), &grad_out));
    ::tensorflow::functor::FtPoolGrad<Device, Dtype>::launch(ctx, grad_in, grad_out, stride, pool_size);
  }

private:
  std::vector<float> stride;
  std::vector<float> pool_size;
};

#define REGISTER_CUSTOM_OP(NAME, DEVICE, T)                       \
  REGISTER_KERNEL_BUILDER(                                        \
      Name(#NAME).Device(DEVICE_##DEVICE).TypeConstraint<T>("T"), \
      NAME##Op<DEVICE##Device, T>)

REGISTER_CUSTOM_OP(FtPool, CPU, uint32);
REGISTER_CUSTOM_OP(FtPool, CPU, int32);
REGISTER_CUSTOM_OP(FtPool, CPU, float);
REGISTER_CUSTOM_OP(FtPool, CPU, double);
REGISTER_CUSTOM_OP(FtPoolGrad, CPU, float);
REGISTER_CUSTOM_OP(FtPoolGrad, CPU, double);

}  // namespace tensorflow
