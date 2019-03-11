// 2018, Patrick Wieschollek <mail@patwie.com>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

using ::tensorflow::shape_inference::InferenceContext;
using ::tensorflow::shape_inference::ShapeHandle;

namespace shape_inference {
Status UnchangedShape(InferenceContext* c) {
  c->set_output(0, c->input(0));
  return Status::OK();
}
}  // namespace shape_inference

REGISTER_OP("MatrixAdd")
    .Input("x: T")
//    .Attr("ksize: list(float)")
    .Attr("stride: list(float)")
//    .Attr("bias: float")
    .Attr("T: realnumbertype = DT_FLOAT")
//    .Input("y: T")
    .Output("output: T")
   .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      // we require the input to have 4 axes
      ShapeHandle shape_hnd;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &shape_hnd));

      ShapeHandle x_shape = c->input(0);

      // specify output-shape
      // this could be "c->set_output(0, x_shape);"
      // but we do it explicitly
      auto N = c->Dim(c->input(0), 0);
      auto H = c->Dim(c->input(0), 1);
      auto W = c->Dim(c->input(0), 2);
      auto C = c->Dim(c->input(0), 3);

      // we can also use the Attr here
      std::vector<float> stride_;
      c->GetAttr("stride", &stride_);

      c->Divide(H, stride_[0], false, &H);
      c->Divide(W, stride_[1], false, &W);

      c->set_output(0, c->MakeShape({N, H, W, C}));

      return Status::OK();
    })
    .Doc(R"doc(
Add two matrices and a constant

This computes `x`+`y`+`bias` for two matrices.
x: A batch of matrices [N, H, W, C].
output: A batch of matrices [N, H, W, C] containing the sum plus bias.
)doc");

REGISTER_OP("MatrixAddGrad")
    .Input("x: T")
    .Input("gradients: T")
    .Output("grad_a: T")
    .Attr("T: realnumbertype")
    .Attr("stride: list(float)")
    .SetShapeFn([](InferenceContext* c) {
        // we require the input to have 4 axes
        ShapeHandle shape_hnd;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &shape_hnd));

        ShapeHandle x_shape = c->input(0);

        // specify output-shape
        // this could be "c->set_output(0, x_shape);"
        // but we do it explicitly
        auto N = c->Dim(c->input(0), 0);
        auto H = c->Dim(c->input(0), 1);
        auto W = c->Dim(c->input(0), 2);
        auto C = c->Dim(c->input(0), 3);

        // we can also use the Attr here
        c->set_output(0, c->MakeShape({N, H, W, C}));

        return Status::OK();
    //c->set_output(0, c->MakeShape({N, H, W, C}));
    //c->set_output(0, c->input(0));  // grad_a has same shape as x
    //  return ::tensorflow::Status::OK();
    })
    .Doc(R"doc(
Returns gradients of "x + y + bias".
)doc");

}  // namespace tensorflow
