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

REGISTER_OP("FtPool")
.Input("x: T")
.Attr("stride: list(float)")
.Attr("T: realnumbertype = DT_FLOAT")
.Output("output: T")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    ShapeHandle shape_hnd;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &shape_hnd));

    ShapeHandle x_shape = c->input(0);
    // save tensor shape into separated variables
    auto N = c->Dim(c->input(0), 0);
    auto H = c->Dim(c->input(0), 1);
    auto W = c->Dim(c->input(0), 2);
    auto C = c->Dim(c->input(0), 3);
    // save stride so we can use it to calculate output shape
    std::vector<float> stride;
    c->GetAttr("stride", &stride);

    c->Divide(H, stride[0], false, &H);
    c->Divide(W, stride[1], false, &W);

    c->set_output(0, c->MakeShape({N, H, W, C}));

    return Status::OK();
})
.Doc(R"doc(

     )doc");

REGISTER_OP("FtPoolGrad")
.Input("x: T")
.Input("gradients: T")
.Output("grad_a: T")
.Attr("T: realnumbertype")
.Attr("stride: list(float)")
.SetShapeFn([](InferenceContext* c) {
    // we require the input to have 4 axes
    ShapeHandle shape_hnd;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &shape_hnd));
    // gradients are definied for every single input value (4D tensor that is being pooled)
    auto N = c->Dim(c->input(0), 0);
    auto H = c->Dim(c->input(0), 1);
    auto W = c->Dim(c->input(0), 2);
    auto C = c->Dim(c->input(0), 3);

    c->set_output(0, c->MakeShape({N, H, W, C}));

    return Status::OK();
})
.Doc(R"doc(
     )doc");

}  // namespace tensorflow
