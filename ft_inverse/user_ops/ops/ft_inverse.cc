// 2018, Patrick Wieschollek <mail@patwie.com>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "math.h"

namespace tensorflow {

using ::tensorflow::shape_inference::InferenceContext;
using ::tensorflow::shape_inference::ShapeHandle;

namespace shape_inference {
Status UnchangedShape(InferenceContext* c) {
    c->set_output(0, c->input(0));
    return Status::OK();
}
}  // namespace shape_inference

REGISTER_OP("FtInverse")
.Input("x: T")
.Attr("stride: list(float)")
.Attr("pool_size: list(float)")
.Attr("T: realnumbertype = DT_FLOAT")
.Output("output: T")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    //std::cout << "FtPool == Start..." << std::endl;
    ShapeHandle shape_hnd;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &shape_hnd));
    // save tensor shape into separated variables
    auto N = c->Dim(c->input(0), 0);
    auto H = c->Dim(c->input(0), 1);
    auto W = c->Dim(c->input(0), 2);
    auto C = c->Dim(c->input(0), 3);
    if(c->Value(H) < 0 || c->Value(W) < 0)
        return Status(tensorflow::error::Code::INVALID_ARGUMENT, "Ft inverse was called with negative width or/and height values");
    // save stride so we can use it to calculate output shape
    std::vector<float> stride;
    c->GetAttr("stride", &stride);
    float abc;
    //std::cout << "abc is " << abc << std::endl;
    //std::cout << stride[0] << ", " << stride[1] << std::endl;
    H = c->MakeDim(round(c->Value(H)*stride[0]));
    W = c->MakeDim(round(c->Value(W)*stride[1]));
    c->set_output(0, c->MakeShape({N, H, W, C}));
    //std::cout << "...FtPool == End" << std::endl << std::endl;
    return Status::OK();
})
.Doc(R"doc(

     )doc");

REGISTER_OP("FtInverseGrad")
.Input("x: T")
.Input("gradients: T")
.Output("grad_a: T")
.Attr("T: realnumbertype")
.Attr("stride: list(float)")
.Attr("pool_size: list(float)")
.SetShapeFn([](InferenceContext* c) {
    //std::cout << "\nFtPoolGrad == Start..." << std::endl;
    // we require the input to have 4 axes
    ShapeHandle shape_hnd;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &shape_hnd));
    // gradients are definied for every single input value (4D tensor that is being pooled)
    auto N = c->Dim(c->input(0), 0);
    auto H = c->Dim(c->input(0), 1);
    auto W = c->Dim(c->input(0), 2);
    auto C = c->Dim(c->input(0), 3);

    c->set_output(0, c->MakeShape({N, H, W, C}));
    //std::cout << "...FtPoolGrad == End" << std::endl << std::endl;
    return Status::OK();
})
.Doc(R"doc(
     )doc");

}  // namespace tensorflow
