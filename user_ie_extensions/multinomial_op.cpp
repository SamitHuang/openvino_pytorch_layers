// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "op.hpp"

using namespace TemplateExtension;

constexpr ngraph::NodeTypeInfo MultinomialOp::type_info;

//TODO: inputs and attributes for the op
//! [op:ctor]
MultinomialOp::MultinomialOp(const ngraph::Output<ngraph::Node>& inp) : Op({inp}) {
    constructor_validate_and_infer_types();
}
//! [op:ctor]

//TODO: infer
//! [op:validate]
void MultinomialOp::validate_and_infer_types() {
    //Input: [batch_size, K_probs]
    //Output: [batch_size, 1] 
    auto outShape = get_input_partial_shape(0);  
    //printf("=====>From multinomial_op.cpp,  Debug: input shape is: %d %d \r\n", outShape[0], outShape[1]);
    outShape[1] = 1;
    //printf("=====>From multinomial_op.cpp,  Debug: output shape is: %d %d \r\n", outShape[0], outShape[1]);
    //TODO: change the output type and shape if needed! set output node 0 as dtype int32 with outShape
    set_output_type(0, ngraph::element::i32, outShape);
}
//! [op:validate]

//! [op:copy]
std::shared_ptr<ngraph::Node> MultinomialOp::clone_with_new_inputs(const ngraph::OutputVector &new_args) const {
    //TODO: != #input ports
    if (new_args.size() != 1) {
        throw ngraph::ngraph_error("Incorrect number of new arguments");
    }
    //TODO: include all input ports and used attributes 
    return std::make_shared<MultinomialOp>(new_args.at(0));
}
//! [op:copy]

//! [op:visit_attributes]
bool MultinomialOp::visit_attributes(ngraph::AttributeVisitor &visitor) {
    //TODO: if we need to use attributes, ref to shuffle_channels.cpp
    return true;
}
//! [op:visit_attributes]
