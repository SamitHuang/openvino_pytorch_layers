// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "extension.hpp"
#include "cpu_kernel.hpp"
#include "op.hpp"
#include <ngraph/factory.hpp>
#include <ngraph/opsets/opset.hpp>

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

using namespace TemplateExtension;

//! [extension:GetVersion]
void Extension::GetVersion(const InferenceEngine::Version *&versionInfo) const noexcept {
    static InferenceEngine::Version ExtensionDescription = {
        {1, 0},           // extension API version
        "1.0",
        "template_ext"    // extension description message
    };

    versionInfo = &ExtensionDescription;
}
//! [extension:GetVersion]

//! [extension:getOpSets]
std::map<std::string, ngraph::OpSet> Extension::getOpSets() {
    std::map<std::string, ngraph::OpSet> opsets;
    ngraph::OpSet opset;
    //TODO: add your op
    opset.insert<UnpoolOp>();
    opset.insert<FFTOp>();
    opset.insert<GridSampleOp>();
    opset.insert<MultinomialOp>();

    opsets["extension"] = opset;
    return opsets;
}
//! [extension:getOpSets]

//! [extension:getImplTypes]
std::vector<std::string> Extension::getImplTypes(const std::shared_ptr<ngraph::Node> &node) {
    //TODO: add your op
    if (std::dynamic_pointer_cast<UnpoolOp>(node) ||
        std::dynamic_pointer_cast<GridSampleOp>(node) ||
        std::dynamic_pointer_cast<FFTOp>(node) ||
        std::dynamic_pointer_cast<MultinomialOp>(node) 
        ) {
        return {"CPU"};
    }
    return {};
}
//! [extension:getImplTypes]

//! [extension:getImplementation]
InferenceEngine::ILayerImpl::Ptr Extension::getImplementation(const std::shared_ptr<ngraph::Node> &node, const std::string &implType) {
    if (std::dynamic_pointer_cast<UnpoolOp>(node) && implType == "CPU") {
        return std::make_shared<UnpoolImpl>(node);
    }
    if (std::dynamic_pointer_cast<FFTOp>(node) && implType == "CPU") {
        return std::make_shared<FFTImpl>(node);
    }
    if (std::dynamic_pointer_cast<GridSampleOp>(node) && implType == "CPU") {
        return std::make_shared<GridSampleImpl>(node);
    }
    if (std::dynamic_pointer_cast<MultinomialOp>(node) && implType == "CPU") {
        return std::make_shared<MultinomialImpl>(node);
    }
    return nullptr;
}
//! [extension:getImplementation]

//! [extension:CreateExtension]
// Exported function
INFERENCE_EXTENSION_API(InferenceEngine::StatusCode) InferenceEngine::CreateExtension(InferenceEngine::IExtension *&ext,
                                                                                      InferenceEngine::ResponseDesc *resp) noexcept {
    try {
        ext = new Extension();
        return OK;
    } catch (std::exception &ex) {
        if (resp) {
            std::string err = ((std::string) "Couldn't create extension: ") + ex.what();
            err.copy(resp->msg, 255);
        }
        return InferenceEngine::GENERAL_ERROR;
    }
}
//! [extension:CreateExtension]
