// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "cpu_kernel.hpp"
#include "op.hpp"
#include <details/ie_exception.hpp>
#include <ie_layouts.h>
#include "ie_parallel.hpp"

using namespace TemplateExtension;

//! [cpu_implementation:ctor]
MultinomialImpl::MultinomialImpl(const std::shared_ptr<ngraph::Node> &node) {
    try {
        auto castedNode = std::dynamic_pointer_cast<MultinomialOp>(node);
        if (!castedNode)
            THROW_IE_EXCEPTION << "Cannot create implementation for unknown operation!";
        //TODO: # of input ports check, change it accordingly 
        if (castedNode->inputs().size() != 1 || castedNode->outputs().size() != 1)
            THROW_IE_EXCEPTION << "Cannot create implementation for operation with incorrect number of inputs or outputs!";
        if (castedNode->get_input_partial_shape(0).is_dynamic() || castedNode->get_output_partial_shape(0).is_dynamic())
            THROW_IE_EXCEPTION << "Cannot create implementation for op with dynamic shapes!";
        //TODO: # dim of the input and output, change accord.. (4 NCHW for image) 
        if (castedNode->get_input_shape(0).size() != 2 || castedNode->get_output_shape(0).size() != 2)
            THROW_IE_EXCEPTION << "Operation supports only 2d tensors for input and output.";
        //TODO: input/output datatype check, change a.. 
        if (castedNode->get_input_element_type(0) != ngraph::element::f32 || castedNode->get_output_element_type(0) != ngraph::element::i32){
            THROW_IE_EXCEPTION << "Operation supports only input FP32 tensors and output INT32 tensors." << "But current input type and output type:  " << castedNode->get_input_element_type(0)  << castedNode->get_output_element_type(0);
        }
        //TODO: set input and output shape. 
        inpShape = castedNode->get_input_shape(0);
        outShape = castedNode->get_output_shape(0); 
        // sample_size = castedNode->sample_size; 
    } catch (InferenceEngine::details::InferenceEngineException& ex) {
        error = ex.what();
    }
    printf("=> Debug MulImpl inpShape");
    //printf("=> Debug MulImpl inpShape %d %d, outShape %d %d \r\n", inpShape[0], inpShape[1], outShape[0], outShape[1]);
}
//! [cpu_implementation:ctor]

//! [cpu_implementation:getSupportedConfigurations]
InferenceEngine::StatusCode MultinomialImpl::getSupportedConfigurations(std::vector<InferenceEngine::LayerConfig> &conf,
                                                                         InferenceEngine::ResponseDesc *resp) noexcept {
     std::vector<InferenceEngine::DataConfig> inDataConfig;
     std::vector<InferenceEngine::DataConfig> outDataConfig;
     InferenceEngine::SizeVector order = {0, 1}; //TODO: what is it? check correctness. ----------------------------> <---------------------
     
     //printf("=> Debug getSupportedConfig 0\r\n");
     printf("Flag 1\r\n"); 
     // Allow any offset before data
     size_t offset((std::numeric_limits<size_t>::max)());
     //printf("=> Debug inpShape %d, %d\r\n", inpShape[0], inpShape[1]);
     // Input shape TODO: change a.. data type and shape, order 
     InferenceEngine::DataConfig inpConf;
     //inpConf.desc = InferenceEngine::TensorDesc(InferenceEngine::Precision::FP32, inpShape, {inpShape, order, offset});
    
     inpConf.desc = InferenceEngine::TensorDesc(InferenceEngine::Precision::FP32, inpShape, InferenceEngine::Layout::NC);
     inDataConfig.push_back(inpConf);

     printf("Flag 4\r\n"); 
     //printf("=> Debug outShape %d, %d\r\n", outShape[0], outShape[1]); //wrong
     // Output shape. TODO: change a.. 
     InferenceEngine::DataConfig outConf;
     //outConf.desc = InferenceEngine::TensorDesc(InferenceEngine::Precision::I32, outShape, {outShape, order, offset});
     outConf.desc = InferenceEngine::TensorDesc(InferenceEngine::Precision::I32, outShape, InferenceEngine::Layout::NC);
     outDataConfig.push_back(outConf);

     InferenceEngine::LayerConfig layerConfig;
     layerConfig.inConfs = inDataConfig;
     layerConfig.outConfs = outDataConfig;

     conf.push_back(layerConfig);
    
     printf("=> getSupportedConfig 2\r\n");

     return InferenceEngine::StatusCode::OK;
}
//! [cpu_implementation:getSupportedConfigurations]

//! [cpu_implementation:init]
InferenceEngine::StatusCode MultinomialImpl::init(InferenceEngine::LayerConfig &config, InferenceEngine::ResponseDesc *resp) noexcept {
    printf("start init\r\n");
    try {
        // TODO: change a...
        if (config.inConfs.size() != 1 || config.outConfs.size() != 1) {
            THROW_IE_EXCEPTION << "Operation cannot be initialized with incorrect number of inputs/outputs!";
        }

        if (config.inConfs[0].desc.getDims().size() != 2 || config.outConfs[0].desc.getDims().size() != 2) {
            THROW_IE_EXCEPTION << "Operation can be initialized only with 2d input/output tensors!";
        }

        if (config.outConfs[0].desc.getPrecision() != InferenceEngine::Precision::I32 ||
                config.inConfs[0].desc.getPrecision() != InferenceEngine::Precision::FP32)  {
            THROW_IE_EXCEPTION << "Operation supports only FP32 precisions for input and I32 for output!";
        }
    } catch (InferenceEngine::details::InferenceEngineException& ex) {
        if (resp) {
            strncpy(resp->msg, error.c_str(), sizeof(resp->msg) - 1);
            resp->msg[sizeof(resp->msg)-1] = 0;
        }
        return InferenceEngine::GENERAL_ERROR;
    }

    printf("=> Debug init done \r\n");
    return InferenceEngine::OK;
}
//! [cpu_implementation:init]

//! [cpu_implementation:execute]
InferenceEngine::StatusCode MultinomialImpl::execute(std::vector<InferenceEngine::Blob::Ptr> &inputs,
                                                    std::vector<InferenceEngine::Blob::Ptr> &outputs,
                                                    InferenceEngine::ResponseDesc *resp) noexcept {
    const float* inpData  = inputs[0]->cbuffer().as<float*>();

    int* outData = outputs[0]->buffer().as<int*>();

    std::vector<size_t> inpDims = inputs[0]->getTensorDesc().getDims();
    std::vector<size_t> outDims = outputs[0]->getTensorDesc().getDims();

    const int batch     = inpDims[0]; //batch_size
    const int num_classes  = inpDims[1]; 
    //const int sampel_size = outDims[1];
    // navie test, pick the largest prob for each sample in batch 
    for (int b=0; b<batch; b++){
        int row_bias = b*num_classes;
        float max_prob = -100.0;
        int argmax_res = -1;
        for (int k=0; k<num_classes; k++){
            float prob = inpData[row_bias+k];
            printf("===> batch %d, class %d, prob %f", b, k, prob);
            if (prob > max_prob){ 
                max_prob = prob;
                argmax_res = k;
            }
        } 
        outData[b] = argmax_res; 
        printf("===> batch %d, argmax class %d", b, outData[b]);
    }

    return InferenceEngine::OK;
}
//! [cpu_implementation:execute]
