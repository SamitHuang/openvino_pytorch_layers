// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "cpu_kernel.hpp"
#include "op.hpp"
#include <details/ie_exception.hpp>
#include <ie_layouts.h>
#include "ie_parallel.hpp"
//#include <opencv2/opencv.hpp>
#include <math.h> //TODO: use MKL math lib instead

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
    //printf("=> Debug MulImpl inpShape");
}
//! [cpu_implementation:ctor]

//! [cpu_implementation:getSupportedConfigurations]
InferenceEngine::StatusCode MultinomialImpl::getSupportedConfigurations(std::vector<InferenceEngine::LayerConfig> &conf,
                                                                         InferenceEngine::ResponseDesc *resp) noexcept {
     std::vector<InferenceEngine::DataConfig> inDataConfig;
     std::vector<InferenceEngine::DataConfig> outDataConfig;
     InferenceEngine::SizeVector order = {0, 1}; //TODO: what is it? check correctness. ----------------------------> <---------------------
     
     //printf("=> Debug getSupportedConfig 0\r\n");
     // Allow any offset before data
     size_t offset((std::numeric_limits<size_t>::max)());
     //printf("=> Debug inpShape %d, %d\r\n", inpShape[0], inpShape[1]);
     // Input shape TODO: change a.. data type and shape, order 
     InferenceEngine::DataConfig inpConf;
     //inpConf.desc = InferenceEngine::TensorDesc(InferenceEngine::Precision::FP32, inpShape, {inpShape, order, offset});
    
     inpConf.desc = InferenceEngine::TensorDesc(InferenceEngine::Precision::FP32, inpShape, InferenceEngine::Layout::NC);
     inDataConfig.push_back(inpConf);

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
    
     //printf("=> getSupportedConfig 2\r\n");

     return InferenceEngine::StatusCode::OK;
}
//! [cpu_implementation:getSupportedConfigurations]

//! [cpu_implementation:init]
InferenceEngine::StatusCode MultinomialImpl::init(InferenceEngine::LayerConfig &config, InferenceEngine::ResponseDesc *resp) noexcept {
    //printf("start init\r\n");
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

    //printf("=> Debug init done \r\n");
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

    //TODO: no need to creat the rng in every inference. create it globally when the network loaded. 
    //The seed for rng should fixed once c++ build. claim it globally should be fine. 
    //cv::RNG rng;   

    for (int b=0; b<batch; b++){
        int offset = b*num_classes;  
        float maxi = -std::numeric_limits<float>::infinity(); 
        for (int k=0; k<num_classes; k++){
            //float prob = inpData[row_bias+k];
            double u = rng.uniform((double)0, (double)1);
            // TODO: use mkl math log
            float sum_g_a = inpData[offset+k] + (-log(-log(u))); //log is natural. get gumbel dist. samples from uniform. and add with input
            if (sum_g_a > maxi){
                maxi = sum_g_a;
                outData[b] = k;
            }
        } 
    }
    return InferenceEngine::OK;
}
//! [cpu_implementation:execute]
