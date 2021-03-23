source /opt/intel/openvino_2021/bin/setupvars.sh
source /opt/intel/oneapi/setvars.sh

cd build
make -j$(nproc --all)

model_path=/bigdata/work/tnn/openvino_test/random_tfmodel_onnx2ir/random_tfmodel.xml
inference_code_path=/opt/intel/openvino_2021/deployment_tools/inference_engine/samples/python/classification_sample_async
custom_ops_path=/opt/intel/openvino_2021/deployment_tools/custom_ops/user_ie_extensions/build/libuser_cpu_extension.so

python $inference_code_path/multi_inputs_mlp_demo.py -m $model_path -d CPU -i . \
    -l $custom_ops_path 
