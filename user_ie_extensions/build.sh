source /opt/intel/openvino_2021/bin/setupvars.sh
source /opt/intel/oneapi/setvars.sh

rm build -rf
mkdir build && cd build
#compiler option icpx  for C++ and dpcpp for DPC++
cmake .. -DUSE_MKL=YES -DCMAKE_CXX_COMPILER=dpcpp 
make -j$(nproc --all)

