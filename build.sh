spack load glog
spack load sentencepiece
rm -rf ./build
mkdir build
cd build
cmake ..
make