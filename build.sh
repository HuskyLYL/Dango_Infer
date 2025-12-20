spack load glog
spack load sentencepiece
spack load googletest
rm -rf ./build
mkdir build
cd build
cmake ..
make