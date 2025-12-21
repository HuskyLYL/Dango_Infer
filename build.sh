source /home/lyl/spack/share/spack/setup-env.sh
spack load glog
spack load sentencepiece
spack load googletest
rm -rf ./build
mkdir build
cd build
cmake ..
make