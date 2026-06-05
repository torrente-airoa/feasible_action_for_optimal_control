apt-get update
apt-get install -y libqhull-dev clang-14 cmake pybind11-dev libeigen3-dev
pip install "numpy<2"

git submodule update --init --recursive external

cd external/daqp
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make install
