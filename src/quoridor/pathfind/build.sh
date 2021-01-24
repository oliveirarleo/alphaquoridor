rm -rf build
mkdir build
cd build
#cmake ..
cmake -DCMAKE_PREFIX_PATH=/usr/local/lib/python3.6/dist-packages/ ..
make
cd ..