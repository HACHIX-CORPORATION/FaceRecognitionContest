# 概要
1. pythonのface_recognitionの性能を評価するものになります。

# Install libraries
1. Update repo: 
   - sudo apt-get update
1. Install cmake if it's not installed
   - sudo apt-get install build-essential cmake
1. Install and compile dlib
   - Clone the dlib library in some directory of the system
      - git clone https://github.com/davisking/dlib.git
   - get into the cloned directory
      - cd dlib
   - create build directory inside the cloned directory
      - mkdir build
   - Switch to the created directory
      - cd build
   - generate a Makefile in the current directory
      - cmake ..
   - Build dlib !
   - cmake --build .
   - cd ..
   - python3 setup.py install
1. Install and compile dlib

# conda 環境
1. conda create -n OmniFace python=3.6
2. pip install Pillow
3. pip install face_recognition

# 実行
1. cd face_recognition_lib
2. python face_recognition_lib.py