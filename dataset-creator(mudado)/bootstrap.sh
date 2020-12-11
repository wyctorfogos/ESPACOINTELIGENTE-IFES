#!/bin/bash
	
# Get super user privileges
if [[ $EUID != 0 ]]; then
  export wasnt_root=true
  sudo -E "$0" "$@"
fi

if [[ $EUID == 0 ]]; then

  apt-get purge -y python-opencv python3-opencv

  packages=(git build-essential wget pkg-config unzip ca-certificates sudo \
    libjpeg8-dev libtiff5-dev libjasper-dev libpng12-dev \
    libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
    ffmpeg x264 libx264-dev \
    libgtk2.0-dev libatlas-base-dev gfortran \
    python3 python3-dev)
    echo "[$EUID] |>>| installing distro packages: ${packages[*]}"
    apt update
    apt install --no-install-recommends -y ${packages[*]} 

  invalid_cmake_version=false
  if command -v cmake > /dev/null ; then 
    cmake_version=`cmake --version | grep -o -E "([0-9]{1,}\.)+[0-9]{1,}"`
    cmake_version=(`echo $cmake_version | tr . ' '`)
    if [ ${cmake_version[1]} -lt 10 ]; then
      echo "[$EUID] |>>| cmake 3.10+ required"
      invalid_cmake_version=true
    fi
  fi

  if [[ -z `command -v cmake` ]] || [[ $invalid_cmake_version == true ]]; then
    echo "[$EUID] |>>| installing cmake..."; 
    wget https://cmake.org/files/v3.11/cmake-3.11.1-Linux-x86_64.sh
    mkdir -p /opt/cmake
    sh cmake-3.11.1-Linux-x86_64.sh --skip-license --prefix=/opt/cmake
    ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake
    rm cmake-3.11.1-Linux-x86_64.sh
  fi

  if ! command -v ninja > /dev/null; then 
    echo "[$EUID] |>>| installing ninja..."; 
    wget https://github.com/ninja-build/ninja/releases/download/v1.8.2/ninja-linux.zip
    unzip ninja-linux.zip
    rm ninja-linux.zip
    mv ninja /usr/bin
  fi

  if ! command -v pip3 > /dev/null; then 
    wget https://bootstrap.pypa.io/get-pip.py
    python3 get-pip.py
    rm -f get-pip.py
  fi
fi

if [[ $EUID != 0 || -z ${wasnt_root} ]]; then
  pip3 uninstall --yes opencv-python
  pip3 install --user numpy
  wget https://github.com/opencv/opencv/archive/3.4.3.tar.gz
  tar xf 3.4.3.tar.gz
  rm -f 3.4.3.tar.gz
  cd opencv-3.4.3/
  mkdir -p build
  cd build
  cmake \
    -D CMAKE_BUILD_TYPE=RELEASE \
    -D WITH_QT=OFF \
    -D WITH_GTK=ON \
  	-D CMAKE_INSTALL_PREFIX=/usr/local \
  	-G Ninja \
    -D CMAKE_MAKE_PROGRAM=/usr/bin/ninja ..
  ninja
  sudo ninja install
  cd ../../
  rm -rf opencv-3.4.3/
fi