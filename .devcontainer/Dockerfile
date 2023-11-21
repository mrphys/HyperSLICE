ARG TF_VERSION=2.8.0
FROM tensorflow/tensorflow:${TF_VERSION}-gpu

RUN apt-get install wget

# https://developer.nvidia.com/blog/updating-the-cuda-linux-gpg-repository-key/
RUN apt-key del 7fa2af80 && \
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb && \
dpkg -i cuda-keyring_1.0-1_all.deb && \
rm /etc/apt/sources.list.d/cuda*.list

# Create non-root user.
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME && \
    useradd --uid $USER_UID --gid $USER_GID -m $USERNAME && \
    # Add user to sudoers.
    apt-get update && \
    apt-get install -y sudo && \
    echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME && \
    chmod 0440 /etc/sudoers.d/$USERNAME && \
    # Change default shell to bash.
    usermod --shell /bin/bash $USERNAME

# Install git including LFS.
RUN apt-get update && \
    apt-get install -y git && \
    curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && \
    apt-get install -y git-lfs

RUN pip install --upgrade pip

# To enable plotting.
RUN apt-get update && \
    apt-get install -y libcairo2-dev libgirepository1.0-dev libgtk-3-dev python3-gi-cairo ffmpeg && \
    pip install pycairo PyGObject

#Install necessities for making C++ files
RUN apt update && \
    apt install g++ 

# Using devtoolset with correct manylinux2014 libraries.
ARG PREFIX=/dt9/usr
ARG CC=gcc
ARG CXX="${PREFIX}/bin/g++"
ARG LIBDIR="${PREFIX}/lib"
ARG INCLUDEDIR="${PREFIX}/include"
ARG CFLAGS="-O3 -march=x86-64 -mtune=generic -fPIC"

# Install FFTW3.
RUN cd /opt && \
    curl -sL http://www.fftw.org/fftw-3.3.9.tar.gz | tar xz && \
    cd fftw-3.3.9 && \
    ./configure CC="${CC}" CFLAGS="${CFLAGS}" --enable-openmp --enable-float && \
    make && \
    make install && \
    ./configure CC="${CC}" CFLAGS="${CFLAGS}" --enable-openmp && \
    make && \
    make install

# Install spiral waveform.
RUN cd /opt && \
    git clone https://github.com/mrphys/spiral-waveform --branch v1.0.0 && \
    cd spiral-waveform && \
    make install

ARG TFMRI_VERSION=0.18.0
# Install tensorflow utilities.
RUN pip install tensorflow-datasets tensorflow-addons tensorboard tensorflow_mri==$TFMRI_VERSION

# Install additional packages.
RUN pip install ffmpeg-python scipy scikit-learn scikit-image ipykernel google-api-python-client pydicom jupyter opencv-python gdown

#Install additional packages 2
RUN pip install keras-tuner==1.4.6