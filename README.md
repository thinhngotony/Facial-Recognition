

  <h1 align="center">Recognition and Streaming in Realtime</h1>

  <p align="center">
    Design by Tony Ngô
    <br />
    <br />
    <a href="https://drive.google.com/drive/folders/1c5vk-14No03_705vYpSwFOcN7DeLiV8V?usp=sharing"><strong>Explore full source code »</strong></a>
    <br />
    <br />
    <a href="https://drive.google.com/file/d/1Vv-2d_wdnpByLtC6wmC7UwAxZiqLqjZt/view?usp=sharing">View Demo</a>
    ·
    <a href="facebook.com/tonyngo0206">Report Bug</a>
    ·
    <a href="facebook.com/tonyngo0206">Request Feature</a>
  </p>
</p>



<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project



### Built With

* [Cuda Tookit + CudaDNN]()
* [Face Recognition]()
* [OpenCV]()
* [Flask]()



<!-- GETTING STARTED -->
## Getting Started

To run these code please do the following steps

### Prerequisites

This is an example of how to list things you need to use the software and how to install them.
* All dependences
  ```sh
  pip install cmake
  pip install flask
  pip install face_recognition
  pip install tensorflow-gpu
  pip install opencv-contrib-python
  ```

### Installation

1. Download CudaTookit and CudaDNN from NVDIA or this link
   ```sh
   https://drive.google.com/drive/folders/1AruLdIXXJFrG7BkDT2-T_clontU7em1f?usp=sharing
   ```
2. Install CudaTookit

3. Copy all of files in bin, include, lib FOLDERS from ..\cudnn-11.4-windows-x64-v8.2.2.26\cuda to corresponding FOLDERS C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.4

4. Copy below codes to terminal and run to build DLIB with CUDA
   ```sh
   git clone https://github.com/davisking/dlib.git
   cd dlib
   mkdir build
   cd build
   cmake .. -DDLIB_USE_CUDA=1 -DUSE_AVX_INSTRUCTIONS=1
   cmake --build .
   cd ..
   python setup.py install 
   ```
   
5. Check by these code
   ```sh
   python
   import dlib
   dlib.DLIB_USE_CUDA
   ```
   
   If the result returns "True" that you have successfully installed and built Dlib with CUDA. But it's print "False" you need to check above logs to find the errors, may be you copy cuDNN files not enough


<!-- USAGE EXAMPLES -->
## Usage

Exactly how to use I write it in file Readme.md of its folder, so let's check inside folders

_For more examples, please refer to the [Documentation](https://example.com)_



<!-- ROADMAP -->
## Roadmap

It contains in source code folders in above links



<!-- CONTRIBUTING -->
## Contributing

You can pull requests and email me for development 

<!-- LICENSE -->
## License

Distributed under the . See `LICENSE` for more information.



<!-- CONTACT -->
## Contact

Your Name - [@thinhngotony](https://twitter.com/thinhngotony) - email - [thinhngo.tony@gmail.com]

More project Link: [https://github.com/thinhngotony](https://github.com/thinhngotony)



