# Elementary operations on Coral Edge TPU

This repository allows user to compile and run elementary (TensorFlow) operations, such as Convolution 2D and DepthwiseConvolution2D, on Coral Edge TPU.

## Requirements

### Create model
* Python 3.5–3.8
* Tensorflow 2 (2.4.1 or higher)
    ```
    pip3 install --upgrade pip
    pip3 install "tensorflow>=2.1"
    ```
* Flatbuffers
    
    **Mac**
    ```
    brew install flatbuffers
    ```
    **Linux**
    ```
    sudo apt-add-repository ppa:hnakamur/flatbuffers
    sudo apt update
    sudo apt install -y flatbuffers-compiler
    ```

* EdgeTPU Compiler

    **See README inside folder elemntary-ops** 
### Run model

For running the created models, it is enough to follow the step in [Coral AI - Get started with the USB Accelerator](https://coral.ai/docs/accelerator/get-started/#1-install-the-edge-tpu-runtime). In the end, you will have installed:

* Edge TPU runtime
* PyCoral library

## Failed to load delegate from libedgetpu.so.1

**Check whether Coral USB is connected**
```
lsusb

# Expected:
Bus 001 Device 005: ID 18d1:9302 Google Inc.
# Or
Bus 001 Device XXX: ID 1a6e:089a Global ...
```

**udev rules**

Add the following to */etc/udev/rules.d/99-edgetpu-accelerator.rules* file:
```
SUBSYSTEM=="usb",ATTRS{idVendor}=="1a6e",MODE="0666",GROUP="plugdev"
SUBSYSTEM=="usb",ATTRS{idVendor}=="089a",MODE="0666",GROUP="plugdev"
```

Restart `udev` service:
```
sudo service udev restart
sudo udevadm control --reload-rules
```

**Add current user to `plugdev` group**
```
sudo usermod -aG plugdev $USER
```

Restart the system:
```
sudo reboot
```
