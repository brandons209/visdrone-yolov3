#VisDrone DET Objects in Images with YoloV3    
[LICENSE](LICENSE)

A research project I was assigned by UCF's [CRCV](http://crcv.ucf.edu/) (Center for Research in Computer Vision). The goal is to detect objects in images taken from a UAV. Objects are usually small and there are many objects in one image. Objects can be cars, pedestrians, vans, trucks, tricycles, motorcycles, awning-tricycles, and a few others. Check out the VisDrone challenge [here](http://www.aiskyeye.com/).

### Network
I used a yoloV3 pytorch implementation based off of code from [this](https://github.com/eriklindernoren/PyTorch-YOLOv3) repo here. The network transform the input images to 608x608 for detections.

### TODO
mAP calculations return 0 for all classes, from issues on the aforementioned repo this could be how the weights are saved, but I have not been able to pin down the issue. Also, there might be some issues with a few of the labels in the dataset, but I have not found any yet, as I get index errors when training I have to catch.
