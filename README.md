# ActionProject
基于跟踪，骨骼的手势识别
本项目是基于tensorrt/python3实现的手势识别。
涉及跟踪模型（Yolo4 + deepsort）, 骨骼检测（trt_pose）和手势分类。
硬件采用nvidia gpu/ NX都可以。

第一个人是在中间位置，在此位置采集相应的区域来进行骨骼检测，，找最大IOU。
