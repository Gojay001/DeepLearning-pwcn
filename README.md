# DeepLearning-pwcn
There are paper with code and note in terms of deep learning.
```
|- Classification
    |- NIN(Network In Network)
    |- GoogLeNet(Inception v1)
    |- ResNet
|- Detection
    |- Faster R-CNN
|- Tracking
    |- DepthTrack
    |- BinocularTrack
    |- SiamMask
    |- Tracktor
    |- GlobalTrack
    |- PAMCC-AOT
    |- Flow-Fuse Tracker
    |- JRMOT
|- Few-Shot Learning
    |- RN(Relation Network)
```

## Paper

### Image Classification
| Net | Title | Conf | Code |
|:--------|:--------:|:--------:|:--------:|
| LeNet-5 | [Gradient-based learning applied to document recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf) | IEEE(1998) | [code]()
| NIN | [Network In Network](https://arxiv.org/pdf/1312.4400.pdf) | arXiv(2013) | [code](https://github.com/Gojay001/DeepLearning-pwcn/tree/master/Classification/NIN/Code)
| GoogLeNet | [Going deeper with convolutions](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Szegedy_Going_Deeper_With_2015_CVPR_paper.pdf) | CVPR(2015) | [code](https://github.com/Gojay001/DeepLearning-pwcn/tree/master/Classification/GoogLeNet/Code)
| ResNet | [Deep Residual Learning for Image Recognition](http://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf) | CVPR(2016) | [code](https://github.com/Gojay001/DeepLearning-pwcn/tree/master/Classification/ResNet/Code)

### Object Detection
| Net | Title | Conf | Code |
|:--------|:--------:|:--------:|:--------:|
| RCNN | [Rich Feature Hierarchies for Accurate Object Detection and Semantic Segmentation](http://openaccess.thecvf.com/content_cvpr_2014/papers/Girshick_Rich_Feature_Hierarchies_2014_CVPR_paper.pdf) | CVPR(2014) | [code]()
| Faster R-CNN | [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/pdf/1506.01497.pdf) | NIPS(2015) | [code](https://github.com/Gojay001/faster-rcnn.pytorch)

### Object Tracking
| Net | Title | Conf | Code |
|:--------|:--------:|:--------:|:--------:|
| DepthTrack | [Real-time depth-based tracking using a binocular camera](https://github.com/Gojay001/DeepLearning-pwcn/tree/master/Tracking/Binocular%20camera/DepthTrack.pdf) | WCICA(2016) | [code]
| BinocularTrack | [Research on Target Tracking Algorithm Based on Parallel Binocular Camera](https://github.com/Gojay001/DeepLearning-pwcn/blob/master/Tracking/Binocular%20camera/BinocularTrack.pdf) | ITAIC(2019) | [code]
| SiamMask | [Fast Online Object Tracking and Segmentation: A Unifying Approach](http://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Fast_Online_Object_Tracking_and_Segmentation_A_Unifying_Approach_CVPR_2019_paper.pdf) | CVPR(2019) | [code](https://github.com/Gojay001/SiamMask)
| Tracktor | [Tracking without bells and whistles](https://arxiv.org/pdf/1903.05625.pdf) | ICCV(2019) | [code](https://github.com/Gojay001/tracking_wo_bnw)
| GlobalTrack | [GlobalTrack: A Simple and Strong Baseline for Long-term Tracking](https://arxiv.org/pdf/1912.08531.pdf) | AAAI(2020) | [code](https://github.com/huanglianghua/GlobalTrack)
| PAMCC-AOT | [Pose-Assisted Multi-Camera Collaboration for Active Object Tracking](https://arxiv.org/pdf/2001.05161.pdf) | AAAI(2020) | [code]
| FFT | [Multiple Object Tracking by Flowing and Fusing](https://arxiv.org/abs/2001.11180) | arXiv(2020) | [code]
| JRMOT | [JRMOT: A Real-Time 3D Multi-Object Tracker and a New Large-Scale Dataset](https://arxiv.org/pdf/2002.08397.pdf) | arXiv(2020) | [code]

### Few-Shot Learning
| Net | Title | Conf | Code |
|:--------|:--------:|:--------:|:--------:|
| RN | [Learning to Compare: Relation Network for Few-Shot Learning](https://arxiv.org/abs/1711.06025) | CVPR(2018) | [code](https://github.com/Gojay001/LearningToCompare_FSL)