# DeepLearning-pwcn
There are paper with code and note in terms of deep learning.
```
|- Classification
    |- LeNet-5
    |- NIN(Network In Network)
    |- GoogLeNet(Inception v1)
    |- ResNet
|- Detection
    |- RCNN
    |- Faster R-CNN
|- Tracking
    |- MOT
        |- Tracktor
        |- Flow-Fuse Tracker
        |- JRMOT
        |- Tracklet
    |- VOT
        |- DepthTrack
        |- BinocularTrack
        |- SiamRPN++
        |- SiamMask
        |- GlobalTrack
        |- PAMCC-AOT
        |- TSDM
|- Few-Shot Learning
    |- RN(Relation Network)
|- GAN
    |- BeautyGAN
|- Image Generation
    |- ImageTransformer
```

## Paper

## Image Classification
| Title | Paper | Conf | Code |
|:--------|:--------:|:--------:|:--------:|
| LeNet-5 | [Gradient-based learning applied to document recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf) | IEEE(1998) | [code]
| [NIN](https://gojay.top/2019/08/31/NIN-Network-In-Network/) | [Network In Network](https://arxiv.org/pdf/1312.4400.pdf) | arXiv(2013) | [PyTorch](https://github.com/Gojay001/DeepLearning-pwcn/tree/master/Classification/NIN/Code)
| [GoogLeNet](https://gojay.top/2019/09/05/GoogLeNet/) | [Going deeper with convolutions](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Szegedy_Going_Deeper_With_2015_CVPR_paper.pdf) | CVPR(2015) | [PyTorch](https://github.com/Gojay001/DeepLearning-pwcn/tree/master/Classification/GoogLeNet/Code)
| [ResNet](https://gojay.top/2019/09/08/ResNet/) | [Deep Residual Learning for Image Recognition](http://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf) | CVPR(2016) | [PyTorch](https://github.com/Gojay001/DeepLearning-pwcn/tree/master/Classification/ResNet/Code)

## Object Detection
| Title | Paper | Conf | Code |
|:--------|:--------:|:--------:|:--------:|
| RCNN | [Rich Feature Hierarchies for Accurate Object Detection and Semantic Segmentation](http://openaccess.thecvf.com/content_cvpr_2014/papers/Girshick_Rich_Feature_Hierarchies_2014_CVPR_paper.pdf) | CVPR(2014) | [code]
| [Faster R-CNN](https://gojay.top/2019/10/19/Faster-R-CNN/) | [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/pdf/1506.01497.pdf) | NIPS(2015) | [PyTorch](https://github.com/Gojay001/faster-rcnn.pytorch)

## Object Tracking
| Title | Paper | Conf | Code |
|:--------|:--------:|:--------:|:--------:|
| DepthTrack | [Real-time depth-based tracking using a binocular camera](https://github.com/Gojay001/DeepLearning-pwcn/tree/master/Tracking/Binocular%20camera/DepthTrack.pdf) | WCICA(2016) | [code]
| BinocularTrack | [Research on Target Tracking Algorithm Based on Parallel Binocular Camera](https://github.com/Gojay001/DeepLearning-pwcn/blob/master/Tracking/Binocular%20camera/BinocularTrack.pdf) | ITAIC(2019) | [code]
| [SiamRPN++](https://gojay.top/2020/05/09/SiamRPN++/) | [SiamRPN++: Evolution of Siamese Visual Tracking with Very Deep Networks](http://openaccess.thecvf.com/content_CVPR_2019/papers/Li_SiamRPN_Evolution_of_Siamese_Visual_Tracking_With_Very_Deep_Networks_CVPR_2019_paper.pdf) | CVPR(2019) | [PyTorch](https://github.com/STVIR/pysot)
| [SiamMask](https://gojay.top/2019/11/26/SiamMask/) | [Fast Online Object Tracking and Segmentation: A Unifying Approach](http://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Fast_Online_Object_Tracking_and_Segmentation_A_Unifying_Approach_CVPR_2019_paper.pdf) | CVPR(2019) | [PyTorch](https://github.com/Gojay001/SiamMask)
| [Tracktor](https://gojay.top/2019/11/09/Tracktor/) | [Tracking without bells and whistles](https://arxiv.org/pdf/1903.05625.pdf) | ICCV(2019) | [PyTorch](https://github.com/Gojay001/tracking_wo_bnw)
| [GlobalTrack](https://gojay.top/2020/01/04/GlobalTrack/) | [GlobalTrack: A Simple and Strong Baseline for Long-term Tracking](https://arxiv.org/pdf/1912.08531.pdf) | AAAI(2020) | [PyTorch](https://github.com/huanglianghua/GlobalTrack)
| [PAMCC-AOT](https://gojay.top/2020/02/25/PAMCC-AOT/) | [Pose-Assisted Multi-Camera Collaboration for Active Object Tracking](https://arxiv.org/pdf/2001.05161.pdf) | AAAI(2020) | [code]
| [FFT](https://gojay.top/2020/03/05/FFT-Flow-Fuse-Tracker/) | [Multiple Object Tracking by Flowing and Fusing](https://arxiv.org/abs/2001.11180) | arXiv(2020) | [code]
| [JRMOT](https://gojay.top/2020/02/28/JRMOT/) | [JRMOT: A Real-Time 3D Multi-Object Tracker and a New Large-Scale Dataset](https://arxiv.org/pdf/2002.08397.pdf) | arXiv(2020) | [code]
| [Tracklet](https://gojay.top/2020/03/26/Tracklet/) | [Multi-object Tracking via End-to-end Tracklet Searching and Ranking](https://arxiv.org/abs/2003.02795) | arXiv(2020) | [code]
| [TSDM](https://gojay.top/2020/05/23/TSDM/) | [TSDM: Tracking by SiamRPN++ with a Depth-refiner and a Mask-generator](https://arxiv.org/abs/2005.04063) | arXiv(2020) | [PyTorch](https://github.com/Gojay001/TSDM)

## Few-Shot Learning
| Title | Paper | Conf | Code |
|:--------|:--------:|:--------:|:--------:|
| [RN](https://gojay.top/2019/08/21/RN-Realation-Network/) | [Learning to Compare: Relation Network for Few-Shot Learning](https://arxiv.org/abs/1711.06025) | CVPR(2018) | [PyTorch](https://github.com/Gojay001/LearningToCompare_FSL)

## Generative Adversarial Network
| Title | Paper | Conf | Code |
|:--------|:--------:|:--------:|:--------:|
| BeautyGAN | [Beautygan: Instance-level facial makeup transfer with deep generative adversarial network](http://colalab.org/media/paper/BeautyGAN-camera-ready.pdf) | ACM(2018) | [TensorFlow](http://liusi-group.com/projects/BeautyGAN)

## Image Generation
| Title | Paper | Conf | Code |
|:--------|:--------:|:--------:|:--------:|
| [ImageTransformer](https://gojay.top/2020/05/15/Image-Transformer/) | [Image Transformer](https://arxiv.org/abs/1802.05751) | arXiv(2018) | [code]