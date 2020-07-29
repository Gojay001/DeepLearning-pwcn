# DeepLearning-pwcn
There are paper with code and note in terms of deep learning.

- [Classification](#Image-Classification)
    - LeNet-5
    - AlexNet
    - NIN(Network In Network)
    - VGG
    - GoogLeNet(Inception-v1)
    - ResNet
    - Inception-v4
    - DenseNet
    - ShuffleNet
    - MobileNetV3
- [Detection](#Object-Detection)
    - One-stage
        - SSD
        - YOLO
        - YOLOv2
        - YOLOv3
        - YOLOv4
    - Two-stage
        - R-CNN
        - Fast R-CNN
        - Faster R-CNN
        - FPN
        - Mask R-CNN
- [Detection-3D](#3D-Object-Detection)
    - PV-RCNN
- [Tracking](#Object-Tracking)
    - MOT
        - SORT
        - DeepSORT
        - Tracktor
        - Flow-Fuse Tracker
        - JRMOT
        - Tracklet
        - FairMOT
    - VOT
        - DepthTrack
        - BinocularTrack
        - SiamRPN++
        - SiamMask
        - GlobalTrack
        - PAMCC-AOT
        - TSDM
- [FSS](#Few-Shot-Segmentation)
    - OSLSM
    - CENet(Combinatorial Embedding Network)
    - PANet(Prototype Alignment)
    - PGNet(Pyramid Graph Network)
    - AMP(Adaptive Masked Proxies)
    - CRNet(Cross-Reference Network)
    - FGN(Fully Guided Network)
    - DoG-BConvLSTM
    - SG-One
    - LTM(Local Transformation Module)
- [FSL](#Few-Shot-Learning)
    - RN(Relation Network)
- [GAN](#Generative-Adversarial-Network)
    - BeautyGAN
- [Image Generation](#Image-Generation)
    - ImageTransformer
- [Survey](#Survey)
    - 3D-Detection-Survey-2019
    - FSL-Survey-2019
    - MOT-Survey-2020

## Paper

## Image Classification
| Title | Paper | Conf | Code |
|:--------|:--------:|:--------:|:--------:|
| LeNet-5 | [Gradient-based learning applied to document recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf) | IEEE(1998) | [code]
| AlexNet | [ImageNet Classification with Deep Convolutional Neural Networks](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) | NIPS(2012) | [code]
| [NIN](https://gojay.top/2019/08/31/NIN-Network-In-Network/) | [Network In Network](https://arxiv.org/abs/1312.4400) | arXiv(2013) | [PyTorch](https://github.com/Gojay001/DeepLearning-pwcn/tree/master/Classification/NIN/Code)
| VGG | [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556) | ICLR(2015) | [code]
| [GoogLeNet](https://gojay.top/2019/09/05/GoogLeNet/) | [Going deeper with convolutions](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Szegedy_Going_Deeper_With_2015_CVPR_paper.pdf) | CVPR(2015) | [PyTorch](https://github.com/Gojay001/DeepLearning-pwcn/tree/master/Classification/GoogLeNet/Code)
| [ResNet](https://gojay.top/2019/09/08/ResNet/) | [Deep Residual Learning for Image Recognition](http://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf) | CVPR(2016) | [PyTorch](https://github.com/Gojay001/DeepLearning-pwcn/tree/master/Classification/ResNet/Code)
| Inception-v4 | [Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](https://www.aaai.org/ocs/index.php/AAAI/AAAI17/paper/viewFile/14806/14311) | AAAI(2017) | [code]
| DenseNet | [Densely Connected Convolutional Networks](http://openaccess.thecvf.com/content_cvpr_2017/papers/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.pdf) | CVPR(2017) | [code]
| ShuffleNet | [ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices](http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_ShuffleNet_An_Extremely_CVPR_2018_paper.pdf) | CVPR(2018) | [code]
| MobileNetV3 | [Searching for MobileNetV3](http://openaccess.thecvf.com/content_ICCV_2019/papers/Howard_Searching_for_MobileNetV3_ICCV_2019_paper.pdf) | ICCV(2019) | [code]
> More information can be found in [Awesome - Image Classification](https://github.com/weiaicunzai/awesome-image-classification).

## Object Detection
| Title | Paper | Conf | Code |
|:--------|:--------:|:--------:|:--------:|
| R-CNN | [Rich Feature Hierarchies for Accurate Object Detection and Semantic Segmentation](http://openaccess.thecvf.com/content_cvpr_2014/papers/Girshick_Rich_Feature_Hierarchies_2014_CVPR_paper.pdf) | CVPR(2014) | [code]
| Fast R-CNN | [Fast R-CNN](http://openaccess.thecvf.com/content_iccv_2015/papers/Girshick_Fast_R-CNN_ICCV_2015_paper.pdf) | ICCV(2015) | [code]
| [Faster R-CNN](https://gojay.top/2019/10/19/Faster-R-CNN/) | [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497) | NIPS(2015) | [PyTorch](https://github.com/Gojay001/faster-rcnn.pytorch)
| SSD | [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325) | ECCV(2016) | [Caffe](https://github.com/weiliu89/caffe/tree/ssd)
| YOLO | [You Only Look Once: Unified, Real-Time Object Detection](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Redmon_You_Only_Look_CVPR_2016_paper.pdf) | CVPR(2016) | [code]
| YOLOv2 | [YOLO9000: Better, Faster, Stronger](http://openaccess.thecvf.com/content_cvpr_2017/papers/Redmon_YOLO9000_Better_Faster_CVPR_2017_paper.pdf) | CVPR(2017) | [code]
| FPN | [Feature Pyramid Networks for Object Detection](http://openaccess.thecvf.com/content_cvpr_2017/papers/Lin_Feature_Pyramid_Networks_CVPR_2017_paper.pdf) | CVPR(2017) | [code]
| Mask R-CNN | [Mask R-CNN](http://openaccess.thecvf.com/content_ICCV_2017/papers/He_Mask_R-CNN_ICCV_2017_paper.pdf) | ICCV(2017) | [PyTorch](https://github.com/facebookresearch/detectron2)
| YOLOv3 | [YOLOv3: An Incremental Improvement](https://arxiv.org/abs/1804.02767) | arXiv(2018) | [Offical](https://github.com/pjreddie/darknet)
| YOLOv4 | [YOLOv4: Optimal Speed and Accuracy of Object Detection](https://arxiv.org/abs/2004.10934) | arXiv(2020) | [Offical](https://github.com/AlexeyAB/darknet)
> More information can be found in [awesome-object-detection](https://github.com/amusi/awesome-object-detection).

## 3D Object Detection
| Title | Paper | Conf | Code |
|:--------|:--------:|:--------:|:--------:|
| [PV-RCNN](https://gojay.top/2020/06/23/PV-RCNN/) | [PV-RCNN: Point-Voxel Feature Set Abstraction for 3D Object Detection](http://openaccess.thecvf.com/content_CVPR_2020/papers/Shi_PV-RCNN_Point-Voxel_Feature_Set_Abstraction_for_3D_Object_Detection_CVPR_2020_paper.pdf) | CVPR(2020) | [PyTorch](https://github.com/sshaoshuai/PV-RCNN)

## Object Tracking
| Title | Paper | Conf | Code |
|:--------|:--------:|:--------:|:--------:|
| [SORT](https://gojay.top/2020/06/14/SORT/) | [Simple Online and Realtime Tracking](https://arxiv.org/abs/1602.00763) | ICIP(2016) | [PyTorch](https://github.com/abewley/sort)
| DepthTrack | [Real-time depth-based tracking using a binocular camera](https://github.com/Gojay001/DeepLearning-pwcn/tree/master/Tracking/Binocular%20camera/DepthTrack.pdf) | WCICA(2016) | [code]
| [DeepSORT](https://gojay.top/2020/06/20/DeepSORT/) | [Simple Online and Realtime Tracking with a Deep Association Metric](https://arxiv.org/abs/1703.07402) | ICIP(2017) | [PyTorch](https://github.com/nwojke/deep_sort)
| BinocularTrack | [Research on Target Tracking Algorithm Based on Parallel Binocular Camera](https://github.com/Gojay001/DeepLearning-pwcn/blob/master/Tracking/Binocular%20camera/BinocularTrack.pdf) | ITAIC(2019) | [code]
| [SiamRPN++](https://gojay.top/2020/05/09/SiamRPN++/) | [SiamRPN++: Evolution of Siamese Visual Tracking with Very Deep Networks](http://openaccess.thecvf.com/content_CVPR_2019/papers/Li_SiamRPN_Evolution_of_Siamese_Visual_Tracking_With_Very_Deep_Networks_CVPR_2019_paper.pdf) | CVPR(2019) | [PyTorch](https://github.com/STVIR/pysot)
| [SiamMask](https://gojay.top/2019/11/26/SiamMask/) | [Fast Online Object Tracking and Segmentation: A Unifying Approach](http://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Fast_Online_Object_Tracking_and_Segmentation_A_Unifying_Approach_CVPR_2019_paper.pdf) | CVPR(2019) | [PyTorch](https://github.com/Gojay001/SiamMask)
| [Tracktor](https://gojay.top/2019/11/09/Tracktor/) | [Tracking without bells and whistles](https://arxiv.org/abs/1903.05625) | ICCV(2019) | [PyTorch](https://github.com/Gojay001/tracking_wo_bnw)
| [GlobalTrack](https://gojay.top/2020/01/04/GlobalTrack/) | [GlobalTrack: A Simple and Strong Baseline for Long-term Tracking](https://arxiv.org/abs/1912.08531) | AAAI(2020) | [PyTorch](https://github.com/huanglianghua/GlobalTrack)
| [PAMCC-AOT](https://gojay.top/2020/02/25/PAMCC-AOT/) | [Pose-Assisted Multi-Camera Collaboration for Active Object Tracking](https://arxiv.org/abs/2001.05161) | AAAI(2020) | [code]
| [FFT](https://gojay.top/2020/03/05/FFT-Flow-Fuse-Tracker/) | [Multiple Object Tracking by Flowing and Fusing](https://arxiv.org/abs/2001.11180) | arXiv(2020) | [code]
| [JRMOT](https://gojay.top/2020/02/28/JRMOT/) | [JRMOT: A Real-Time 3D Multi-Object Tracker and a New Large-Scale Dataset](https://arxiv.org/abs/2002.08397) | arXiv(2020) | [code]
| [Tracklet](https://gojay.top/2020/03/26/Tracklet/) | [Multi-object Tracking via End-to-end Tracklet Searching and Ranking](https://arxiv.org/abs/2003.02795) | arXiv(2020) | [code]
| [TSDM](https://gojay.top/2020/05/23/TSDM/) | [TSDM: Tracking by SiamRPN++ with a Depth-refiner and a Mask-generator](https://arxiv.org/abs/2005.04063) | arXiv(2020) | [PyTorch](https://github.com/Gojay001/TSDM)
| [FairMOT](https://gojay.top/2020/05/25/FairMOT/) | [A Simple Baseline for Multi-Object Tracking](https://arxiv.org/abs/2004.01888) | arXiv(2020) | [PyTorch](https://github.com/Gojay001/FairMOT)

## Few-Shot Segmentation
| Title | Paper | Conf | Code |
|:--------|:--------:|:--------:|:--------:|
| OSLSM | [One-Shot Learning for Semantic Segmentation](https://arxiv.org/abs/1709.03410) | arXiv(2017) | [Caffe](https://github.com/lzzcd001/OSLSM)
| CENet | [Learning Combinatorial Embedding Networks for Deep Graph Matching](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_Learning_Combinatorial_Embedding_Networks_for_Deep_Graph_Matching_ICCV_2019_paper.pdf) | ICCV(2019) | [Pytorch](https://github.com/Thinklab-SJTU/PCA-GM)
| PANet | [PANet: Few-Shot Image Semantic Segmentation with Prototype Alignment](http://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_PANet_Few-Shot_Image_Semantic_Segmentation_With_Prototype_Alignment_ICCV_2019_paper.pdf) | ICCV(2019) | [PyTorch](https://github.com/kaixin96/PANet)
| [PGNet](https://gojay.top/2020/07/28/PGNet/) | [Pyramid Graph Networks with Connection Attentions for Region-Based One-Shot Semantic Segmentation](https://openaccess.thecvf.com/content_ICCV_2019/papers/Zhang_Pyramid_Graph_Networks_With_Connection_Attentions_for_Region-Based_One-Shot_Semantic_ICCV_2019_paper.pdf) | ICCV(2019) | [code]
| AMP | [AMP: Adaptive Masked Proxies for Few-Shot Segmentation](https://openaccess.thecvf.com/content_ICCV_2019/papers/Siam_AMP_Adaptive_Masked_Proxies_for_Few-Shot_Segmentation_ICCV_2019_paper.pdf) | ICCV(2019) | [Pytorch](https://github.com/MSiam/AdaptiveMaskedProxies)
| [CRNet](https://gojay.top/2020/07/10/CRNet/) | [CRNet: Cross-Reference Networks for Few-Shot Segmentation](https://openaccess.thecvf.com/content_CVPR_2020/papers/Liu_CRNet_Cross-Reference_Networks_for_Few-Shot_Segmentation_CVPR_2020_paper.pdf) | CVPR(2020) | [code]
| FGN | [FGN: Fully Guided Network for Few-Shot Instance Segmentation](https://openaccess.thecvf.com/content_CVPR_2020/papers/Fan_FGN_Fully_Guided_Network_for_Few-Shot_Instance_Segmentation_CVPR_2020_paper.pdf) | CVPR(2020) | [code]
| DoG-BConvLSTM | [On the Texture Bias for Few-Shot CNN Segmentation](https://arxiv.org/abs/2003.04052) | arXiv(2020) | [TensorFlow](https://github.com/rezazad68/fewshot-segmentation)
| SG-One | [SG-One: Similarity Guidance Network for One-Shot Semantic Segmentation](https://arxiv.org/abs/1810.09091) | ITC(2020) | [PyTorch](https://github.com/xiaomengyc/SG-One)
| LTM | [A New Local Transformation Module for Few-Shot Segmentation](https://arxiv.org/abs/1910.05886) | ICMM(2020) | [code]
> More information can be found in [Few-Shot-Semantic-Segmentation-Papers](https://github.com/xiaomengyc/Few-Shot-Semantic-Segmentation-Papers).

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

## Survey
| Title | Paper | Conf |
|:--------|:--------:|:--------:|
| 3D-Detection-Survey-2019 | [A Survey on 3D Object Detection Methods for Autonomous Driving Applications](http://wrap.warwick.ac.uk/114314/1/WRAP-survey-3D-object-detection-methods-autonomous-driving-applications-Arnold-2019.pdf) | ITS(2019)
| [FSL-Survey-2019](https://gojay.top/2020/07/07/FSL-Survey-2019/) | [Generalizing from a Few Examples: A Survey on Few-Shot Learning](https://arxiv.org/abs/1904.05046) | CSUR(2019)
| MOT-Survey-2020 | [Deep Learning in Video Multi-Object Tracking: A Survey](https://arxiv.org/abs/1907.12740) | Neurocomputing(2020)