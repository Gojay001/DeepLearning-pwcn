# DeepLearning-pwcn [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
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
    - DLA(Deep Layer Aggregation)
    - ShuffleNet
    - MobileNetV3
- [Detection](#Object-Detection)
    - One-stage
        - SSD
        - YOLO
        - YOLOv2
        - RetinaNet
        - YOLOv3
        - CornerNet
        - CenterNet
        - YOLOv4
        - YOLOF
    - Two-stage
        - R-CNN
        - SPP
        - Fast R-CNN
        - Faster R-CNN
        - FPN
- [Segmentation](#Object-Segmentation)
    - FCN
    - U-Net
    - Seg-Net
    - DeepLab V1
    - PSPNet
    - DeepLab V2
    - Mask R-CNN
    - DeepLab V3
    - DeepLab V3+
    - DGCNet
- [Tracking](#Object-Tracking)
    - MOT
        - SORT
        - DeepSORT
        - Tracktor
        - Flow-Fuse Tracker
        - JRMOT
        - Tracklet
        - FairMOT
        - DMCT(Deep Multi-Camera Tracking)
        - CenterPoint
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
    - co-FCN
    - AMP(Adaptive Masked Proxies)
    - SG-One(Similarity Guidance)
    - CENet(Combinatorial Embedding Network)
    - PANet(Prototype Alignment)
    - CANet(Class Agnostic)
    - PGNet(Pyramid Graph Network)
    - CRNet(Cross-Reference Network)
    - FGN(Fully Guided Network)
    - OTB(On the Texture Bias)
    - LTM(Local Transformation Module)
    - SimPropNet(Similarity Propagation)
    - PPNet(Part-aware Prototype)
    - PFENet(Prior Guided Feature Enrichment Network)
    - PMMs(Prototype Mixture Models)
    - GFS-Seg(Generalized Few-Shot)
    - SCL(Self-Corss Learning)
    - ASGNet(Adaptive Superpixel-guided Network)
- [RGBD-SOT](#Salient-Object-Detection)
    - UC-Net
    - JL-DCF(Joint Learning and Densely-Cooperative Fusion)
    - SA-Gate(Separation-and-Aggregation Gate)
    - BiANet(Bilateral Attention Network)
- [Detection-3D](#3D-Object-Detection)
    - PV-RCNN
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
| DLA | [Deep Layer Aggregation](https://openaccess.thecvf.com/content_cvpr_2018/papers/Yu_Deep_Layer_Aggregation_CVPR_2018_paper.pdf) | CVPR(2018) | [PyTorch](https://github.com/ucbdrive/dla)
| ShuffleNet | [ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices](http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_ShuffleNet_An_Extremely_CVPR_2018_paper.pdf) | CVPR(2018) | [code]
| MobileNetV3 | [Searching for MobileNetV3](http://openaccess.thecvf.com/content_ICCV_2019/papers/Howard_Searching_for_MobileNetV3_ICCV_2019_paper.pdf) | ICCV(2019) | [code]
> More information can be found in [Awesome - Image Classification](https://github.com/weiaicunzai/awesome-image-classification).

## Object Detection
| Title | Paper | Conf | Code |
|:--------|:--------:|:--------:|:--------:|
| R-CNN | [Rich Feature Hierarchies for Accurate Object Detection and Semantic Segmentation](http://openaccess.thecvf.com/content_cvpr_2014/papers/Girshick_Rich_Feature_Hierarchies_2014_CVPR_paper.pdf) | CVPR(2014) | [code]
| SPP | [Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition](https://link.springer.com/content/pdf/10.1007/978-3-319-10578-9_23.pdf) | TPAMI(2015) | [code]
| Fast R-CNN | [Fast R-CNN](http://openaccess.thecvf.com/content_iccv_2015/papers/Girshick_Fast_R-CNN_ICCV_2015_paper.pdf) | ICCV(2015) | [code]
| [Faster R-CNN](https://gojay.top/2019/10/19/Faster-R-CNN/) | [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497) | NIPS(2015) | [PyTorch](https://github.com/Gojay001/faster-rcnn.pytorch)
| SSD | [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325) | ECCV(2016) | [Caffe](https://github.com/weiliu89/caffe/tree/ssd)
| YOLO | [You Only Look Once: Unified, Real-Time Object Detection](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Redmon_You_Only_Look_CVPR_2016_paper.pdf) | CVPR(2016) | [code]
| YOLOv2 | [YOLO9000: Better, Faster, Stronger](http://openaccess.thecvf.com/content_cvpr_2017/papers/Redmon_YOLO9000_Better_Faster_CVPR_2017_paper.pdf) | CVPR(2017) | [code]
| FPN | [Feature Pyramid Networks for Object Detection](http://openaccess.thecvf.com/content_cvpr_2017/papers/Lin_Feature_Pyramid_Networks_CVPR_2017_paper.pdf) | CVPR(2017) | [code]
| RetinaNet | [Focal Loss for Dense Object Detection](https://openaccess.thecvf.com/content_ICCV_2017/papers/Lin_Focal_Loss_for_ICCV_2017_paper.pdf) | ICCV(2017) | [code]
| YOLOv3 | [YOLOv3: An Incremental Improvement](https://arxiv.org/abs/1804.02767) | arXiv(2018) | [Offical](https://github.com/pjreddie/darknet)
| CornerNet | [CornerNet: Detecting Objects as Paired Keypoints](https://openaccess.thecvf.com/content_ECCV_2018/papers/Hei_Law_CornerNet_Detecting_Objects_ECCV_2018_paper.pdf) | ECCV(2018) | [PyTorch](https://github.com/princeton-vl/CornerNet)
| CenterNet | [Objects as Points](https://arxiv.org/abs/1904.07850) | arXiv(2019) | [PyTorch](https://github.com/xingyizhou/CenterNet)
| YOLOv4 | [YOLOv4: Optimal Speed and Accuracy of Object Detection](https://arxiv.org/abs/2004.10934) | arXiv(2020) | [Offical](https://github.com/AlexeyAB/darknet)
| YOLOF | [You Only Look One-level Feature](https://arxiv.org/pdf/2103.09460.pdf) | arXiv(2021) | [PyTorch](https://github.com/megvii-model/YOLOF)
> More information can be found in [awesome-object-detection](https://github.com/amusi/awesome-object-detection).

## Object Segmentation
| Title | Paper | Conf | Code |
|:--------|:--------:|:--------:|:--------:|
| FCN | [Fully convolutional networks for semantic segmentation](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Long_Fully_Convolutional_Networks_2015_CVPR_paper.pdf) | CVPR(2015) | [PyTorch](https://github.com/yassouali/pytorch_segmentation)
| U-Net | [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597) | MICCAI(2015) | [PyTorch](https://github.com/yassouali/pytorch_segmentation)
| Seg-Net | [SegNet: A Deep Convolutional Encoder-Decoder Architecture for Robust Semantic Pixel-Wise Labelling](https://arxiv.org/abs/1505.07293) | arXiv(2015) | [PyTorch](https://github.com/yassouali/pytorch_segmentation)
| DeepLab V1 | [Semantic Image Segmentation with Deep Convolutional Nets and Fully Connected CRFs](https://arxiv.org/abs/1412.7062) | arXiv(2014) / ICLR(2015) | [PyTorch](https://github.com/yassouali/pytorch_segmentation)
| PSPNet | [Pyramid Scene Parsing Network](https://openaccess.thecvf.com/content_cvpr_2017/papers/Zhao_Pyramid_Scene_Parsing_CVPR_2017_paper.pdf) | CVPR(2017) | [PyTorch](https://github.com/yassouali/pytorch_segmentation)
| DeepLab V2 | [DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs](https://arxiv.org/abs/1606.00915) | arXiv(2016) / TPAMI(2017) | [PyTorch](https://github.com/yassouali/pytorch_segmentation)
| [Mask R-CNN](https://gojay.top/2020/08/17/Mask-R-CNN/) | [Mask R-CNN](http://openaccess.thecvf.com/content_ICCV_2017/papers/He_Mask_R-CNN_ICCV_2017_paper.pdf) | ICCV / TPAMI(2017) | [PyTorch](https://github.com/facebookresearch/detectron2)
| DeepLab V3 | [Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1706.05587) | arXiv(2017) | [PyTorch](https://github.com/yassouali/pytorch_segmentation)
| DeepLab V3+ | [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://openaccess.thecvf.com/content_ECCV_2018/papers/Liang-Chieh_Chen_Encoder-Decoder_with_Atrous_ECCV_2018_paper.pdf) | ECCV(2018) | [PyTorch](https://github.com/yassouali/pytorch_segmentation)
| DGCNet | [Dual Graph Convolutional Network for Semantic Segmentation](https://arxiv.org/pdf/1909.06121.pdf) | BMVC(2019) | [PyTorch](https://github.com/lxtGH/GALD-DGCNet)

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
| DMCT | [Real-time 3D Deep Multi-Camera Tracking](https://arxiv.org/abs/2003.11753) | arXiv(2020) | [code]
| CenterPoint | [Center-based 3D Object Detection and Tracking](https://arxiv.org/pdf/2006.11275.pdf) | CVPR(2021) | [PyTorch](https://github.com/tianweiy/CenterPoint)

## Few-Shot Segmentation
| Title | Paper | Conf | Code |
|:--------|:--------:|:--------:|:--------:|
| [OSLSM](https://gojay.top/2020/10/19/OSLSM/) | [One-Shot Learning for Semantic Segmentation](https://arxiv.org/abs/1709.03410) | BMVC(2017) | [Caffe](https://github.com/lzzcd001/OSLSM)
| [co-FCN](https://gojay.top/2020/10/19/co-FCN/) | [Conditional Networks for Few-Shot Semantic Segmentation](https://openreview.net/pdf?id=SkMjFKJwG) | ICLR(2018) | [code]
| AMP | [AMP: Adaptive Masked Proxies for Few-Shot Segmentation](https://openaccess.thecvf.com/content_ICCV_2019/papers/Siam_AMP_Adaptive_Masked_Proxies_for_Few-Shot_Segmentation_ICCV_2019_paper.pdf) | ICCV(2019) | [Pytorch](https://github.com/MSiam/AdaptiveMaskedProxies)
| [SG-One](https://gojay.top/2020/10/20/SG-One/) | [SG-One: Similarity Guidance Network for One-Shot Semantic Segmentation](https://arxiv.org/abs/1810.09091) | arXiv(2018) / TCYB(2020) | [PyTorch](https://github.com/xiaomengyc/SG-One)
| CENet | [Learning Combinatorial Embedding Networks for Deep Graph Matching](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_Learning_Combinatorial_Embedding_Networks_for_Deep_Graph_Matching_ICCV_2019_paper.pdf) | ICCV(2019) | [Pytorch](https://github.com/Thinklab-SJTU/PCA-GM)
| PANet | [PANet: Few-Shot Image Semantic Segmentation with Prototype Alignment](http://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_PANet_Few-Shot_Image_Semantic_Segmentation_With_Prototype_Alignment_ICCV_2019_paper.pdf) | ICCV(2019) | [PyTorch](https://github.com/kaixin96/PANet)
| [CANet](https://gojay.top/2020/10/20/CANet/) | [CANet: Class-Agnostic Segmentation Networks with Iterative Refinement and Attentive Few-Shot Learning](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhang_CANet_Class-Agnostic_Segmentation_Networks_With_Iterative_Refinement_and_Attentive_Few-Shot_CVPR_2019_paper.pdf) | CVPR(2019) | [PyTorch](https://github.com/icoz69/CaNet)
| [PGNet](https://gojay.top/2020/07/28/PGNet/) | [Pyramid Graph Networks with Connection Attentions for Region-Based One-Shot Semantic Segmentation](https://openaccess.thecvf.com/content_ICCV_2019/papers/Zhang_Pyramid_Graph_Networks_With_Connection_Attentions_for_Region-Based_One-Shot_Semantic_ICCV_2019_paper.pdf) | ICCV(2019) | [code]
| [CRNet](https://gojay.top/2020/07/10/CRNet/) | [CRNet: Cross-Reference Networks for Few-Shot Segmentation](https://openaccess.thecvf.com/content_CVPR_2020/papers/Liu_CRNet_Cross-Reference_Networks_for_Few-Shot_Segmentation_CVPR_2020_paper.pdf) | CVPR(2020) | [code]
| FGN | [FGN: Fully Guided Network for Few-Shot Instance Segmentation](https://openaccess.thecvf.com/content_CVPR_2020/papers/Fan_FGN_Fully_Guided_Network_for_Few-Shot_Instance_Segmentation_CVPR_2020_paper.pdf) | CVPR(2020) | [code]
| OTB | [On the Texture Bias for Few-Shot CNN Segmentation](https://arxiv.org/abs/2003.04052) | arXiv(2020) | [TensorFlow](https://github.com/rezazad68/fewshot-segmentation)
| [LTM](https://gojay.top/2020/07/29/LTM/) | [A New Local Transformation Module for Few-Shot Segmentation](https://arxiv.org/abs/1910.05886) | MMMM(2020) | [code]
| SimPropNet | [SimPropNet: Improved Similarity Propagation for Few-shot Image Segmentation](https://arxiv.org/abs/2004.15014) | IJCAI(2020) | [code]
| [PPNet](https://gojay.top/2020/12/02/PPNet/) | [Part-aware Prototype Network for Few-shot Semantic Segmentation](https://arxiv.org/abs/2007.06309) | ECCV(2020) | [PyTorch](https://github.com/Xiangyi1996/PPNet-PyTorch)
| PFENet | [PFENet: Prior Guided Feature Enrichment Network for Few-shot Segmentation](https://arxiv.org/abs/2008.01449) | TPAMI(2020) | [PyTorch](https://github.com/Jia-Research-Lab/PFENet)
| PMMs | [Prototype Mixture Models for Few-shot Semantic Segmentation](https://arxiv.org/abs/2008.03898) | ECCV(2020) | [PyTorch](https://github.com/Yang-Bob/PMMs)
| GFS-Seg | [Generalized Few-Shot Semantic Segmentation](https://arxiv.org/abs/2010.05210) | arXiv(2020) | [code]
| SCL | [Self-Guided and Cross-Guided Learning for Few-Shot Segmentation](https://arxiv.org/pdf/2103.16129.pdf) | CVPR(2021) | [PyTorch](https://github.com/zbf1991/SCL)
| ASGNet | [Adaptive Prototype Learning and Allocation for Few-Shot Segmentation](https://arxiv.org/pdf/2104.01893.pdf) | CVPR(2021) | [PyTorch](https://github.com/Reagan1311/ASGNet)
> More information can be found in [Few-Shot-Semantic-Segmentation-Papers](https://github.com/xiaomengyc/Few-Shot-Semantic-Segmentation-Papers).

## Salient Object Detection
| Title | Paper | Conf | Code |
|:--------|:--------:|:--------:|:--------:|
| UC-Net | [UC-Net: Uncertainty Inspired RGB-D Saliency Detection via Conditional Variational Autoencoders](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhang_UC-Net_Uncertainty_Inspired_RGB-D_Saliency_Detection_via_Conditional_Variational_Autoencoders_CVPR_2020_paper.pdf) | CVPR(2020) | [PyTorch](https://github.com/JingZhang617/UCNet)
| JL-DCF | [JL-DCF: Joint Learning and Densely-Cooperative Fusion Framework for RGB-D Salient Object Detection](https://openaccess.thecvf.com/content_CVPR_2020/papers/Fu_JL-DCF_Joint_Learning_and_Densely-Cooperative_Fusion_Framework_for_RGB-D_Salient_CVPR_2020_paper.pdf) | CVPR(2020) | [PyTorch](https://github.com/jiangyao-scu/JL-DCF-pytorch)
| SA-Gate | [Bi-directional Cross-Modality Feature Propagation with Separation-and-Aggregation Gate for RGB-D Semantic Segmentation](https://arxiv.org/pdf/2007.09183.pdf) | ECCV(2020) | [Code](https://github.com/charlesCXK/RGBD_Semantic_Segmentation_PyTorch)
| BiANet | [Bilateral Attention Network for RGB-D Salient Object Detection](https://arxiv.org/pdf/2004.14582.pdf) | TIP(2021) | [Code]

## 3D Object Detection
| Title | Paper | Conf | Code |
|:--------|:--------:|:--------:|:--------:|
| [PV-RCNN](https://gojay.top/2020/06/23/PV-RCNN/) | [PV-RCNN: Point-Voxel Feature Set Abstraction for 3D Object Detection](http://openaccess.thecvf.com/content_CVPR_2020/papers/Shi_PV-RCNN_Point-Voxel_Feature_Set_Abstraction_for_3D_Object_Detection_CVPR_2020_paper.pdf) | CVPR(2020) | [PyTorch](https://github.com/sshaoshuai/PV-RCNN)

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