#WDNet: A Novel Wavelet-guided Hierarchical Diffusion Network for Multi-Target Segmentation in Colonoscopy Images

1、Abstract. Semantic segmentation in colonoscopy images is pivotal in
aiding healthcare professionals to interpret images and enhance diagnos-
tic precision. Nonetheless, the detection of polyps and instruments is
challenged by the difficulty in capturing the textures and edges of tiny
lesions, and these challenges are exacerbated by low contrast, inconsis-
tent illumination, and noise. To address these challenges, we introduce
WDNet, a network adopting a multi-tiered feature extraction and fu-
sion approach, with each encoder layer amalgamating local and global
information to construct expressive high-level representations. The input
of the network is derived from wavelet transform to dissect images into
low- and high-frequency sub-bands, utilizing learnable soft-thresholding
to diminish noise while maintaining essential features. High-frequency
data are adept at capturing details and edges, whereas low-frequency
data furnish a global context. Moreover, WDNet harnesses a diffusion-
based decoding mechanism with adaptive step sizes to amplify target
region features and mitigate background interference, achieving meticu-
lous segmentation. Comprehensive experiments conducted on a new sur-
gical dataset, along with public benchmarks underscore its remarkable
performance. WDNet not only exhibits state-of-the-art performance of
semantic segmentation in colonoscopy images with remarkable detail and
boundary accuracy but also stands out in processing speed, facilitating
the swift handling of extensive datasets.

2、MindSpore
You need to run cd mindspore first.
Environment Configuration:
Python: 3.10.0
Training Configuration:
Assigning your costumed path, like --save_model , --train_img_dir and so on in train.py.
Just enjoy it!
Testing Configuration:
After you download all the pre-trained model and testing dataset, just run test.py to generate the final prediction map: replace your trained model directory (--pth_path).
Just enjoy it!

3.Training/Testing
The training and testing experiments are conducted using PyTorch with a single NVIDIA 4080 with 24 GB Memory.

4、Downloading necessary data:


5、 Citation
   Please cite our paper if you find the work useful

6、License
The source code is free for research and education use only. Any comercial use should get formal permission first.

7、 Contact Information
For further information or assistance, please contact the dataset maintainer at 23b928029@stu.hit.edu.cn.



