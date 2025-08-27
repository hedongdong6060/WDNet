WDNet: A Novel Wavelet-guided Hierarchical Diffusion Network for Multi-Target Segmentation in Colonoscopy Images
Abstract. Semantic segmentation in colonoscopy images is pivotal in
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
