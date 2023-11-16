 
**Score-Based Generative Models for Medical Image Segmentation using Signed Distance Functions**<br>
GCPR 2023<br>
Lea Bogensperger, Dominik Narnhofer, Filip Ilic, Thomas Pock<br>

---

[[Project Page]](https://github.com/leabogensperger/generative-segmentation-sdf)
[[Paper]](https://arxiv.org/abs/2303.05966)


Environment Setup:
```bash
git clone --recurse-submodules git@github.com:leabogensperger/generative-segmentation-sdf.git
conda env create -f env.yaml
conda activate generative_segmentation_sdf
```

# Score-Based Generative Models for Medical Image Segmentation using Signed Distance Functions

This repository contains the code to train a generative model that learns the conditional distribution of implicit segmentation masks in the form of signed distance function conditioned on a specific input image. The generative model is set up as a score-based diffusion model with a variance-exploding scheme -- however, later experiments have shown that the variance-preserving scheme seems numerically a bit more stable for this case, therefore this option is now also included (set the param *sde* in *SMLD* of the config file to either *ve*/*vp*).

<img src="assets/process_sde.png" alt="drawing" width="420"/>

# Instructions

1) Run by specifying a config file:
```python 
python main.py --config "cfg/monuseg.yaml"
```

2) Sample (set experiment folder in config file):
```python 
python sample.py --config "cfg/monuseg.yaml"
```

Note: the pre-processed data sets will be uploaded later. The data set is specified by the config file. The root directory is set with <data_path> in the config file, which must contain csv files for train and test mode with columns *filename* and *maskname* of all pre-processed patches. Moreover, it must contain the folders *Trainig_patches* and *Test_patches*, which include for each patch a .png file of the input image and a .npy file of the sdf transformed segmentation mask.

# Sampling

The sampling process of the proposed approach is shown using the predictor-corrector sampling algorithm (see Algorithm 1 in the paper). 
In the top row there are four different condition images and the center row contains the generated/predicted SDF masks. 
Further, the bottom row displays the corresponding binary masks, which are obtained only indirectly from thresholding the predicted SDF masks. 

<img src="assets/sampling_sdf.gif">

# Cite

```bibtex
@misc{
    bogensperger2023scorebased,
    title={Score-Based Generative Models for Medical Image Segmentation using Signed Distance Functions}, 
    author={Lea Bogensperger and Dominik Narnhofer and Filip Ilic and Thomas Pock},
    year={2023},
    eprint={2303.05966},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
