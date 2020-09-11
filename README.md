
# brown_phototour_revisited
> The package for local patch descriptors evaluation, which takes into account image indexes and second nearest neighbor ratio (SNN) filtering strategy. It is in agreement with IMC benchmark and practice, unlike the original protocol.


This file will become your README and also the index of your documentation.

## Install

`pip install brown_phototour_revisited`

## How to use

There are 3 main modules of the package: dataset, extraction and benchmarking. 
To run the benchmark one needs two things:
 - extract the desccriptors with either 'extract_pytorchinput_descriptors' or 'extract_numpyinput_descriptors'
 - get the mean average precision (mAP) with 'evaluate_mAP_snn_based'
 
Here we will show how to evaluate several descriptors: Pytorch-based HardNet, OpenCV SIFT, skimage BRIEF.



```python
!pip install kornia
```

The code below will download the HardNet, trained on Liberty dataset, download the Notredame subset and extracts the local patch descriptors into the dict. Note, that we should not evaluate descriptor on the same subset, as it was trained on.

```python
import torch
import kornia

from brown_phototour_revisited.dataset import *
from brown_phototour_revisited.extraction import *
from brown_phototour_revisited.benchmarking import *

model = kornia.feature.HardNet(True).eval()

descs_out_dir = 'data/descriptors'
download_dataset_to = 'data/dataset'
patch_size = 32 # HardNet expects 32x32 patches

desc_dict = extract_pytorchinput_descriptors(model,
                                'HardNet+Liberty',
                                subset= 'notredame', 
                                path_to_save_dataset = download_dataset_to,
                                path_to_save_descriptors = descs_out_dir,
                                patch_size = patch_size, 
                                device = torch.device('cuda:0'))
```

```python
print (desc_dict.keys())
```

Function **extract_pytorchinput_descriptors** expects **torch.nn.Module**, which takes (B, 1, patch_size, patch_size) torch.Tensor input and outputs (B, desc_dim) torch.Tensor.

Now we will calculate mAP.

```python
mAP = evaluate_mAP_snn_based(desc_dict['descriptors'],
                             desc_dict['labels'], 
                             desc_dict['img_idxs'],
                             path_to_save_mAP = 'data/mAP/HardNet+Liberty_notredame.npy',
                            backend='pytorch-cuda')
print (f'HardNetLib mAP on Notredame = {mAP:.5f}')
```

Now we will evaluate OpenCV SIFT descriptor. 
Function **extract_numpyinput_descriptors** expects function or object, which takes (patch_size, patch_size) input and outputs (desc_dim) np.array.
As OpenCV doesn't provide such function, we will create it ourselves.

```python
import cv2
import numpy as np
patch_size = 65
def get_center_kp(PS=65.):
    c = PS/2.0
    center_kp = cv2.KeyPoint()
    center_kp.pt = (c,c)
    center_kp.size = 2*c/5.303
    return center_kp


sift = cv2.SIFT_create()
center_kp = get_center_kp(patch_size)

def extract_opencv_sift(patch):
    return sift.compute((255*patch).astype(np.uint8),[center_kp])[1][0].reshape(128)

descs_out_dir = 'data/descriptors'
download_dataset_to = 'data/dataset'


desc_dict_sift = extract_numpyinput_descriptors(extract_opencv_sift,
                                'OpenCV_SIFT',
                                subset= 'notredame', 
                                path_to_save_dataset = download_dataset_to,
                                path_to_save_descriptors = descs_out_dir,
                                patch_size = patch_size)
```

```python
mAP_SIFT = evaluate_mAP_snn_based(desc_dict_sift['descriptors'],
                             desc_dict_sift['labels'], 
                             desc_dict_sift['img_idxs'],
                            path_to_save_mAP = 'data/mAP/OpenCV_SIFT65_notredame.npy',
                            backend='pytorch-cuda')
print (f'OpenCV SIFT PS = {patch_size}, mAP on Notredame = {mAP_SIFT:.5f}')
```

Now, let's try some binary descriptor, like BRIEF. Evaluation so far supports two metrics: **euclidean** and **hamming**.
Function **extract_numpyinput_descriptors** expects function or object, which takes (patch_size, patch_size) input and outputs (desc_dim) np.array.
As skimage doesn't provide exactly such function, we will create it ourselves by placing "detected" keypoint in the center of the patch.

```python
import numpy as np
from skimage.feature import BRIEF
patch_size = 65
BR = BRIEF(patch_size = patch_size)
def extract_skimage_BRIEF(patch):
    BR.extract(patch.astype(np.float64), np.array([patch_size/2.0, patch_size/2.0]).reshape(1,2))
    return BR.descriptors.astype(np.float32)

desc_dict_brief = extract_numpyinput_descriptors(extract_skimage_BRIEF,
                                'skimage_BRIEF',
                                subset= 'notredame', 
                                path_to_save_dataset = download_dataset_to,
                                path_to_save_descriptors = descs_out_dir,
                                patch_size = patch_size)
```

That's will take a while. 

```python
mAP_BRIEF = evaluate_mAP_snn_based(desc_dict_brief['descriptors'].astype(np.bool),
                             desc_dict_brief['labels'], 
                             desc_dict_brief['img_idxs'],
                             path_to_save_mAP = 'data/mAP/skimageBRIEF_notredame.npy',
                             backend='numpy',
                             distance='hamming')
print (f'skimage BRIEF PS = {patch_size}, mAP on Notredame = {mAP_BRIEF:.5f}')
```

The original Brown benchmark consider evaluation, similar to cross-validation: train descriptor on one subset, evaluate on two others, repeat for all, so 6 evaluations are required. 
For the handcrafted descriptors, or those, that are trained on 3rd party datasets, only 3 evaluations are necessary. 

We have function, which does all these evaluations: **full_evaluation**, which internally calls the functions we discussed above.

```python
import torch
import kornia
from brown_phototour_revisited.benchmarking import *
patch_size = 65 # SIFT performs better with bigger patch size.

model = kornia.feature.SIFTDescriptor(patch_size, rootsift=True).eval()

descs_out_dir = 'data/descriptors'
download_dataset_to = 'data/dataset'
results_dir = 'data/mAP'
desc_dict = full_evaluation(model,
                                'Kornia RootSIFT',
                                path_to_save_dataset = download_dataset_to,
                                path_to_save_descriptors = descs_out_dir,
                                path_to_save_mAP = results_dir,
                                patch_size = patch_size, 
                                device = torch.device('cuda:0'), 
                           distance='euclidean',
                           backend='pytorch-cuda')
```

If you use the benchmark, please cite it:

    @misc{BrownRevisited2020,
      title={UBC PhotoTour Revisied},
      author={Mishkin, Dmytro},
      year={2020},
      url = {https://github.com/ducha-aiki/brown_phototour_revisited}
    }
