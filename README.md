
# brown_phototour_revisited
> The package for local patch descriptors evaluation, which takes into account image indexes and second nearest neighbor ratio (SNN) filtering strategy. It is in agreement with IMC benchmark and practice, unlike the original protocol. The benchmark is not "test benchmark" by amy means. Rather it is intended to be used as validation/development set for local patch descriptor learning and/or crafting.


## Install

`pip install brown_phototour_revisited`

## How to use

There is a single function, which does everything for you: `full_evaluation`. The original Brown benchmark consider evaluation, similar to cross-validation: train descriptor on one subset, evaluate on two others, repeat for all, so 6 evaluations are required. For the handcrafted descriptors, or those, that are trained on 3rd party datasets, only 3 evaluations are necessary.  We are following it here as well.

However, if you need to run some tests separately, or reuse some functions -- we will cover the usage below.
In the following example we will show how to use `full_evaluation` to evaluate SIFT descriptor as implemented in kornia.

```python
# !pip install kornia
```

    Requirement already satisfied: kornia in /home/mishkdmy/.local/lib/python3.7/site-packages (0.4.1+0c9e625)
    Requirement already satisfied: numpy in /home/mishkdmy/.conda/envs/fastai1/lib/python3.7/site-packages (from kornia) (1.18.0)
    Requirement already satisfied: torch<1.7.0,>=1.6.0 in /home/mishkdmy/.local/lib/python3.7/site-packages (from kornia) (1.6.0)
    Requirement already satisfied: future in /home/mishkdmy/.local/lib/python3.7/site-packages (from torch<1.7.0,>=1.6.0->kornia) (0.18.2)


```python
import torch
import kornia
from IPython.display import clear_output
from brown_phototour_revisited.benchmarking import *
patch_size = 65 

model = kornia.feature.SIFTDescriptor(patch_size, rootsift=True).eval()

descs_out_dir = 'data/descriptors'
download_dataset_to = 'data/dataset'
results_dir = 'data/mAP'

results_dict = {}
results_dict['Kornia RootSIFT'] = full_evaluation(model,
                                'Kornia RootSIFT',
                                path_to_save_dataset = download_dataset_to,
                                path_to_save_descriptors = descs_out_dir,
                                path_to_save_mAP = results_dir,
                                patch_size = patch_size, 
                                device = torch.device('cuda:0'), 
                           distance='euclidean',
                           backend='pytorch-cuda')
clear_output()
print_results_table(results_dict)
```

    ------------------------------------------------------------------------------
    Mean Average Precision wrt Lowe SNN ratio criterion on UBC Phototour Revisited
    ------------------------------------------------------------------------------
    trained on       liberty notredame  liberty yosemite  notredame yosemite
    tested  on           yosemite           notredame            liberty
    ------------------------------------------------------------------------------
    Kornia RootSIFT        56.70              47.71               48.09 
    ------------------------------------------------------------------------------


## Precomputed benchmark results

We have pre-computed some results for you. The implementation is in the following notebooks in the [examples](examples/) dir:

- [Deep descriptors](examples/evaluate_deep_descriptors.ipynb)
- [Non-deep descriptors](examples/evaluate_non_deep_descriptors.ipynb)

The final table is the following:


**If you found any mistake, please open an issue**

# Detailed examples of usage

There are 3 main modules of the package: `dataset`, `extraction` and `benchmarking`. 
    
To run the benchmark manually one needs two things:
 - extract the descriptors with either `extract_pytorchinput_descriptors` or `extract_numpyinput_descriptors`
 - get the mean average precision (mAP) with `evaluate_mAP_snn_based`
 
Here we will show how to evaluate several descriptors: Pytorch-based HardNet, OpenCV SIFT, skimage BRIEF.

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

    # Found cached data data/dataset/notredame.pt
    data/descriptors/HardNet+Liberty_32px_notredame.npy already exists, loading


```python
print (desc_dict.keys())
```

    dict_keys(['descriptors', 'labels', 'img_idxs'])


Function `extract_pytorchinput_descriptors` expects `torch.nn.Module`, which takes `(B, 1, patch_size, patch_size)` `torch.Tensor` input and outputs `(B, desc_dim)` `torch.Tensor`.

Now we will calculate mAP.

```python
mAP = evaluate_mAP_snn_based(desc_dict['descriptors'],
                             desc_dict['labels'], 
                             desc_dict['img_idxs'],
                             path_to_save_mAP = 'data/mAP/HardNet+Liberty_notredame.npy',
                            backend='pytorch-cuda')
print (f'HardNetLib mAP on Notredame = {mAP:.5f}')
```

    Found saved results data/mAP/HardNet+Liberty_notredame.npy, loading
    HardNetLib mAP on Notredame = 0.61901


Now we will evaluate OpenCV SIFT descriptor. 
Function `extract_numpyinput_descriptors` expects function or object, which takes (patch_size, patch_size) input and outputs (desc_dim) np.array.

As OpenCV doesn't provide such function, we will create it ourselves, or rather take already implemented from [HPatches benchmark repo](https://github.com/hpatches/hpatches-benchmark/blob/master/python/extract_opencv_sift.py#L43)

```python
import cv2
import numpy as np
patch_size = 65

# https://github.com/hpatches/hpatches-benchmark/blob/master/python/extract_opencv_sift.py#L43
def get_center_kp(PS=65.):
    c = PS/2.0
    center_kp = cv2.KeyPoint()
    center_kp.pt = (c,c)
    center_kp.size = PS/5.303
    return center_kp


sift = cv2.SIFT_create()
center_kp = get_center_kp(patch_size)

def extract_opencv_sift(patch):
    #Convert back to UINT8 and provide aux keypoint in the center of the patch
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

    # Found cached data data/dataset/notredame.pt




    <div>
        <style>
            /* Turns off some styling */
            progress {
                /* gets rid of default border in Firefox and Opera. */
                border: none;
                /* Needs to be in here for Safari polyfill so background images work as expected. */
                background-size: auto;
            }
            .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
                background: #F44336;
            }
        </style>
      <progress value='468159' class='' max='468159' style='width:300px; height:20px; vertical-align: middle;'></progress>
      100.00% [468159/468159 04:11<00:00]
    </div>
    


```python
mAP_SIFT = evaluate_mAP_snn_based(desc_dict_sift['descriptors'],
                             desc_dict_sift['labels'], 
                             desc_dict_sift['img_idxs'],
                            path_to_save_mAP = 'data/mAP/OpenCV_SIFT65_notredame.npy',
                            backend='pytorch-cuda')
print (f'OpenCV SIFT PS = {patch_size}, mAP on Notredame = {mAP_SIFT:.5f}')
```

    Found saved results data/mAP/OpenCV_SIFT65_notredame.npy, loading
    OpenCV SIFT PS = 65, mAP on Notredame = 0.45530


Now, let's try some binary descriptor, like BRIEF. Evaluation so far supports two metrics: `"euclidean"` and
`"hamming"`.

Function `extract_numpyinput_descriptors` expects function or object, which takes `(patch_size, patch_size)` input and outputs `(desc_dim)` `np.array`.

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

    # Found cached data data/dataset/notredame.pt
    data/descriptors/skimage_BRIEF_65px_notredame.npy already exists, loading


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

    Found saved results data/mAP/skimageBRIEF_notredame.npy, loading
    skimage BRIEF PS = 65, mAP on Notredame = 0.44817


If you use the benchmark, please cite it:

    @misc{BrownRevisited2020,
      title={UBC PhotoTour Revisied},
      author={Mishkin, Dmytro},
      year={2020},
      url = {https://github.com/ducha-aiki/brown_phototour_revisited}
    }
