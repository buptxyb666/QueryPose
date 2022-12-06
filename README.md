## QueryPose: Sparse Multi-Person Pose Regression via Spatial-Aware Part-Level Query

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

👏👏👏👏👏👏👏👏👏👏👏 QueryPose is a sparse end-to-end multi-person pose regression framework:
![](readme/framework.jpg)
> [**QueryPose: Sparse Multi-Person Pose Regression via Spatial-Aware Part-Level Query**](https://openreview.net/forum?id=tbId-oAOZo),            
> Yabo Xiao, Kai Su, Xiaojuan Wang, Dongdong Yu, Lei Jin, Mingshu He, Zehuan Yuan;        
> *Published on NeurIPS 2022*   

## Highlights

- **Simple:** QueryPose is sparse end-to-end multi-person pose regression framework, which can directly predict multi-person keypoint sequences from the input image. Our method is built upon Sparse RCNN without transformer encoder.

- **Generalizability:** QueryPose is able to achieve the competitive performance on diverise scenes.


## Models
MS COCO mini-val; 
The results are only with 100 queries;
Backbone | keypoint AP | keypoint AP \* | Times(ms) | download
--- |:---:|:---:|:---:|:---:
[HRNet32_100pro_3x](projects/QueryPose/configs/querypose.hrnet32.100pro.3x.yaml) | 69.8 | 72.3  | 97 | [model]() 
[HRNet48_100pro_3x](projects/QueryPose/configs/querypose.hrnet48.100pro.3x.yaml) | 71.0 | 73.4  | 101 | [model]() 
[swinL_100pro_3x](projects/QueryPose/configs/querypose.swinL.100pro.3x.yaml) | 71.2 | 73.3  | 110 | [model]() 



#### Notes
- We are restructuring our code for accelerating training and inference (reducing the GPU memory significantly as well). Working in Progress 👷‍♂️👷‍♂️👷‍♂️!!! The more effecient version will coming soon.
- The time is calculated on Tesla A100, Other GPU cards are not fully tested.
- For QueryPose, we leverage the heatmap-pretrain model for HRNet-series backbone instead of using online auxiliary heatmap learning in other regression-based methods. We observe that the online auxiliary heatmap learning is useless for our method.
<!-- The current version can reproduce the results reported in paper.  -->

## Installation
The codebases are built on top of [Detectron2](https://github.com/facebookresearch/detectron2) and [Sparse RCNN](https://github.com/PeizeSun/SparseR-CNN). If you have any environment or compilation issues, please refer to them.

1. Install and compile
```
git clone git@github.com:buptxyb666/QueryPose.git
cd QueryPose
python setup.py build develop

pip install timm
```

2. Link coco dataset path to QueryPose/datasets/coco
```
mkdir -p datasets/coco
ln -s /path_to_coco_dataset/annotations datasets/coco/annotations
ln -s /path_to_coco_dataset/train2017 datasets/coco/train2017
ln -s /path_to_coco_dataset/val2017 datasets/coco/val2017
```

3. Train QueryPose
```
python projects/querypose/train_net.py --num-gpus 8 \
    --config-file projects/querypose/configs/querypose.hrnet32.100pro.3x.yaml
```

4. Evaluate QueryPose
```
python projects/querypose/train_net.py --num-gpus 8 \
    --config-file projects/querypose/configs/querypose.hrnet32.100pro.3x.yaml \
    --eval-only MODEL.WEIGHTS path/to/model.pth
```


## Citing

If you find this project useful for your research, please use the following BibTeX entry:

```BibTeX

@inproceedings{xiaoquerypose,
  title={QueryPose: Sparse Multi-Person Pose Regression via Spatial-Aware Part-Level Query},
  author={Xiao, Yabo and Su, Kai and Wang, Xiaojuan and Yu, Dongdong and Jin, Lei and He, Mingshu and Yuan, Zehuan},
  booktitle={Advances in Neural Information Processing Systems}
}

```
