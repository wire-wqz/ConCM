# ConCM: Consistency-Driven Calibration and Matching for Few-Shot Class-Incremental Learning

## Abstract
Few-Shot Class-Incremental Learning (FSCIL) requires models to adapt to novel classes with limited supervision while preserving learned knowledge. Existing prospective learning-based space construction methods reserve space to accommodate novel classes. However, prototype deviation and structure fixity limit the expressiveness of the embedding space. In contrast to fixed space reservation, we explore the optimization of feature-structure dual consistency and propose a Consistency-driven Calibration and Matching Framework (ConCM) that systematically mitigate the knowledge conflict inherent in FSCIL. Specifically, inspired by hippocampal associative memory, we design a memory-aware prototype calibration that extracts generalized semantic attributes from base classes and reintegrates them into novel classes to enhance the conceptual center consistency of features. Further, we propose dynamic structure matching, which adaptively aligns the calibrated features to a session-specific optimal manifold space, ensuring cross-session structure consistency. Theoretical analysis shows that our method satisfies both geometric optimality and maximum matching, thereby overcoming the need for class-number priors. On large-scale FSCIL benchmarks including mini-ImageNet and CUB200, ConCM achieves state-of-the-art performance, surpassing the current optimal method by 3.20% and 3.68% in harmonic accuracy of incremental sessions.

## ConCM Framework

<img src='./figures/framework.jpg'>

## Results

<img src='./figures/visualization.png'>
<img src='./figures/sota.png'>

## Requirements
- Python 3.6
- [PyTorch xxx](https://pytorch.org)
- tqdm
- matplotlib
- scikit-learn
- numpy
- pandas


## Datasets and pretrained models
We follow [FSCIL](https://github.com/xyutao/fscil) setting and use the same data index_list for training splits across incremental sessions. The datasets are made readily available by the authors of [CEC](https://github.com/icoz69/CEC-CVPR2021?tab=readme-ov-file#datasets-and-pretrained-models) in their github repository. Follow their provided instructions to download and unzip. Please make sure to overwrite the correct path in the shell script.


You can download the pretrained models [here](https:xxxxxxxxxxxxxx). Place the downloaded models under `./params` and unzip it. 



## Extract prior knowledge
The `./prior` folder contains the preprocessed `<mini_imagenet/cub200>_part_prior_train.pickle` files. You can use them directly in the train process without further processing.

 If you want to rebuild the semantic prior knowledge of the dataset. Please download the file of [**glove_840b_300d**](https://nlp.stanford.edu/data/glove.840B.300d.zip) and then perform:
```bash
    python ./prior/get_miniimagenet_primitive_knowledge.py
```
```bash
    python ./prior/get_cub200_primitive_knowledge.py
```

## Training
Find the scripts with the best hyperparameters under `./scripts`. 

As an example, to run the mini-ImageNet experiment from the paper:

    $ chmod +x ./scripts/run_miniimage.sh
    $ ./scripts/run_miniimage.sh

For the above experiments find the computed metrics available under: `./checkpoints/mini_imagenet/result.txt`




