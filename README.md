# ConCM: Consistency-Driven Calibration and Matching for Few-Shot Class-Incremental Learning

## Abstract
Few-Shot Class Incremental Learning (FSCIL) is crucial for adapting to the complex open-world environments. Contemporary prospective learning-based space construction methods struggle to balance old and new knowledge, as prototype bias and rigid structures limit the expressive capacity of the embedding space. Different from these strategies, we rethink the optimization dilemma from the perspective of feature-structure dual consistency, and propose a Consistency-driven Calibration and Matching (ConCM) framework that systematically mitigates the knowledge conflict inherent in FSCIL. Specifically, inspired by hippocampal associative memory, we design a memory-aware prototype calibration that extracts generalized semantic attributes from base classes and reintegrates them into novel classes to enhance the conceptual center consistency of features. Further, to consolidate memory associations, we propose dynamic structure matching, which adaptively aligns the calibrated features to a session-specific optimal manifold space, ensuring cross-session structure consistency. This process requires no class-number priors and is theoretically guaranteed to achieve geometric optimality and maximum matching. On large-scale FSCIL benchmarks including mini-ImageNet, CIFAR100 and CUB200, ConCM achieves state-of-the-art performance, with harmonic accuracy gains of up to 3.68% in incremental sessions.

## ConCM Framework

<img src='https://anonymous.4open.science/r/ConCM-7385/figures/framework.png'>

## Results

<img src='https://anonymous.4open.science/r/ConCM-7385/figures/visualization.png'>
<img src='https://anonymous.4open.science/r/ConCM-7385/figures/sota.png'>

## Requirements
- Python 3.10
- [PyTorch 2.5.1](https://pytorch.org)
- tqdm
- matplotlib
- scikit-learn
- numpy
- pandas


## Datasets and pretrained models
We follow [FSCIL](https://github.com/xyutao/fscil) setting and use the same data index_list for training splits across incremental sessions. The datasets are made readily available by the authors of [CEC](https://github.com/icoz69/CEC-CVPR2021?tab=readme-ov-file#datasets-and-pretrained-models) in their github repository. Follow their provided instructions to download and unzip. Please make sure to overwrite the correct path in the shell script.


You can download the pretrained models [here](https://send.now/p59qm6ospr7d). Place the downloaded models under `./params` and unzip it. 



## Extract prior knowledge
The `./prior` folder contains the preprocessed `<mini_imagenet/cub200>_part_prior_train.pickle` files. You can use them directly in the train process without further processing.

 If you want to rebuild the semantic prior knowledge of the dataset. Please download the file of [**glove_840b_300d**](https://nlp.stanford.edu/data/glove.840B.300d.zip) and then perform:
 ```bash
    python ./prior/get_cub200_primitive_knowledge.py
```

```bash
   python ./prior/get_cifar100_primitive_knowledge.py
```

```bash
   python ./prior/get_miniimagenet_primitive_knowledge.py
```


## Training
Find the scripts with the best hyperparameters under `./scripts`. 

As an example, to run the CUB200 experiment from the paper:

 ```bash
    chmod +x ./scripts/run_cub.sh
    ./scripts/run_cub.sh
```
To run the CIFAR100 experiment:

 ```bash
    chmod +x ./scripts/run_cifar100.sh
    ./scripts/run_cifar100.sh
```

To run the mini-ImageNet experiment:

 ```bash
    chmod +x ./scripts/run_miniimage.sh
    ./scripts/run_miniimage.sh
```

For the above experiments find the computed metrics available under: `./checkpoints/<dataset>/result.txt`


## Acknowledgment
Our project references the codes in the following repositories.

- [CEC](https://github.com/icoz69/CEC-CVPR2021)
- [fscil](https://github.com/xyutao/fscil)
- [NC-FSCIL](https://github.com/NeuralCollapseApplications/FSCIL)
- [OrCo](https://github.com/noorahmedds/OrCo)






