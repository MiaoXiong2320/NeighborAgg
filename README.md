# Birds of a Feather Trust Together: Knowing When to Trust a Classifier via Neighborhood Aggregation



## Abstract

How do we know when the predictions made by a classifier can be trusted? This is a fundamental question which also has immense practical applications, especially in safety-critical areas such as medicine and autonomous driving. The de facto approach is to use the classifier's softmax outputs as a proxy for trustworthiness; however, a recent line of research has shown that this approach tends to yield over-confident results. In this work, we argue that the trustworthiness of a classifier's prediction for a sample is highly associated with two contributing factors: the neighborhood information of the sample and the classifier's output. To combine the best of both worlds, we develop a learnable approach **NeighborAgg** that flexibly and selectively fuses the classifier's output and the neighborhood information. Theoretically, we show that NeighborAgg can be seen as a generalized version of a one-hop graph convolutional network, inheriting the powerful modeling ability to capture the similarity of the sample to its neighbors within each class. Empirically, extensive experiments on image and tabular benchmarks verify our theory and suggest that NeighborAgg outperforms other methods, achieving state-of-the-art trustworthiness performance.

## Installation
1. Install the dependencies using pip:
```bash
$ pip install -r requirement.txt
```


## Running the code

### Training
First, checkpoints of pre-trained base classifier should be put in the output folder and `BASE_CKPT` in the bash file should be modified. 
The architecture of the base classifier also needs to be put into the folder `output` . 
Besides, base classifiers can also be trained from scratch using neural network architecture in `classifiers`. After setting paths, run following code:
```bash
$ cd script
$ bash train_base_clf.sh
```

### Testing
To test your model, use the following command:
```bash
$ cd scripts
$ bash eval_image.sh 
$ bash eval_tabular.sh
```
