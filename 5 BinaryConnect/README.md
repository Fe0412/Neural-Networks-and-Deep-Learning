# README #

This repository is for the final project for E6040 at Columbia University.

Group member:
Yi Luo (yl3364)   
Fan Yang (fy2207)  
Jingyi Yuan (jy2736)  
Xiaowen Zhang (xz2461)  
  
### Overview ###

This project reproduced the basic results of the experiments in [BinaryConnect](http://arxiv.org/abs/1511.00363). The basic functions and network layer classes are in the file BC_layers.py and BC_utils.py. Folders contain Jupyter notebooks and codes for different experiments. For the report of this project, please refer to Coursework. Moreover, you can get access to the report through [sharelatex](https://www.sharelatex.com/project/571e811b9f81d2815728ebbf). You can also check the contribution of works through the commit history in Bitbucket and sharelatex.

* MNIST  
The network structure for MNIST is the same as the one mentioned in the paper. To see the entire experiment configuration and results, please refer to mnist/BC_MNIST_final.ipynb.

* CIFAR-10  
Although I had the access to a server that has 2 Titan X GPU and 64G RAM, it still needs more than 20 hours to run a single experiment using the original setting of the network. Hence we halve the number of hidden units in the convolutional layers for time and computational power reasons.Please go to cifar/BC_CIFAR_det_binary.ipynb, cifar/BC_CIFAR_sto_binary.ipynb, cifar/BC_CIFAR_no_reg.ipynb for details.

* SVHN  
This dataset is too large for us to run even one single experiment. Titan X needs about 50 hours for a single experiment, so I don't know how the authors finished training the network within 2 days using a Titan black GPU. That seems to be impossible. Nevertheless, we implemented the training function for SVHN. Please refer to svhn/BC_SVHN.ipynb

### Code reuse ###

We directly use the code in [pylearn2/scripts/datasets](https://github.com/lisa-lab/pylearn2/tree/master/pylearn2/scripts/datasets) to download and preprocess the datasets. We use Lasagne to implement the necessary layers used in the experiments. Detailed annotations can be found in the codes.  
There are several parameters the authors did not mention in the paper, but appeared in their [implementation of the work](https://github.com/MatthieuCourbariaux/BinaryConnect) (e.g. H, W_LR_scale). When we were implementing the utilities, the performance of the network were extremely bad. Hence we referred to their implementation to find out some clarifications about the parameters. However, our implementation is different from them, although we found that in some cases they are similar. That's because this work is relatively simple - could be implemented within few lines of codes using Lasagne. Hence, the basic layout looks similar.

Yi Luo
05-07-2016