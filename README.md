# CS6910_Assignment2
Moving on with Convolutional Neural Networks
### Convolutional Neural Networks (CNNs)

**Convolutional Neural Networks (CNNs)** are a class of deep neural networks primarily designed for processing structured grid data, such as images. CNNs are composed of multiple layers, including convolutional layers, pooling layers, and fully connected layers. The key operations in CNNs are:

- **Convolutional Layer**: This layer applies a set of filters to the input data, enabling the network to learn spatial hierarchies and local patterns in the data.
  
- **Pooling Layer**: This layer downsamples the spatial dimensions of the input, reducing the computational complexity and making the network more robust to variations in the input.

- **Fully Connected Layer**: This layer connects every neuron from one layer to every neuron in the next layer, enabling the network to learn global patterns in the data.

CNNs have achieved remarkable success in various computer vision tasks, such as image classification, object detection, and image segmentation. They are widely used in applications like facial recognition, medical image analysis, and autonomous driving.

### GoogLeNet

**GoogLeNet**, also known as Inception v1, is a deep convolutional neural network architecture designed by researchers at Google. It was the winner of the ILSVRC (ImageNet Large Scale Visual Recognition Challenge) 2014 in both classification and detection tasks.

Key features of GoogLeNet include:

- **Inception Modules**: GoogLeNet introduces the concept of "Inception modules," which are multi-branch convolutional blocks that allow the network to capture features at various scales and resolutions efficiently.

- **Global Average Pooling**: Instead of using fully connected layers, GoogLeNet uses global average pooling to reduce the spatial dimensions of the feature maps and directly produce the final predictions, which reduces overfitting and the number of parameters in the network.

- **Auxiliary Classifiers**: To mitigate the vanishing gradient problem during training, GoogLeNet includes auxiliary classifiers in the middle of the network to encourage the network to learn more discriminative features.

GoogLeNet demonstrated state-of-the-art performance on the ImageNet dataset with significantly fewer parameters compared to previous deep learning models. Its efficient architecture and innovative design principles have influenced the development of subsequent CNN architectures, such as Inception v2, v3, and v4.

In this assignment we have tried to utilise the power of CNN model built on our own, and used GoogleNET a pretrained model, fine tuned it on the inaturalist dataset.
You can download Inaturalist [here](https://storage.googleapis.com/wandb_datasets/nature_12K.zip)
## General Instructions :
1. If running on a local host: Ensure Python is present in your system and also see if these libraries are present in your system
   - pytorch lightning [(lightning)](https://lightning.ai/docs/pytorch/stable/)
   - weights and biases [(wandb)](https://docs.wandb.ai/?_gl=1*1lup0xs*_ga*NzgyNDk5ODQuMTcwNTU4MzMwNw..*_ga_JH1SJHJQXJ*MTcxMDY3NjQ2MS43Ny4xLjE3MTA2NzY0NjQuNTcuMC4w)
   - scikit-learn [(sklearn)](https://scikit-learn.org/stable/)
   - [matplotlib](https://matplotlib.org/)
3. If running on colab/kaggle ignore point 1.
4. If running on local host ensure CUDA is present in system else install anaconda, it provides a virtual environment for your codes to run, for fast execution time use either NVIDIA GPU's or use Kaggle.
5. Ensure you have pasted the paths to the inaturalist dataset in the Dataloader code
6. There is only 1file this time so no worries.

follow this guide to install Python in your system:
1. Windows: https://kinsta.com/knowledgebase/install-python/#windows
2. Linux: https://kinsta.com/knowledgebase/install-python/#linux
3. MacOS: https://kinsta.com/knowledgebase/install-python/#mac

### ENSURE PYTORCH LIGHTNING IS PRESENT IN YOUR SYSTEM
if the libraries are not present just run the command:


``` pip install lightning ```


``` pip install pytorch ```


``` pip install wandb ```


Also ensure anaconda is present, in your system, if not present Download Anaconda [(here)](https://www.anaconda.com/download)

## Running the program:
### FOR PART A
Run the command(Runs in default settings mentioned in table below): 
``` python train_partA.py ```

How to pass arguments:
``` python train_partA.py -e 10 -lr 0.001 -b 32```

#### Available commands
| Name        | Default Value   | Description  |
| --------------------- |-------------| -----|
| -wp --wandb_project | myprojectname	| Project name used to track experiments in Weights & Biases dashboard |
| -we	--wandb_entity| myname | Wandb Entity used to track experiments in the Weights & Biases dashboard. |
|-e, --epochs|5|Number of epochs to train neural network.|
|-b, --batch_size|16|Batch size used to train neural network.|
|-o, --optimizer	|Mish|choices: ["Mish", "ReLU", "GELU", "CELU","SiLU","Tanh"]|
|-lr, --learning_rate|0.01|Learning rate used to optimize model parameters|
|-a, --activation|tanh|	choices: ["identity", "sigmoid", "tanh", "ReLU"]|
|-ds,--dense_size|1024|Number of hidden neurons in a fully connected layer|
|-fpl,--filter_per_layers|64|Number of filters to be used per convolution layer|
|-d,--dropout|0|Dropout probability in dense layer|
|-s,--stride|1|length of stride in maxpooling layer|
|-bn,--batchnorm|yes|yes if want to use batch normalization else no|
|-fl,--filter_length|3|length of the filter|
|-fo,--filterorg|same|same will keep same number of filters every layer, double doubles and half halves every layer(maxpool + conv)|

### FOR PART B
The models implement googleNET, a very famous architecture which won the IMAGENET 2014
Run the command(Runs in default settings mentioned in table below): 
``` python train_partB.py ```

How to pass arguments:
``` python train_partB.py -e 10 -lr 0.001 -b 32```

#### Available commands
| Name        | Default Value   | Description  |
| --------------------- |-------------| -----|
| -wp --wandb_project | myprojectname	| Project name used to track experiments in Weights & Biases dashboard |
| -we	--wandb_entity| myname | Wandb Entity used to track experiments in the Weights & Biases dashboard. |
|-e, --epochs|5|Number of epochs to train neural network.|
|-b, --batch_size|16|Batch size used to train neural network.|
|-lr, --learning_rate|0.01|Learning rate used to optimize model parameters|
|-fz,--freeze|5|choices: [0-15]|
