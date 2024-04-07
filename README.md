# cs6910_assignment2
## General Instructions :
1. If running on a local host: Ensure Python is present in your system and also see if these libraries are present in your system
   - pytorch lightning [(lightning)](https://lightning.ai/docs/pytorch/stable/)
   - weights and biases [(wandb)](https://docs.wandb.ai/?_gl=1*1lup0xs*_ga*NzgyNDk5ODQuMTcwNTU4MzMwNw..*_ga_JH1SJHJQXJ*MTcxMDY3NjQ2MS43Ny4xLjE3MTA2NzY0NjQuNTcuMC4w)
   - scikit-learn [(sklearn)](https://scikit-learn.org/stable/)
   - [matplotlib](https://matplotlib.org/)
3. If running on colab/kaggle ignore point 1.
4. If running on local host ensure CUDA is present in system else install anaconda, it provides a virtual environmanet for your codes to run
5. There is only 1file this time so no worries.

follow this guide to install Python in your system:
1. Windows: https://kinsta.com/knowledgebase/install-python/#windows
2. Linux: https://kinsta.com/knowledgebase/install-python/#linux
3. MacOS: https://kinsta.com/knowledgebase/install-python/#mac

### ENSURE PYTORCH LIGHTNING IS PRESENT IN YOUR SYSTEM
if the libraries are not present just run the command:


``` pip install lightning ```


``` pip install pytorch ```

Also ensure anaconda is present, in your system, if not present Download Anaconda [(here)](https://www.anaconda.com/download)

## Running the program:
### FOR PART A
Run the command(Runs in default settings mentioned in table below): 
``` python train_partA.py ```

How to pass arguments:
``` python train.py -e 10 -lr 0.001 -b 32```

#### Available commands
| Name        | Default Value   | Description  |
| --------------------- |-------------| -----|
| -wp --wandb_project | myprojectname	| Project name used to track experiments in Weights & Biases dashboard |
| -we	--wandb_entity| myname | Wandb Entity used to track experiments in the Weights & Biases dashboard. |
|-e, --epochs|5|Number of epochs to train neural network.|
|-b, --batch_size|16|Batch size used to train neural network.|
|-l, --loss|cross_entropy|choices: ["mean_squared_error", "cross_entropy"]|
|-o, --optimizer	|nadam|choices: ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"]|
|-lr, --learning_rate|0.01|Learning rate used to optimize model parameters|
|-m, --momentum	|0.9|Momentum used by momentum and nag optimizers.|
|-beta, --beta	|0.9|Beta used by rmsprop optimizer|
|-beta1, --beta1|0.9|Beta1 used by adam and nadam optimizers.|
|-beta2, --beta2|0.99|Beta2 used by adam and nadam optimizers.|
|-eps, --epsilon|1e-10|Epsilon used by optimizers.|
|-w_d, --weight_decay|0.0005|	Weight decay/Regularization coefficient used by optimizers.|
|-w_i, --weight_init|Xavier|	choices: ["random", "Xavier"]|
|-nhl, --num_layers|5|Number of hidden layers used in feedforward neural network.|
|-sz, --hidden_size	|32|	Number of hidden neurons in a feedforward layer.|
|-a, --activation|tanh|	choices: ["identity", "sigmoid", "tanh", "ReLU"]|
|-cm, --conf_mat|default=0|Signifies whether you want to see the confusion matrix for the run|
|-fpl,'--filter_per_layers|64|Number of filters to be used per convolution layer|
