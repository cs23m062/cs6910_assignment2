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
``` python train.py -e 10 -lr 0.001 -cm 1```
