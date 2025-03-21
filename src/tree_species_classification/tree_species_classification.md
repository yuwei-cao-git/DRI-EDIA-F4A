# Deep learning in Tree species classification Tutorial

<table class="tfo-notebook-buttons" align="left" width=90%>
  <td width=30% align="left">     <a target="_blank" href="https://colab.research.google.com/github/yuwei-cao-git/DRI-EDIA-F4A/blob/main/src/tree_species_classification/tree_species_classification.ipynb"><img src="https://tensorflow.google.cn/images/colab_logo_32px.png">Run in Google Colab </a> </td>
  <td width=30% align="left">     <a target="_blank" href="https://github.com/yuwei-cao-git/DRI-EDIA-F4A/blob/main/src/tree_species_classification/tree_species_classification.ipynb"><img src="https://tensorflow.google.cn/images/GitHub-Mark-32px.png">View on Github</a> </td>
  <td width=30% align="left">     <a href="https://drive.google.com/uc?id=1I8Lb3mAlkrUSSmdTyLQPQ52HhsGbF6qX"><img src="https://tensorflow.google.cn/images/download_logo_32px.png">Download Data</a> </td>
</table>




---
---

⚛ **Workflow**

1. Set up the Dataset
2. Create a model
3. Train
4. Test/Visualize result
5. Tune the network
6. Save/Depoly your model
7. Scale up your model

# Install and load required packages


```python
#Uncomment this line to install packages
# %pip install lightning gdown
```


```python
import os
import shutil
import lightning as L
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.loggers import TensorBoardLogger
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn
import pandas as pd
import geopandas as gpd
from sklearn.model_selection import train_test_split
from PIL import Image
import pandas as pd
from pathlib import Path
from sklearn.metrics import confusion_matrix
import sklearn
```

# Download data


```python
#Download the zipped tree crown data
!gdown 1svN8wVUmgvyQeOgj_NZkQtp7m7ehUEu2
```


```python
#Remove data dir if it already exists
if os.path.exists("data"):
    shutil.rmtree("data")

#Unzip the data
!unzip qc_crowns.zip -d data/

#Remove zip file
!rm qc_crowns.zip
```


```python
# List files in the current directory
!ls
```

# Load Crown Data


```python
#Load the crown polygons
crowns_df = gpd.read_file('data/tree_crowns_subset.gpkg')

# Map class labels to binary values
label_mapping = {'coniferous': 0, 'deciduous': 1}
crowns_df['label'] = crowns_df['species_type'].map(label_mapping)

#Set data dir
img_dir = 'data/clipped_crowns'
img_fpaths = list(Path(img_dir).glob("*.png"))

#Convert fpaths ls to data frame
img_df = pd.DataFrame(img_fpaths, columns=['fpath'])
img_df['crown_id'] = img_df['fpath'].apply(lambda x: int(x.stem.split(".")[0].split("_")[1]))

#Join with crowns_df
crowns_df = crowns_df.merge(img_df, on='crown_id', how='left')
crowns_df
```


```python
import seaborn as sns
import matplotlib.pyplot as plt

# Create the count plot with 'label'
ax = sns.countplot(data=crowns_df, x='label', hue='label', palette='viridis', legend=False)

# Add a custom legend
legend_labels = {0: 'Coniferous', 1: 'Deciduous'}
handles = [plt.Rectangle((0, 0), 1, 1, color=ax.patches[i].get_facecolor()) for i in range(len(legend_labels))]
plt.legend(handles, legend_labels.values(), title="Tree Type")

# Set labels and title
plt.xlabel('Label')
plt.ylabel('Count')
plt.title('Distribution of Labels')

plt.show()
```

# Set up Pytorch Dataset


```python
class TreeCrownDataset(Dataset):
    def __init__(self, crowns_df, split, target_res=256, train_augmentations=[]):
        self.target_res = target_res
        self.split = split
        self.crowns_df = crowns_df
        self.train_augmentations = train_augmentations

        # Create a transform to resize and normalize the crown images
        self.transforms = [
            transforms.Resize((target_res, target_res)),
            transforms.ToTensor(),
        ]

        #Add additional transforms for data augmentation if using train dataset
        if self.split == 'train':
            self.transforms.extend(self.train_augmentations)

        # Build transform pipeline
        self.transforms = transforms.Compose(self.transforms)


    def __len__(self):
        return len(self.crowns_df)

    def __getitem__(self, idx):

        target_crown = self.crowns_df.iloc[idx]

        label = torch.tensor(target_crown['label']).long()

        crown_img = Image.open(target_crown['fpath']).convert('RGB')

        crown_tensor = self.transforms(crown_img)

        crown_id = target_crown['crown_id']

        return crown_tensor, label, crown_id
```

# Set up the Lightning Data Module


```python
class TreeCrownDataModule(L.LightningDataModule):
    def __init__(self, crowns_df, batch_size=32, train_augmentations=[]):
        super().__init__()
        self.crowns_df = crowns_df
        self.batch_size = batch_size

    def setup(self, stage=None):

        #Split data into three dataframes for train/val/test
        train_val_df, self.test_df = train_test_split(self.crowns_df,
                                                      test_size=0.15,
                                                      random_state=42)

        self.train_df, self.val_df = train_test_split(train_val_df,
                                                      test_size=0.17,
                                                      random_state=42)

        #Report dataset sizes
        for name, df in [("Train", self.train_df),
                         ("Val", self.val_df),
                         ("Test", self.test_df)]:

            print(f"{name} dataset size: {len(df)}",
                  f"({round(len(df)/len(crowns_df)*100, 0)}%)")

        # Instantiate datasets
        self.train_dataset = TreeCrownDataset(self.train_df, split='train')

        self.val_dataset = TreeCrownDataset(self.val_df, split='val')

        self.test_dataset = TreeCrownDataset(self.test_df, split='test')

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          shuffle=False)

    def predict_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          shuffle=False
                          )

#Set the training data augmentations
train_augmentations = [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation([-90, 90]),
                transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0))
                ]

# Test the datamodule
crowns_datamodule = TreeCrownDataModule(crowns_df, train_augmentations=train_augmentations)
crowns_datamodule.setup()

# Test loading a sample
sample = crowns_datamodule.train_dataset[0]
print(sample[0].shape)
print(sample[1])
```

# Set up The Convolutional Neural Network (CNN)


```python
class CNN(L.LightningModule):
    def __init__(self, lr, pretrained_weights=True):
        super(CNN, self).__init__()

        self.model = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained_weights else None) # IMAGENET1K_V2 vs. random init

        # Modify the final fc layer of model to output a single value for binary classification
        self.model.fc = nn.Linear(self.model.fc.in_features, 1)

        #Add sigmoid activation to the end model
        self.model = nn.Sequential(self.model, nn.Sigmoid())

        self.criterion = nn.BCELoss()

        self.lr = lr

        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self(x).squeeze()
        loss = self.criterion(y_hat, y.float())
        self.log('train_loss', loss, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self(x).squeeze()
        loss = self.criterion(y_hat, y.float())
        self.log('val_loss', loss, on_epoch=True, on_step=False)
        return loss

    def predict_step(self, batch, batch_idx):
        x, y, id = batch
        y_hat = self(x).squeeze()

        return y_hat, y, id

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
```


```python
#Instantiate the model with 1 class (present/absent)
model = CNN(lr=0.0001)
print(model)

#Try passing some data through the model
batch, labels, ids = next(iter(crowns_datamodule.train_dataloader()))

# Pass batch through the model
y_hat = model(batch)

print("\nCrown IDs:\n", ids)

print("\nImage batch shape:\n", batch.shape)

print("\nOutput tensor shape:\n", y_hat.shape)

#View the predicted class probabilities
print("\nPredicted class probabilities:\n",
      y_hat.detach().cpu().numpy().squeeze())
```

# Set up Lightning Trainer


```python
# put together
crowns_datamodule = TreeCrownDataModule(crowns_df, train_augmentations=[])
crowns_datamodule.setup()
model = CNN(lr=0.0001)
tensorboard_logger = TensorBoardLogger('', name='lightning_logs', version=0)
trainer = L.Trainer(max_epochs=10, logger=[tensorboard_logger], devices=1)
```

## Fit the model


```python
trainer.fit(model, datamodule=crowns_datamodule)
```

# Visualize learning curve


```python
%reload_ext tensorboard
%tensorboard --logdir=lightning_logs/
```


```python
def calc_test_oa():
    #Test the model on the test set
    out = trainer.predict(model, datamodule=crowns_datamodule, return_predictions=True)

    # Separate predictions and targets from output
    pred_class_probs = np.concatenate([batch[0] for batch in out])
    obs = np.concatenate([batch[1] for batch in out])
    ids = np.concatenate([batch[2] for batch in out])

    #Convert to obs-pred dataframe
    test_df = pd.DataFrame({'obs': obs, 'pred_class_probs': pred_class_probs, 'crown_id': ids})

    #Convert class probabilities to binary predictions
    test_df['pred_boolean_class'] = (test_df['pred_class_probs'] > 0.5)

    #Convert binary predictions to integers
    test_df['pred'] = test_df['pred_boolean_class'].astype(int)

    #Add a column for correct/incorrect predictions
    test_df['correct'] = test_df['obs'] == test_df['pred']

    #Join with crowns_df
    test_df = test_df.merge(crowns_df, on='crown_id', how='left')

    #Calculate overall accuracy using sklearn
    overall_acc = sklearn.metrics.accuracy_score(y_true=test_df['obs'], y_pred=test_df['pred'])


    #Check how many crowns were classified correctly
    n_correct = len(test_df[test_df['correct'] == True])

    print(f"Summary: {n_correct} / {len(test_df)} crowns were classified correctly.")
    return overall_acc, test_df
```


```python
overall_acc, test_df = calc_test_oa()
print(f"Overall accuracy: {overall_acc:.2f}")
```


```python
print(label_mapping)

#Generate a confusion matrix using seaborn
cm = confusion_matrix(y_true=test_df['obs'],
                      y_pred=test_df['pred'])

#Plot the confusion matrix
classes = ['Coniferous', 'Deciduous']
sns.heatmap(cm, annot=True,
            cmap='YlGn',
            xticklabels=classes,
            yticklabels=classes)


plt.xlabel('Predicted')
plt.ylabel('Observed')
plt.title('Confusion Matrix')
plt.show()


```


```python
# Let's view the incorrectly classified crowns
incorrect_df = test_df[test_df['correct'] == False]

#Plot incorrecty classified coniferous/deciduous crowns

for c_type in test_df['species_type'].unique():

    print(f"\nIncorrectly classified {c_type} crowns.\n")

    # Filter the incorrect crowns by species type
    incorrect_type_df = test_df[(test_df['correct'] == False) & (test_df['species_type'] == c_type)]

    # Number of images
    num_images = len(incorrect_type_df)

    # Determine the grid size
    grid_size = int(num_images**0.5) + 1

    # Create a figure and axes
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))

    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    # Read the incorrect crown files and plot them
    for ax, fpath in zip(axes, incorrect_type_df['fpath']):
        img = Image.open(fpath)
        ax.imshow(img)
        ax.axis('off')

    # Hide any remaining empty subplots
    for ax in axes[num_images:]:
        ax.axis('off')

    plt.tight_layout()

    plt.show()
```

---

# Tune hyperparameters

Forget about ML for a second. Imagine you are baking a cookie. You have 3 things you can change about the cookie:

- Sugar type (white, brown, cane)
- Baking time (15 minutes, 30 minutes)
- Cooking temperature (360, 400 degrees)

There are 12 possible variations of cookies you can make. One of them will be the most delicious.

To find out which cookie tastes the best, you need to make all variations and assign a score
- 🤢
- 🤔
- 😆
- 😍

This is called a hyperparameter sweep. Your three hyperparameters are sugar, baking time, cooking temperature.

```
python make_cookie.py --sugar 'white' --baking_time 15 --temperature 400
python make_cookie.py --sugar 'brown' --baking_time 15 --temperature 400

```

🏄🏽‍♀️ **what combination of parameters produces the best performing model?**

The definition of "best" depends on the work you are doing. In general, "best" refers to the lowest loss. At Lightning, we tend to think of "best" as the lowest loss for the least amount of time spent training.

If we run this training script with different hyperparameter combinations, it produces different loss curves

##### test 1: pretrained weigths


```python
# put together
crowns_datamodule = TreeCrownDataModule(crowns_df, train_augmentations=[])
crowns_datamodule.setup()
csv_logger = CSVLogger('', name='logs', version=1)
tensorboard_logger = TensorBoardLogger('', name='lightning_logs', version=1)
model = CNN(lr=0.01, pretrained_weights=False)
trainer = L.Trainer(max_epochs=10, logger=[csv_logger, tensorboard_logger], devices=1)
trainer.fit(model, datamodule=crowns_datamodule)
```


```python
overall_acc, test_df = calc_test_oa()
print(f"Overall accuracy: {overall_acc:.2f}")
```

##### test 2: different learning rate


```python
# put together
model = CNN(lr=0.01)
csv_logger = CSVLogger('', name='logs', version=2)
tensorboard_logger = TensorBoardLogger('', name='lightning_logs', version=2)
trainer = L.Trainer(max_epochs=10, logger=[csv_logger, tensorboard_logger], devices=1)
trainer.fit(model, datamodule=crowns_datamodule)
```


```python
overall_acc, test_df = calc_test_oa()
print(f"Overall accuracy: {overall_acc:.2f}")
```


```python
%reload_ext tensorboard
%tensorboard --logdir=lightning_logs/
```

# Save/Depoly your model


```python
trainer.save_checkpoint(filepath=".ckpt/model.ckpt")
```


```python
model = CNN.load_from_checkpoint(".ckpt/model.ckpt", lr=0.01)
model.freeze()

crowns_datamodule = TreeCrownDataModule(crowns_df, train_augmentations=[])
crowns_datamodule.setup()
test_predictions = trainer.predict(model, datamodule=crowns_datamodule)
```


```python
overall_acc, test_df = calc_test_oa()
print(f"Overall accuracy: {overall_acc:.2f}")
```

TorchScript allows you to serialize your models in a way that it can be loaded in non-Python environments. The LightningModule has a handy method to_torchscript() that returns a scripted module which you can save or directly use.


```python
script = model.to_torchscript()

# save for use in production environment
torch.jit.save(script, ".ckpt/model.pt")

# use it
#Try passing some data through the model
batch, labels, ids = next(iter(crowns_datamodule.test_dataloader()))

scripted_module = torch.jit.load(".ckpt/model.pt")
output = scripted_module(batch)
```

---

# Scale up your model/dataset

You can either make all cookies sequentially (which will take you 4.5 hours). Or you can get 12 kitchens and cook them all in parallel, and you'll know in 30 minutes.

If a kitchen is a GPU, then you need 12 GPUs to run each experiment to see which cookie is the best. The power of Lightning is the ability to run sweeps like this on 12 different GPUs (or 1,000 GPUs if you'd like) to get you the best version of a model fast.

Train on GPUs
The Trainer will run on all available GPUs by default. Make sure you’re running on a machine with at least one GPU. There’s no need to specify any NVIDIA flags as Lightning will do it for you.


```python
from lightning import Trainer

# run on as many GPUs as available by default
trainer = Trainer(accelerator="auto", devices="auto", strategy="auto")
# equivalent to
trainer = Trainer()

# run on one GPU
trainer = Trainer(accelerator="gpu", devices=1)
# run on multiple GPUs
trainer = Trainer(accelerator="gpu", devices=8)
# choose the number of devices automatically
trainer = Trainer(accelerator="gpu", devices="auto")
```

Train on Slurm Cluster


```python
# train.py
def main(args):
    model = CNN(args)

    trainer = Trainer(accelerator="gpu", devices=8, num_nodes=4, strategy="ddp")

    trainer.fit(model)


if __name__ == "__main__":
    args = ...  # you can use your CLI parser of choice, or the `LightningCLI` or using config.yaml
    # TRAIN
    main(args)
```


```python
%%writefile submit.sh
# (submit.sh)
#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=4             # This needs to match Trainer(num_nodes=...)
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8   # This needs to match Trainer(devices=...)
#SBATCH --mem=0
#SBATCH --time=0-02:00:00

# activate conda env
source activate $1

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# on your cluster you might need these:
# set the network interface
# export NCCL_SOCKET_IFNAME=^docker0,lo

# might need the latest CUDA
# module load NCCL/2.4.7-1-cuda.10.0

# run script from above
srun python3 train.py
```


```python
%%!
sbatch submit.sh
```

Or you can even parallel the baking procedure...

![image](https://shared.fastly.steamstatic.com/store_item_assets/steam/apps/448510/ss_2a35c15c78f06dd4f23dab8a1e1917a835d0062d.1920x1080.jpg?t=1741368176)

# wandb sweep


```python
import wandb

wandb.login()
```


```python
%%html
<iframe src="https://api.wandb.ai/links/ubc-yuwei-cao/ebnspmv1" style="border:none;height:1024px;width:100%">
```
