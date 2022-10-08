import ISnippet from './ISnippet';

const databackup: ISnippet[] = [
  {
    id: 1,
    title: 'Training Loop',
    subtitle: 'Creates a training loop with progressbar!',
    tags: ['PyTorch', 'Numpy'],
    content: `
    class Net(nn.Module):
  def __init__(self):
      super(Net, self).__init__()
      self.conv1 = nn.Conv2d(1, 20, 5, 1)
      self.conv2 = nn.Conv2d(20, 50, 5, 1)
      self.fc1 = nn.Linear(4*4*50, 500)
      self.fc2 = nn.Linear(500, 10)
  
  def forward(self, x):
      x = F.relu(self.conv1(x))
      x = F.max_pool2d(x, 2, 2)
      x = F.relu(self.conv2(x))
      x = F.max_pool2d(x, 2, 2)
      x = x.view(-1, 4*4*50)
      x = F.relu(self.fc1(x))
      x = self.fc2(x)
      return F.log_softmax(x, dim=1)
      `,
  },
  {
    id: 2,
    title: 'Training Loop',
    subtitle: 'Creates a training loop with progressbar!',
    tags: ['PyTorch', 'Numpy'],
    content: `
    class Net(nn.Module):
  def __init__(self):
      super(Net, self).__init__()
      self.conv1 = nn.Conv2d(1, 20, 5, 1)
      self.conv2 = nn.Conv2d(20, 50, 5, 1)
      self.fc1 = nn.Linear(4*4*50, 500)
      self.fc2 = nn.Linear(500, 10)
  
  def forward(self, x):
      x = F.relu(self.conv1(x))
      x = F.max_pool2d(x, 2, 2)
      x = F.relu(self.conv2(x))
      x = F.max_pool2d(x, 2, 2)
      x = x.view(-1, 4*4*50)
      x = F.relu(self.fc1(x))
      x = self.fc2(x)
      return F.log_softmax(x, dim=1)
      `,
  },
  {
    id: 3,
    title: 'Optimizer',
    subtitle: 'Creates an optimizer and plots loss',
    tags: ['PyTorch', 'Matplotlib'],
    content: `
    class Net(nn.Module):
  def __init__(self):
      super(Net, self).__init__()
      self.conv1 = nn.Conv2d(1, 20, 5, 1)
      self.conv2 = nn.Conv2d(20, 50, 5, 1)
      self.fc1 = nn.Linear(4*4*50, 500)
      self.fc2 = nn.Linear(500, 10)
  
  def forward(self, x):
      x = F.relu(self.conv1(x))
      x = F.max_pool2d(x, 2, 2)
      x = F.relu(self.conv2(x))
      x = F.max_pool2d(x, 2, 2)
      x = x.view(-1, 4*4*50)
      x = F.relu(self.fc1(x))
      x = self.fc2(x)
      return F.log_softmax(x, dim=1)

  def forward(self, x):
      x = F.relu(self.conv1(x))
      x = F.max_pool2d(x, 2, 2)
      x = F.relu(self.conv2(x))
      x = F.max_pool2d(x, 2, 2)
      x = x.view(-1, 4*4*50)
      x = F.relu(self.fc1(x))
      x = self.fc2(x)
      return F.log_softmax(x, dim=1)
      `,
  },
];

const data: ISnippet[] = [
  {
    id: 1,
    title: 'Load MNIST',
    subtitle:
      'Loads MNIST Dataset from torchvision and creates train, valid, test dataloaders.',
    tags: ['PyTorch'],
    content: `import torch
from torchvision import datasets
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import ToTensor

SPLIT = 0.8
BATCH_SIZE = 64

train_dataset = datasets.MNIST("data", train=True, download=True, transform=ToTensor())
test_dataset = datasets.MNIST("data", train=False, download=True, transform=ToTensor())
train_dataset, valid_dataset = random_split(train_dataset, [int(SPLIT*len(train_dataset)), len(train_dataset)-int(SPLIT*len(train_dataset))], generator=torch.Generator().manual_seed(42))

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False) # Note that shuffle is turned off for the test set

print(f"> Data Loaded: \n  {len(train_dataset):,} train\n  {len(valid_dataset):,} valid\n  {len(test_dataset):,} test")
`,
  },
  {
    id: 2,
    title: 'Show image batch',
    subtitle: 'Shows a batch of images from pytorch dataloaders',
    tags: ['PyTorch', 'Matplotlib', 'Numpy'],
    content: `import matplotlib.pyplot as plt
import numpy as np

def show_image_batch(dataloader, n_rows=8, figsize=(10,10)):
    images, labels = next(iter(dataloader))
    N = len(images)
    fig, axs = plt.subplots(n_rows, N//n_rows, figsize=(10,10), constrained_layout=True)
    
    for image, label, ax in zip(images, labels, axs.ravel()[:N]):
        ax.set_title(label.item())
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(image.squeeze())

show_image_batch(train_dataloader, n_rows=8)`,
  },
  {
    id: 3,
    title: 'Set Device',
    subtitle: 'Set the PyTorch device either to GPU or CPU',
    tags: ['PyTorch'],
    content: `DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {DEVICE} device")`,
  },
  {
    id: 4,
    title: 'Custom Dataset',
    subtitle: 'The boilerplate code for creating a custom PyTorch Datast.',
    tags: ['PyTorch', 'Pandas'],
    content: `import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label`,
  },
  {
    id: 5,
    title: 'Simple Neural Network',
    subtitle: 'Simple PyTorch NeuralNetwork Module',
    tags: ['PyTorch'],
    content: `import torch.nn as nn
from collections import OrderedDict

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            OrderedDict([
                ("linear1", nn.Linear(28*28, 512)),
                ("relu1", nn.ReLU()),
                ("linear2", nn.Linear(512, 512)),
                ("relu2", nn.ReLU()),
                ("linear3", nn.Linear(512, 10)),
            ])
            
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits`,
  },
  {
    id: 6,
    title: 'LeNet',
    subtitle: 'Create the LeNet Architecture in PyTorch',
    tags: ['PyTorch'],
    content: `import torch.nn as nn
from collections import OrderedDict

model = nn.Sequential(OrderedDict([
    # First Convolution Block
    ("conv1", nn.Conv2d(1, 6, 5, padding=2)),
    ("relu1", nn.ReLU()),
    ("avgpool1", nn.AvgPool2d(2, stride=2)),
    
    # Second Convolution Block
    ("conv2", nn.Conv2d(6, 16, 5, padding=2)),
    ("relu2", nn.ReLU()),
    ("avgpool2", nn.AvgPool2d(2, stride=2)),
    
    # Classifier
    ("flatten", nn.Flatten()),
    ("linear1", nn.Linear(784, 120)),
    ("relu3", nn.ReLU()),
    ("linear2", nn.Linear(120, 84)),
    ("relu4", nn.ReLU()),
    ("linear3", nn.Linear(84, 10)),
]))`,
  },
  {
    id: 7,
    title: 'Loss function',
    subtitle: 'Quickly create a PyTorch loss function',
    tags: ['PyTorch'],
    content: `import torch.nn as nn

loss_fn = nn.CrossEntropyLoss()`,
  },
  {
    id: 8,
    title: 'Optimizer',
    subtitle: 'Quickly create a PyTorch optimizer',
    tags: ['PyTorch'],
    content: `import torch

LEARNING_RATE = 0.001
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)`,
  },
  {
    id: 9,
    title: 'Simple Training loop',
    subtitle: 'A simple training loop for PyTorch',
    tags: ['PyTorch'],
    content: `
def train_model(model, dataloader, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")`,
  },
  {
    id: 11,
    title: 'Complete Training loop',
    subtitle: 'A complete training loop with progressbar and loss history.',
    tags: ['PyTorch'],
    content: `from tqdm.notebook import tqdm, trange
import copy
import time

def train_step(model, dataloader, loss_fn, optimizer, current_epoch, num_epochs):
    
    # tqdm progressbar
    pbar = tqdm(total=len(dataloader.dataset), desc=f"Epoch {current_epoch}/{num_epochs} ")
    
    # For every batch
    for batch, (X, y) in enumerate(dataloader):
        
        # Compute prediction and loss
        pred = model(X)
        train_loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        
        # Calculate losses
        train_loss = train_loss.item()
        valid_loss = train_loss # TODO: calculate validation loss
        
        # Update tqdm progressbar
        pbar.set_postfix({"Train Loss":f"{train_loss:.6f}", "Valid Loss":f"{valid_loss:.6f}"})
        pbar.update(len(X))
        
    return train_loss, valid_loss

def train_model(model, train_dataloader, loss_fn, optimizer, epochs, weights_path="model.pt"):
    # Track time
    start_time = time.perf_counter()
    
    # Best model and loss
    best_valid_loss = float('inf')
    best_model = model
    
    # Loss histories for plotting
    train_loss_history = []
    valid_loss_history = []
    
    # For every epoch
    for i in range(epochs):
        # Train step
        train_loss, valid_loss = train_step(model, train_dataloader, loss_fn, optimizer, current_epoch=i+1, num_epochs=epochs)
        
        # Track losses
        train_loss_history.append(train_loss)
        valid_loss_history.append(valid_loss)
        
        # Save best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_model = copy.deepcopy(model)
            torch.save(best_model.state_dict(), weights_path)
            tqdm.write(f"> Best Model Saved: {weights_path}")
    
    print(f"Training Done in {time.perf_counter() - start_time} seconds.")
    print(f"best Validation Loss: {best_valid_loss:.6f}")
    
    return best_model, train_loss_history, valid_loss_history

best_model, best_loss = train_model(model, train_dataloader, loss_fn, optimizer, epochs=4, weights_path="model.pt")
`,
  },
  {
    id: 12,
    title: 'Plot loss',
    subtitle:
      'Plot the training and validation loss history returned by the train_model function',
    tags: ['PyTorch', 'Matplotlib'],
    content: `
def plot_loss(train_loss_history, valid_loss_history):
    plt.figure()
    plt.title("Loss")
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    
    plt.plot(np.arange(len(train_loss_history)), train_loss_history)
    plt.plot(np.arange(len(valid_loss_history)), valid_loss_history)
    
    plt.legend(["Train Loss","Valid Loss"])
    plt.show()

plot_loss(train_loss_history, valid_loss_history)`,
  },
  {
    id: 14,
    title: 'Torchvision transforms',
    subtitle: 'A set of torchvision transforms ready to be passed to a Dataset',
    tags: ['PyTorch'],
    content: `
from torchvision import transforms as T 

train_augs = T.Compose([
    T.RandomHorizontalFlip(p = 0.5),
    T.RandomRotation(degrees=(-20, +20)),
    T.ToTensor()

])

valid_augs = T.Compose([
    T.ToTensor()
])`,
  },
  {
    id: 15,
    title: 'Load datasets using ImageFolder',
    subtitle:
      'Load train and valid image dataset when you have folders of images for each class',
    tags: ['PyTorch'],
    content: `from torchvision.datasets import ImageFolder

trainset = ImageFolder("data/train", transform=train_augs)
validset = ImageFolder("data/valid", transform=valid_augs)`,
  },
  {
    id: 16,
    title: 'PyTorch Multi Transform',
    subtitle:
      'transforms for train and valid Datasets, provided as a dictionary',
    tags: ['PyTorch'],
    content: `data_transforms = {
    'train': T.Compose([
        T.RandomResizedCrop(INPUT_SIZE),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': T.Compose([
        T.Resize(INPUT_SIZE),
        T.CenterCrop(INPUT_SIZE),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}`,
  },
  {
    id: 17,
    title: 'Download data if not exists',
    subtitle:
      "Downloads the data using wget only if that data hasn't been downloaded yet",
    tags: ['Other'],
    content: `!wget -nc https://path.com/to/file`,
  },
  {
    id: 18,
    title: 'Unzip data if not exists',
    subtitle:
      "Unzips the data using unzip tool only if that data hasn't been extracted yet",
    tags: ['Other'],
    content: `!unzip -n "/content/A.zip" -d "/content/A/"`,
  },
  {
    id: 19,
    title: 'Unrar data if not exists',
    subtitle:
      "Unrars the data using unrar tool only if that data hasn't been extracted yet",
    tags: ['Other'],
    content: `!unrar -o -x "/content/A.rar" "/content/A/"`,
  },
  {
    id: 20,
    title: 'Show GPU info',
    subtitle: 'Prints info about your gpu. useful in google colab',
    tags: ['Other'],
    content: `!nvidia-smi`,
  },
  {
    id: 21,
    title: 'Show RAM info',
    subtitle: 'Prints info about your RAM.',
    tags: ['Other'],
    content: `!cat /proc/meminfo`,
  },
  {
    id: 22,
    title: 'Profiling (%prun)',
    subtitle: 'Sample code and usage of profiler in jupyter notebooks',
    tags: ['Other'],
    content: `def sum_of_lists(N):
    total = 0
    for i in range(5):
        L = [j ^ (j >> i) for j in range(N)]
        total += sum(L)
    return total

%prun sum_of_lists(1000000)`,
  },
  {
    id: 23,
    title: 'Line Profiler (%lprun)',
    subtitle: 'Installing and usage example of line_profiler',
    tags: ['Other'],
    content: `!pip install line_profiler

%load_ext line_profiler

%lprun -f sum_of_lists sum_of_lists(5000)`,
  },
  {
    id: 24,
    title: 'Memory Profiler (%memit)',
    subtitle: 'Installing and usage example of memory_profiler',
    tags: ['Other'],
    content: `!pip install memory_profiler

%load_ext memory_profiler

%memit sum_of_lists(1000000)`,
  },
  {
    id: 25,
    title: 'Dataset Loading and Preperation',
    subtitle:
      'Loads a dataset, shuffles it, splits it into train and test set and seperates label from features',
    tags: ['Pandas'],
    content: `# Load data
df = pd.read_csv("/content/heart.csv")

# Shuffle dataframe
df = df.sample(frac=1.0).reset_index(drop=True)

# Seperate X,y
X = df.drop(columns=["target"])
y = df["target"]

# Split to train and test
split = 0.7

X_train = X.iloc[ : int(len(X)*split),:].reset_index(drop=True)
X_test = X.iloc[int(len(X)*split) : ,:].reset_index(drop=True)

y_train = y.iloc[ : int(len(X)*split)].reset_index(drop=True)
y_test = y.iloc[int(len(X)*split) : ].reset_index(drop=True)

print(f"Train X size = {len(X_train)}")
print(f"Train y size = {len(y_train)}")
print(f"Test X size = {len(X_test)}")
print(f"Test y size = {len(y_test)}")`,
  },
  {
    id: 26,
    title: 'Pandas Complete read_csv',
    subtitle: 'read_csv but with custom sperator and header names',
    tags: ['Pandas'],
    content: `# Custom 'sep', 'Nan symbol' and 'header_names'
df = pd.read_csv('crx.data', header=None, sep=",", na_values=["?"], names=["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10", "A11", "A12", "A13", "A14", "A15", "A16"])`,
  },
  {
    id: 27,
    title: 'Find Columns with Missing Values',
    subtitle:
      'What columns of a pandas DataFrame have missing values and how many?',
    tags: ['Pandas'],
    content: `df.isna().sum()`,
  },
  {
    id: 28,
    title: 'One-hot encode Categorical Data',
    subtitle: 'One-hot encode categorical data pandas DataFrame',
    tags: ['Pandas'],
    content: `for col in train_df.dtypes[train_df.dtypes == 'object'].index:
    for_dummy = train_df.pop(col)
    train_df = pd.concat([train_df, pd.get_dummies(for_dummy, prefix=col)], axis=1)`,
  },
  {
    id: 29,
    title: 'Matplotlib customizations',
    subtitle: '',
    tags: ['Matplotlib'],
    content: `import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
sns.set_style("dark") # to disable gridlines

# Subplots
fig, axs = plt.subplots(5, 4, figsize=(20,10), dpi=100)

# Disable Ticks
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
`,
  },
  {
    id: 30,
    title: 'Matplotlib disable ticks',
    subtitle: 'Matplotlib disable ticks',
    tags: ['Matplotlib'],
    content: `ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)`,
  },
  {
    id: 31,
    title: 'Matplotlib text',
    subtitle: 'Matplotlib text customizations',
    tags: ['Matplotlib'],
    content: `ax.text(0.5,
        -0.09,
        "Hiiii",
        transform=ax.transAxes,
        c="r",
        horizontalalignment='center',
        verticalalignment='center',
        fontsize=12)`,
  },
  {
    id: 32,
    title: 'Plot confusion matrix',
    subtitle: "Plot confusion matrix using sk-learn's 'ConfusionMatrixDisplay'",
    tags: ['Matplotlib', 'sk-learn'],
    content: `from sklearn.metrics import ConfusionMatrixDisplay

disp = ConfusionMatrixDisplay(confusion_matrix=PASS_HERE, display_labels=labels)
disp.plot(cmap="gray")
plt.show()`,
  },
  {
    id: 33,
    title: 'Plot histogram of pandas columns',
    subtitle: 'Plot histogram of pandas columns',
    tags: ['Pandas'],
    content: `# Continuous Feature
sns.histplot(data=df.column_name, bins=30, kde=True, color='green')

# Categorical Feature
sns.histplot(data=df.column_name)
df.column_name.value_counts()`,
  },
  {
    id: 34,
    title: 'Tensorboard, Early Stopping and Epoch Dots',
    subtitle: '',
    tags: ['Tensorflow'],
    content: `!pip install git+https://github.com/tensorflow/docs

import tensorflow_docs as tfdocs

import pathlib
import shutil
import tempfile

logdir = pathlib.Path(tempfile.mkdtemp())/"tensorboard_logs"
shutil.rmtree(logdir, ignore_errors=True)

# pass this callback to the 'fit' function
def get_callbacks(name):
  return [
    # useful for trainings with a lot of epochs prints "." instead of every epoch
    tfdocs.modeling.EpochDots(),
    # performs early stopping 
    tf.keras.callbacks.EarlyStopping(monitor='val_binary_crossentropy', patience=200),
    # saves data for tensorboard
    tf.keras.callbacks.TensorBoard(logdir/name),
  ]

# ANDDDD
# Load the TensorBoard notebook extension
%load_ext tensorboard

# Open an embedded TensorBoard viewer
%tensorboard --logdir {logdir}

# ORRRRR
# upload to tensorboard.dev 
tensorboard dev upload --logdir  {logdir}

# and add an IFrame into the notebook
display.IFrame(
    src="https://tensorboard.dev/experiment/vW7jmmF9TmKmy3rbheMQpw/#scalars&_smoothingWeight=0.97",
    width="100%", height="800px")`,
  },
  {
    id: 35,
    title: 'Fourier Analysis',
    subtitle:
      'Quickly display the magnitude and phase of an image using fourier analysis',
    tags: ['OpenCV'],
    content: `def fourier_analysis(img):
    fourier_img = cv2.dft(img.astype(np.float32), flags=cv2.DFT_COMPLEX_OUTPUT)
    fourier_img_shift = np.fft.fftshift(fourier_img)
    real = fourier_img_shift[:,:,0]
    imag = fourier_img_shift[:,:,1]
    magnitude = cv2.magnitude(real, imag)
    phase = cv2.phase(real, imag)
    return magnitude, phase`,
  },
  {
    id: 36,
    title: 'Apply Fourier Filter',
    subtitle:
      'Applies fourier filter to an image given the mask and returns the filtered image',
    tags: ['OpenCV'],
    content: `def apply_fourier_filter(img, mask):
    mask = mask[:, :, np.newaxis]
    img_fourier = cv2.dft(img.astype(np.float32), flags=cv2.DFT_COMPLEX_OUTPUT)
    img_fourier_shift = np.fft.fftshift(img_fourier)
    img_fourier_shift *= mask
    img_fourier_shift_back = np.fft.ifftshift(img_fourier_shift)
    img_fourier_inverse = cv2.idft(img_fourier_shift_back, flags=cv2.DFT_SCALE)

    return img_fourier_inverse`,
  },
  {
    id: 37,
    title: 'GHD OpenCV utilities',
    subtitle:
      'simple functions like disp, rgb, bgr, uint8ify for quickly prototyping opencv code',
    tags: ['OpenCV'],
    content: `def rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def bgr(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

def gray(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def disp(img, title='', s=8, vmin=None, vmax=None):
    plt.figure(figsize=(s,s))
    plt.axis('off')
    if vmin is not None and vmax is not None:
        plt.imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
    else:
        plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.show()

def uint8ify(img):
    img -= img.min()
    img /= img.max()
    img *= 255
    return np.uint8(img)

def overlay(a,b):
    # a and b should be float images and between 0-1

    mask = a >= 0.5 # generate boolean mask of everywhere a > 0.5 
    ab = np.zeros_like(a) # generate an output container for the blended image 

    # now do the blending 
    ab[~mask] = (2*a*b)[~mask] # 2ab everywhere a<0.5
    ab[mask] = (1-2*(1-a)*(1-b))[mask] # else this
    
    return ab

def before_after(img_a, img_b, name='', vmin=None, vmax=None, effect_name='Processed'):
    fig, axs = plt.subplots(1,2, constrained_layout=True, figsize=(10,4))
    axs[0].axis('off')
    axs[1].axis('off')
    axs[0].set_title(f'{name} Original')
    axs[1].set_title(f'{name} {effect_name}')
    if vmin is not None and vmax is not None:
        axs[0].imshow(img_a, cmap='gray', vmin=vmin, vmax=vmax)
        axs[1].imshow(img_b, cmap='gray', vmin=vmin, vmax=vmax)
    else:
        axs[0].imshow(img_a, cmap='gray')
        axs[1].imshow(img_b, cmap='gray')
    plt.show()`,
  },
];

// Get unique tags for filtering
export const tags = [
  ...new Set(
    data.flatMap((snippet) => {
      return snippet.tags;
    })
  ),
];

export default data;
