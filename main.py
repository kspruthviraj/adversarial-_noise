## Standard libraries
import json

import scipy.linalg

## Imports for plotting

import matplotlib
matplotlib.rcParams['lines.linewidth'] = 2.0
sns.set()


## PyTorch
import torch
import torch.nn.functional as F

# Torchvision
# PyTorch Lightning
try:
    import pytorch_lightning as pl
except ModuleNotFoundError:
    !pip install --quiet pytorch-lightning>=1.4
    import pytorch_lightning as pl

# Path to the folder where the datasets are
DATASET_PATH = "/home/EAWAG/kyathasr/GAN_project/GAN_train/Data/"
# DATASET_PATH = "/home/EAWAG/kyathasr/GAN_project/Data/"

# Path to the folder where the trained models are saved
CHECKPOINT_PATH = "/home/EAWAG/kyathasr/GAN_project/GAN_train/GAN_03_checkpoint/"

# Setting the seed
pl.seed_everything(42)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Fetching the device that will be used throughout this notebook
device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
print("Using device", device)

import pandas as pd
import os


def get_norm_mean_and_std(outpath, data_name):
    Data = pd.read_pickle(outpath + data_name)

    trainFilenames0 = Data[0]
    X_train0 = Data[1]
    y_train10 = Data[2]

    CC = np.mean(X_train0, axis=tuple(range(X_train0.ndim - 1)))
    DD = np.std(X_train0, axis=tuple(range(X_train0.ndim - 1)))

    CC1 = torch.tensor(CC)
    DD1 = torch.tensor(DD)

    return CC1, DD1


def get_data_loader(outpath, data_name):
    Data = pd.read_pickle(outpath + data_name)
    classes = np.load(outpath + '/classes.npy')
    class_weights = torch.load(outpath + 'class_weights_tensor.pt')

    trainFilenames0 = Data[0]
    X_train0 = Data[1]
    y_train10 = Data[2]

    testFilenames0 = Data[3]
    X_test0 = Data[4]
    y_test10 = Data[5]

    valFilenames0 = Data[6]
    X_val0 = Data[7]
    y_val10 = Data[8]

    y_test_max = y_test10.argmax(axis=1)  # The class that the classifier would bet on
    y_test_label = np.array([classes[y_test_max[i]] for i in range(len(y_test_max))], dtype=object)

    y_val_max = y_val10.argmax(axis=1)  # The class that the classifier would bet on
    y_val_label = np.array([classes[y_val_max[i]] for i in range(len(y_val_max))], dtype=object)

    y_train_max = y_train10.argmax(axis=1)  # The class that the classifier would bet on
    y_train_label = np.array([classes[y_train_max[i]] for i in range(len(y_train_max))], dtype=object)

    label_encoder = preprocessing.LabelEncoder()

    y_train = label_encoder.fit_transform(y_train_label)
    data_train = X_train0.astype(np.float64)
    data_train = 255 * data_train
    X_train = data_train.astype(np.uint8)

    y_test = label_encoder.fit_transform(y_test_label)
    data_test = X_test0.astype(np.float64)
    data_test = 255 * data_test
    X_test = data_test.astype(np.uint8)

    y_val = label_encoder.fit_transform(y_val_label)
    data_val = X_val0.astype(np.float64)
    data_val = 255 * data_val
    X_val = data_val.astype(np.uint8)

    train_dataset = CreateDataset_with_y(X=X_train, y=y_train)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=False, num_workers=4, pin_memory=True)

    val_dataset = CreateDataset_with_y(X=X_val, y=y_val)
    val_dataloader = DataLoader(val_dataset, batch_size, shuffle=False, num_workers=4, pin_memory=True)

    test_dataset = CreateDataset_with_y(X=X_test, y=y_test)
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_dataloader, val_dataloader, test_dataloader, class_weights, classes



from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from sklearn import preprocessing
batch_size = 24
image_size=224


def extract_data(outpath, data_name):
    Data = pd.read_pickle(outpath + data_name)
    classes = np.load(outpath + '/classes.npy')

    trainFilenames0 = Data[0]
    X_train0 = Data[1]
    y_train10 = Data[2]

    testFilenames0 = Data[3]
    X_test0 = Data[4]
    y_test10 = Data[5]

    valFilenames0 = Data[6]
    X_val0 = Data[7]
    y_val10 = Data[8]

    y_test_max = y_test10.argmax(axis=1)  # The class that the classifier would bet on
    y_test_label = np.array([classes[y_test_max[i]] for i in range(len(y_test_max))], dtype=object)

    return testFilenames0, X_test0, y_test_label


def get_differences(x_test_cscs, x_test_siam, y_test_cscs,
                    pred_cscs, conf_cscs,
                    pred_siam, conf_siam,
                    outpath):
    sum_cscs = []
    sum_siam = []
    sum_diff = []
    sum_cscs_new = []
    new_sum_diff = []
    pred_different = []

    for i in range(len(x_test_cscs)):
        sum_siam.append(np.sum(x_test_siam[i]))
        sum_cscs.append(np.sum(x_test_cscs[i]))
        sum_diff.append(np.sum(x_test_siam[i]) - np.sum(x_test_cscs[i]))

        img_diff_sum_pxl = np.sum(x_test_siam[i] - x_test_cscs[i]) / (128 * 128 * 3)
        new_image_cscs = np.add(x_test_cscs[i], img_diff_sum_pxl)
        new_sum_diff.append(np.sum(x_test_siam[i]) - np.sum(new_image_cscs))

        sum_cscs_new.append(np.sum(new_image_cscs))

        new_image_cscs = []
        img_diff_sum_pxl = []

    for i in range(len(pred_cscs)):
        if pred_cscs[i] != pred_siam[i]:
            AA = 'TRUE'
        else:
            AA = '0'

        pred_different.append(AA)

    data = np.c_[y_test_cscs, pred_cscs, pred_siam, pred_different,
                 sum_cscs, sum_siam, sum_diff, sum_cscs_new, new_sum_diff,
                 conf_cscs, conf_siam,]
    df = pd.DataFrame(data, columns=['GT', 'pred_cscs', 'pred_siam', 'pred_different',
                                     'sum_cscs', 'sum_siam', 'sum_diff', 'sum_cscs_new',
                                     'new_sum_diff', 'conf_cscs', 'conf_siam'])
    df.to_excel(outpath + "/cscs_vs_siam_differences.xlsx")

    return sum_cscs, sum_siam, sum_diff, sum_cscs_new, new_sum_diff


def get_predict_labels(pickle_file_path):
    from_model = pd.read_pickle(pickle_file_path)
    gt_label = from_model[2]
    pred_label = from_model[3]
    prob = from_model[4]
    conf = []
    conf_all = []
    for i in range(len(gt_label)):
        conf.append(np.max(prob[i]))
        conf_all.append(prob[i])

    return gt_label, pred_label, conf, conf_all


class CreateDataset_with_y(Dataset):
    """Characterizes a dataset for PyTorch"""

    def __init__(self, X, y):
        """Initialization"""
        self.X = X
        self.y = y

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.X)

    def __getitem__(self, index):
        """Generates one sample of data"""
        # Select sample
        image = self.X[index]
        label = self.y[index]
        X = self.transform(image)
        y = label
        #         y = self.transform_y(label)
        #         sample = {'image': X, 'label': label}
        sample = [X, y]
        return sample

    transform = T.Compose([
        T.ToPILImage(),
        T.Resize(224),
        T.ToTensor()])
    transform_y = T.Compose([T.ToTensor()])



# outpath ="/home/EAWAG/kyathasr/DEBUG/zoo_model_1/"
outpath="/home/EAWAG/kyathasr/GAN_project/GAN_train/zoo_model_1/"

train_dataloader_cscs, val_dataloader_cscs, test_dataloader_cscs, class_weights_cscs, classes_cscs = get_data_loader(outpath, "Data_CSCS.pickle")
train_dataloader_siam, val_dataloader_siam, test_dataloader_siam, class_weights_siam, classes_siam = get_data_loader(outpath, "Data_SIAM.pickle")

mean_cscs, std_cscs = get_norm_mean_and_std(outpath, "Data_CSCS.pickle")
mean_siam, std_siam = get_norm_mean_and_std(outpath, "Data_SIAM.pickle")


# filename_cscs, x_test_cscs, y_test_cscs, class_weights_cscs = extract_data(outpath, "Data_CSCS.pickle")
# filename_siam, x_test_siam, y_test_siam, class_weights_siam = extract_data(outpath, "Data_SIAM.pickle")

# gt_siam, pred_siam, conf_siam, conf_siam_all = get_predict_labels(outpath + "trained_models_SIAM/CNN_zoo_01/GT_Pred_GTLabel_PredLabel_prob_model_finetuned.pickle")
# gt_cscs, pred_cscs, conf_cscs, conf_cscs_all = get_predict_labels(outpath + "trained_models_CSCS/CNN_zoo_01/GT_Pred_GTLabel_PredLabel_prob_model_finetuned.pickle")

# sum_cscs, sum_siam, sum_diff, sum_cscs_new, new_sum_diff = get_differences(x_test_cscs, x_test_siam, y_test_cscs,
#                                                                            pred_cscs, conf_cscs,
#                                                                            pred_siam, conf_siam,
#                                                                            outpath)



Data = pd.read_pickle(outpath + "Data_CSCS.pickle")

X_train0=Data[1]
y_train10=Data[2]
X_train0.shape

class_weights = class_weights_siam
label_names = classes_cscs


import timm
import torch
import torch.nn as nn

lr=1e-4
weight_decay=3e-2
gpu_id = 0

basemodel = timm.create_model('deit_base_distilled_patch16_224', pretrained=True, num_classes=35)
model = basemodel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# model = nn.DataParallel(model)
model.to(device)

# total parameters and trainable parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"{total_params:,} total parameters.")
total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
print(f"{total_trainable_params:,} training parameters.")



# criterion = LabelSmoothingCrossEntropy()
# criterion = nn.CrossEntropyLoss()
criterion = nn.CrossEntropyLoss(class_weights)

torch.cuda.set_device(gpu_id)
model.cuda(gpu_id)
criterion = criterion.cuda(gpu_id)

# Observe that all parameters are being optimized
# optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr,weight_decay=weight_decay)


checkpoint_path = "/home/EAWAG/kyathasr/GAN_project/GAN_train/"
PATH = checkpoint_path+'/trained_model_finetuned.pth'
checkpoint = torch.load(PATH, map_location='cuda:0')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


os.environ["TORCH_HOME"] = CHECKPOINT_PATH

# No gradients needed for the network
model.eval()
for p in model.parameters():
    p.requires_grad = False


def eval_model(dataset_loader, img_func=None):
    tp, tp_5, counter = 0., 0., 0.
    for imgs, labels in tqdm(dataset_loader, desc="Validating..."):
        imgs = imgs.to(device)
        labels = labels.to(device)
        if img_func is not None:
            imgs = img_func(imgs, labels)
        with torch.no_grad():
            preds = model(imgs)
        tp += (preds.argmax(dim=-1) == labels).sum()
        tp_5 += (preds.topk(5, dim=-1)[1] == labels[...,None]).any(dim=-1).sum()
        counter += preds.shape[0]
    acc = tp.float().item()/counter
    top5 = tp_5.float().item()/counter
    print(f"Top-1 error: {(100.0 * (1 - acc)):4.2f}%")
    print(f"Top-5 error: {(100.0 * (1 - top5)):4.2f}%")
    return acc, top5


_ = eval_model(test_dataloader_cscs)



def show_prediction(img, label, pred, K=5, adv_img=None, noise=None):

    if isinstance(img, torch.Tensor):
        # Tensor image to numpy
        img = img.cpu().permute(1, 2, 0).numpy()
        # img = (img * NORM_STD[None,None]) + NORM_MEAN[None,None]
        img = np.clip(img, a_min=0.0, a_max=1.0)
        label = label.item()

    # Plot on the left the image with the true label as title.
    # On the right, have a horizontal bar plot with the top k predictions including probabilities
    if noise is None or adv_img is None:
        fig, ax = plt.subplots(1, 2, figsize=(10,2), gridspec_kw={'width_ratios': [1, 1]})
    else:
        fig, ax = plt.subplots(1, 5, figsize=(12,2), gridspec_kw={'width_ratios': [1, 1, 1, 1, 2]})

    ax[0].imshow(img)
    ax[0].set_title(label_names[label])
    ax[0].axis('off')

    if adv_img is not None and noise is not None:
        # Visualize adversarial images
        adv_img = adv_img.cpu().permute(1, 2, 0).numpy()
        # adv_img = (adv_img * NORM_STD[None,None]) + NORM_MEAN[None,None]
        adv_img = np.clip(adv_img, a_min=0.0, a_max=1.0)
        ax[1].imshow(adv_img)
        ax[1].set_title('Adversarial')
        ax[1].axis('off')
        # Visualize noise
        noise = noise.cpu().permute(1, 2, 0).numpy()
        noise = noise * 0.5 + 0.5 # Scale between 0 to 1
        ax[2].imshow(noise)
        ax[2].set_title('Noise')
        ax[2].axis('off')
        # buffer
        ax[3].axis('off')

    if abs(pred.sum().item() - 1.0) > 1e-4:
        pred = torch.softmax(pred, dim=-1)
    topk_vals, topk_idx = pred.topk(K, dim=-1)
    topk_vals, topk_idx = topk_vals.cpu().numpy(), topk_idx.cpu().numpy()
    ax[-1].barh(np.arange(K), topk_vals*100.0, align='center', color=["C0" if topk_idx[i]!=label else "C2" for i in range(K)])
    ax[-1].set_yticks(np.arange(K))
    ax[-1].set_yticklabels([label_names[c] for c in topk_idx])
    ax[-1].invert_yaxis()
    ax[-1].set_xlabel('Confidence')
    ax[-1].set_title('Predictions')

    plt.show()
    plt.close()


exmp_batch, label_batch = next(iter(test_dataloader_cscs))
with torch.no_grad():
    preds = model(exmp_batch.to(device))
for i in range(1,17,5):
    show_prediction(exmp_batch[i], label_batch[i], preds[i])


outpath="/home/EAWAG/kyathasr/GAN_project/GAN_train/zoo_model_1/"

filename_cscs, x_test_cscs, y_test_cscs = extract_data(outpath, "Data_CSCS.pickle")
filename_siam, x_test_siam, y_test_siam = extract_data(outpath, "Data_SIAM.pickle")

gt_siam, pred_siam, conf_siam, conf_siam_all = get_predict_labels(outpath + "trained_models_SIAM/CNN_zoo_01/GT_Pred_GTLabel_PredLabel_prob_model_finetuned.pickle")
gt_cscs, pred_cscs, conf_cscs, conf_cscs_all = get_predict_labels(outpath + "trained_models_CSCS/CNN_zoo_01/GT_Pred_GTLabel_PredLabel_prob_model_finetuned.pickle")

sum_cscs, sum_siam, sum_diff, sum_cscs_new, new_sum_diff = get_differences(x_test_cscs, x_test_siam, y_test_cscs,
                                                                           pred_cscs, conf_cscs,
                                                                           pred_siam, conf_siam,
                                                                           outpath)


def create_hist_subplot(indices, x_test_siam, x_test_cscs):
    fig, axs = plt.subplots(3, 3, figsize=(25, 20))
    fig.subplots_adjust(hspace=.5, wspace=.001)
    axs = axs.ravel()

    for i in range(len(indices)):
        index = indices[i]
        image_siam = x_test_siam[index]
        image_cscs = x_test_cscs[index]

        axs[i].hist([image_siam[image_siam != 0], image_cscs[image_cscs != 0]], 20, alpha=0.7, label=['siam', 'cscs'])
        axs[i].set_title(str(index))
        axs[i].legend(['siam', 'cscs'])

    plt.show()


def create_hist_diff_subplot(indices, x_test_siam, x_test_cscs):
    fig, axs = plt.subplots(3, 3, figsize=(25, 20))
    fig.subplots_adjust(hspace=.5, wspace=.001)
    axs = axs.ravel()

    for i in range(len(indices)):
        index = indices[i]
        diff_img = x_test_siam[index] - x_test_cscs[index]

        axs[i].hist(diff_img[diff_img != 0], 20, alpha=0.7)
        axs[i].set_title(str(index))
        axs[i].legend(['diff'])

    plt.show()


def create_hist_subplot_B(indices, x_test_siam, x_test_cscs):
    fig, axs = plt.subplots(3, 3, figsize=(25, 20))
    fig.subplots_adjust(hspace=.5, wspace=.001)
    axs = axs.ravel()

    for i in range(len(indices)):
        index = indices[i]
        image_siam = x_test_siam[index]
        image_cscs = x_test_cscs[index]
        axs[i].hist(image_siam[image_siam != 0], 20, alpha=0.4, label='siam')
        axs[i].hist(image_cscs[image_cscs != 0], 20, alpha=0.4, label='cscs')
        axs[i].set_title(str(index))
        axs[i].legend(['siam', 'cscs'])

    plt.show()


def plot_two_images(indices_true, indices_false, x_test_siam):
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    fig.subplots_adjust(hspace=.5, wspace=.001)
    axs = axs.ravel()

    new_indices1 = [True_indices[:2], False_indices[:2]]
    new_indices = [item for sublist in new_indices1 for item in sublist]

    for i in range(len(new_indices)):
        index = new_indices[i]
        image = x_test_siam[index]

        axs[i].imshow(image)
        axs[i].set_title(str(index))

    plt.show()


def create_two_hist_diff_subplot(indices_true, indices_false, x_test_siam, x_test_cscs):
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    fig.subplots_adjust(hspace=.5, wspace=.001)
    axs = axs.ravel()
    new_indices1 = [True_indices[:2], False_indices[:2]]
    new_indices = [item for sublist in new_indices1 for item in sublist]

    for i in range(len(new_indices)):
        index = new_indices[i]
        diff_img = x_test_siam[index] - x_test_cscs[index]

        axs[i].hist(diff_img[diff_img != 0], 20, alpha=0.7)
        axs[i].set_title(str(index))
        axs[i].legend(['diff'])

    plt.show()


def create_two_hist_subplot(indices_true, indices_false, x_test_siam, x_test_cscs):
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    fig.subplots_adjust(hspace=.5, wspace=.001)
    axs = axs.ravel()
    new_indices1 = [True_indices[:2], False_indices[:2]]
    new_indices = [item for sublist in new_indices1 for item in sublist]

    for i in range(len(new_indices)):
        index = new_indices[i]
        image_siam = x_test_siam[index]
        image_cscs = x_test_cscs[index]

        axs[i].hist([image_siam[image_siam != 0], image_cscs[image_cscs != 0]], 20, alpha=0.7, label=['siam', 'cscs'])
        axs[i].set_title(str(index))
        axs[i].legend(['siam', 'cscs'])

    plt.show()


def get_pertubation_noise(indices_true, indices_false, x_test_cscs, x_test_siam):
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    fig.subplots_adjust(hspace=.5, wspace=.001)
    axs = axs.ravel()
    new_indices1 = [True_indices[:2], False_indices[:2]]
    new_indices = [item for sublist in new_indices1 for item in sublist]

    for i in range(len(new_indices)):
        index = new_indices[i]

        img_diff = x_test_siam[index] - x_test_cscs[index]
        x_max = max(img_diff.min(), img_diff.max(), key=abs)
        image_diff_big = 0.5 * img_diff / x_max
        im_final = 0.5 + image_diff_big

        axs[i].imshow(im_final)
        axs[i].set_title(str(index))

    plt.show()

True_indices = [348,373,774,1149,2267,2447,2662,3580,3907]
False_indices = [75, 189, 223, 436, 1859, 2661, 2893, 3340, 4018]



get_pertubation_noise(True_indices, False_indices, x_test_cscs, x_test_siam)


create_two_hist_subplot(True_indices, False_indices, x_test_siam, x_test_cscs)

create_two_hist_diff_subplot(True_indices, False_indices, x_test_siam, x_test_cscs)

plot_two_images(True_indices, False_indices, x_test_cscs)


plot_two_images(True_indices, False_indices, x_test_siam)


def show_prediction_pillow(img, label, pred, K=5, adv_img=None, noise=None):
    if isinstance(img, torch.Tensor):
        # Tensor image to numpy
        img = img.cpu().permute(1, 2, 0).numpy()
        # img = (img * NORM_STD[None,None]) + NORM_MEAN[None,None]
        img = np.clip(img, a_min=0.0, a_max=1.0)
        label = label.item()

    # Plot on the left the image with the true label as title.
    # On the right, have a horizontal bar plot with the top k predictions including probabilities
    if noise is None or adv_img is None:
        fig, ax = plt.subplots(1, 2, figsize=(10, 2), gridspec_kw={'width_ratios': [1, 1]})
    else:
        fig, ax = plt.subplots(1, 5, figsize=(12, 2), gridspec_kw={'width_ratios': [1, 1, 1, 1, 2]})

    ax[0].imshow(img)
    ax[0].set_title(label_names[label])
    ax[0].axis('off')

    if adv_img is not None and noise is not None:
        # Visualize adversarial images
        adv_img = adv_img.cpu().permute(1, 2, 0).numpy()
        # adv_img = (adv_img * NORM_STD[None,None]) + NORM_MEAN[None,None]
        adv_img = np.clip(adv_img, a_min=0.0, a_max=1.0)
        ax[1].imshow(adv_img)
        ax[1].set_title('Adversarial')
        ax[1].axis('off')
        # Visualize noise
        # noise = noise.numpy()
        noise = noise * 0.5 + 0.5  # Scale between 0 to 1
        ax[2].imshow(noise)
        ax[2].set_title('Noise')
        ax[2].axis('off')
        # buffer
        ax[3].axis('off')

    if abs(pred.sum().item() - 1.0) > 1e-4:
        pred = torch.softmax(pred, dim=-1)
    topk_vals, topk_idx = pred.topk(K, dim=-1)
    topk_vals, topk_idx = topk_vals.cpu().numpy(), topk_idx.cpu().numpy()
    ax[-1].barh(np.arange(K), topk_vals * 100.0, align='center',
                color=["C0" if topk_idx[i] != label else "C2" for i in range(K)])
    ax[-1].set_yticks(np.arange(K))
    ax[-1].set_yticklabels([label_names[c] for c in topk_idx])
    ax[-1].invert_yaxis()
    ax[-1].set_xlabel('Confidence')
    ax[-1].set_title('Predictions')

    plt.show()
    plt.close()


import numpy as np
import scipy.fftpack


def get_noise_characteristics(image_list1, image_list2):
    noise_list = image_list1 - image_list2

    psd = []
    for image in noise_list:
        fft = np.fft.fft2(image)
        psd.append(np.abs(fft) ** 2)

    psd_mean = np.mean(psd, axis=0)
    psd_std = np.std(psd, axis=0)

    return noise_list, psd_mean, psd_std


def compute_psd(noise_list):
    psd_list = []
    for noise in noise_list:
        noise_fft = scipy.fftpack.fftn(noise)
        psd = np.abs(noise_fft) ** 2
        psd_list.append(psd)
    return psd_list


def generate_noisy_images(x_test_cscs, psd):
    noisy_images = []
    noises = []
    for i in range(len(x_test_cscs)):
        noise = np.random.normal(0, 1, size=x_test_cscs[i].shape)
        noise_fft = scipy.fftpack.fftn(noise)
        noise_fft *= np.sqrt(psd / np.abs(noise_fft) ** 2)
        noise = scipy.fftpack.ifftn(noise_fft).real
        noisy_image = x_test_cscs[i] + noise
        noisy_images.append(noisy_image)
        noises.append(noise)
    return np.array(noisy_images), np.array(noises)


noise_list, psd_mean, psd_std = get_noise_characteristics(x_test_cscs, x_test_siam)
psd_list = compute_psd(noise_list)
psd = np.mean(psd_list, axis=0)
noisy_images, noises = generate_noisy_images(x_test_cscs, psd)

# Encode labels
label_encoder = preprocessing.LabelEncoder()
y_test_encoded = label_encoder.fit_transform(y_test_cscs)

# Prepare test data
data_test = noisy_images * 255
X_test_noisy_cscs = data_test.astype(np.uint8)

# Create test dataset and dataloader
test_noisy_dataset = CreateDataset_with_y(X=X_test_noisy_cscs, y=y_test_encoded)
test_noisy_dataloader = DataLoader(test_noisy_dataset, batch_size, shuffle=False, num_workers=4, pin_memory=True)

# Get batch of test data and run predictions
exmp_noisy_batch, label_noisy_batch = next(iter(test_noisy_dataloader))
with torch.no_grad():
    adv_preds = model(exmp_noisy_batch.to(device))

# Visualize predictions for a subset of the data
for i in range(1, 17, 5):
    show_prediction_pillow(exmp_batch[i], label_batch[i], adv_preds[i], adv_img=exmp_noisy_batch[i], noise=noises[i])

## second version
import numpy as np
import scipy.fftpack


def get_noise_characteristics(image_list1, image_list2):
    # Calculate the noise as the difference between the two image lists
    noise_list = image_list1 - image_list2

    # Calculate the power spectral densities of the noise images
    psd_list = []
    for noise in noise_list:
        # Compute the Fourier transform of the noise
        noise_fft = scipy.fftpack.fftn(noise)

        # Compute the power spectral density (PSD) of the noise
        psd = np.abs(noise_fft) ** 2

        # Add the PSD to the list
        psd_list.append(psd)

    # Average the power spectral densities of the noise images
    psd_mean = np.mean(psd_list, axis=0)

    # Get the standard deviation of the power spectral densities of the noise images
    psd_std = np.std(psd_list, axis=0)

    return psd_mean, psd_std


# Get the noise characteristics of the two image lists
psd_mean, psd_std = get_noise_characteristics(x_test_cscs, x_test_siam)

# Generate a new list of noisy images with the same PSD as the original noise
noisy_images = []
noises = []
for i, image in enumerate(x_test_cscs):
    # Generate noise with values close to 0
    noise = np.random.normal(0, 0.1, size=image.shape)

    # Compute the PSD of the noise
    psd_noise = np.abs(np.fft.fft2(noise / np.std(noise))) ** 2 / np.prod(noise.shape)

    # Compute the scaling factor to match the desired PSD
    scaling_factor = np.sqrt(psd_mean / psd_noise)

    # Scale the noise to match the desired PSD
    noise *= scaling_factor

    # Add the noise to the image where the pixel values are greater than 30
    mask = image > 30
    noisy_image = image.copy()
    noisy_image[mask] += noise[mask]

    # Add the noisy image and noise to their respective lists
    noisy_images.append(noisy_image)
    noises.append(noise)

noisy_images = np.array(noisy_images)
noises = np.array(noises)

# Encode labels
label_encoder = preprocessing.LabelEncoder()
y_test_encoded = label_encoder.fit_transform(y_test_cscs)

# Prepare test data
data_test = noisy_images * 255
X_test_noisy_cscs = data_test.astype(np.uint8)

# Create test dataset and dataloader
test_noisy_dataset = CreateDataset_with_y(X=X_test_noisy_cscs, y=y_test_encoded)
test_noisy_dataloader = DataLoader(test_noisy_dataset, batch_size, shuffle=False, num_workers=4, pin_memory=True)

# Get batch of test data and run predictions
exmp_noisy_batch, label_noisy_batch = next(iter(test_noisy_dataloader))
with torch.no_grad():
    adv_preds = model(exmp_noisy_batch.to(device))

# Visualize predictions for a subset of the data
for i in range(1, 17, 5):
    show_prediction_pillow(exmp_batch[i], label_batch[i], adv_preds[i], adv_img=exmp_noisy_batch[i], noise=noises[i])


### third version
import numpy as np
import scipy.fftpack


def get_pil_noise(image_list1, image_list2):
    return image_list1 - image_list2


def get_noise_characteristics(image_list1, image_list2):
    noise_list = get_pil_noise(image_list1, image_list2)

    # Calculate the power spectral densities of the images
    psd = []
    for image in noise_list:
        fft = np.fft.fft2(image)
        psd.append(np.abs(fft) ** 2)

    # Average the power spectral densities
    psd_mean = np.mean(psd, axis=0)

    # Get the standard deviation of the power spectral densities
    psd_std = np.std(psd, axis=0)

    return psd_mean, psd_std


def generate_noisy_images(image_list, psd_mean, psd_std):
    noisy_images = []
    noises = []
    for i in range(len(image_list)):
        # Generate random noise with the same PSD as the original noise
        noise = np.random.normal(0, 1, size=image_list[i].shape)
        noise_fft = scipy.fftpack.fftn(noise)
        noise_fft *= np.sqrt(psd_mean / np.abs(noise_fft) ** 2)
        noise = scipy.fftpack.ifftn(noise_fft).real

        # Add the noise to the image
        noisy_image = image_list[i] + noise

        # Add the noisy image and noise to the lists
        noisy_images.append(noisy_image)
        noises.append(noise)

    return np.array(noisy_images), np.array(noises)


# Load original image lists x_test_cscs and x_test_siam

# Compute the noise characteristics
psd_mean, psd_std = get_noise_characteristics(x_test_cscs, x_test_siam)

# Generate noisy images with the same PSD as the original noise
noisy_images, noises = generate_noisy_images(x_test_cscs, psd_mean, psd_std)

# Print shapes of noisy_images and noises
print("Noisy Images shape:", noisy_images.shape)
print("Noises shape:", noises.shape)

# Encode labels
label_encoder = preprocessing.LabelEncoder()
y_test_encoded = label_encoder.fit_transform(y_test_cscs)

# Prepare test data
data_test = noisy_images * 255
X_test_noisy_cscs = data_test.astype(np.uint8)

# Create test dataset and dataloader
test_noisy_dataset = CreateDataset_with_y(X=X_test_noisy_cscs, y=y_test_encoded)
test_noisy_dataloader = DataLoader(test_noisy_dataset, batch_size, shuffle=False, num_workers=4, pin_memory=True)

# Get batch of test data and run predictions
exmp_noisy_batch, label_noisy_batch = next(iter(test_noisy_dataloader))
with torch.no_grad():
    adv_preds = model(exmp_noisy_batch.to(device))

# Visualize predictions for a subset of the data
for i in range(1, 17, 5):
    show_prediction_pillow(exmp_batch[i], label_batch[i], adv_preds[i], adv_img=exmp_noisy_batch[i], noise=noises[i])


## Using PCA

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Step 1: Get the noise list by subtracting two image lists of RGB numpy array
noise_list = x_test_cscs - x_test_siam

# Step 2: Characterize the noise using PCA
noise_shape = noise_list.shape
noise_list_reshaped = noise_list.reshape((noise_shape[0], -1))
pca = PCA(n_components=3)
pca.fit(noise_list_reshaped)
noise_components = pca.components_.reshape((-1, *noise_shape[1:]))

# Step 3: Use the characterized noise to add it to new unseen images

noisy_images = []
noises=[]
for x in x_test_cscs:
    noise = np.random.normal(scale=0.1, size=noise_shape[1:])
    noise_pca = np.sum([comp*np.random.normal(scale=0.5) for comp in noise_components], axis=0)
    noisy = noise+noise_pca
    noises.append(noisy)
    noisy_image = np.clip(x+noise+noise_pca, 0, 1)
    noisy_images.append(noisy_image)

noisy_images = np.array(noisy_images)
noises = np.array(noises)

# Encode labels
label_encoder = preprocessing.LabelEncoder()
y_test_encoded = label_encoder.fit_transform(y_test_cscs)

# Prepare test data
data_test = noisy_images * 255
X_test_noisy_cscs = data_test.astype(np.uint8)

# Create test dataset and dataloader
test_noisy_dataset = CreateDataset_with_y(X=X_test_noisy_cscs, y=y_test_encoded)
test_noisy_dataloader = DataLoader(test_noisy_dataset, batch_size, shuffle=False, num_workers=4, pin_memory=True)

# Get batch of test data and run predictions
exmp_noisy_batch, label_noisy_batch = next(iter(test_noisy_dataloader))
with torch.no_grad():
    adv_preds = model(exmp_noisy_batch.to(device))

# Visualize predictions for a subset of the data
for i in range(1,17,5):
    show_prediction_pillow(exmp_batch[i], label_batch[i], adv_preds[i], adv_img=exmp_noisy_batch[i], noise=noises[i])


## Using PSD and PCA

import numpy as np
from sklearn.decomposition import PCA


def get_noise_distribution(image_list1, image_list2):
    # Compute the noise by subtracting image_list2 from image_list1
    noise_list = image_list1 - image_list2

    # Compute the power spectral density (PSD) of the noise for each channel separately
    psd_list = []
    for i in range(3):
        psd = np.abs(np.fft.fft2(noise_list[:, :, :, i])) ** 2
        psd_list.append(psd)

    # Perform PCA on the PSDs to get the principal components of the noise distribution
    psd_arr = np.stack(psd_list, axis=-1).reshape(-1, 3)
    pca = PCA(n_components=3)
    pca.fit(psd_arr)
    noise_distribution = pca.components_.T

    return noise_distribution


def add_noise(image_list, noise_distribution, noise_scale):
    noises = []
    noisy_images = []
    for i in range(len(image_list)):
        # Generate random noise from the noise distribution and scale it by noise_scale
        noise = np.random.normal(scale=noise_scale, size=(128, 128, 3))
        noise = noise @ noise_distribution

        # Add the noise to the image list
        noisy_image = np.clip(image_list[i] + noise, 0, 255).astype(np.uint8)
        noisy_images.append(noisy_image)

        # Append noise to noises list
        noises.append(noise)

    return np.array(noises), np.array(noisy_images)


# Load original image lists x_test_cscs and x_test_siam

# Compute the noise distribution
noise_distribution = get_noise_distribution(x_test_cscs, x_test_siam)

# Add noise to new unseen images
new_image_list = x_test_cscs
noises, noisy_images = add_noise(new_image_list, noise_distribution, noise_scale=0.15)

# Print shapes of noisy_images and noises
print("Noisy Images shape:", noisy_images.shape)
print("Noises shape:", noises.shape)

noisy_images = np.array(noisy_images)
noises = np.array(noises)

# Encode labels
label_encoder = preprocessing.LabelEncoder()
y_test_encoded = label_encoder.fit_transform(y_test_cscs)

# Prepare test data
data_test = noisy_images * 255
X_test_noisy_cscs = data_test.astype(np.uint8)

# Create test dataset and dataloader
test_noisy_dataset = CreateDataset_with_y(X=X_test_noisy_cscs, y=y_test_encoded)
test_noisy_dataloader = DataLoader(test_noisy_dataset, batch_size, shuffle=False, num_workers=4, pin_memory=True)

# Get batch of test data and run predictions
exmp_noisy_batch, label_noisy_batch = next(iter(test_noisy_dataloader))
with torch.no_grad():
    adv_preds = model(exmp_noisy_batch.to(device))

# Visualize predictions for a subset of the data
for i in range(1, 17, 5):
    show_prediction_pillow(exmp_batch[i], label_batch[i], adv_preds[i], adv_img=exmp_noisy_batch[i], noise=noises[i])


# Adverserial attacks from literature -- Done only on CSCS data
## Fast Gradient Sign Method (FGSM) attack


def fast_gradient_sign_method(model, imgs, labels, epsilon=0.02):
    # Determine prediction of the model
    inp_imgs = imgs.clone().requires_grad_()
    preds = model(inp_imgs.to(device))
    preds = F.log_softmax(preds, dim=-1)
    # Calculate loss by NLL
    loss = -torch.gather(preds, 1, labels.to(device).unsqueeze(dim=-1))
    loss.sum().backward()
    # Update image to adversarial example as written above
    noise_grad = torch.sign(inp_imgs.grad.to(imgs.device))
    fake_imgs = imgs + epsilon * noise_grad
    fake_imgs.detach_()
    return fake_imgs, noise_grad


# e=0.02  corresponds to changing a pixel value by about 1 in the range of 0 to 255,
# e.g. changing 127 to 128.

adv_imgs, noise_grad = fast_gradient_sign_method(model, exmp_batch, label_batch, epsilon=0.02)
with torch.no_grad():
    adv_preds = model(adv_imgs.to(device))

for i in range(1,17,5):
    show_prediction(exmp_batch[i], label_batch[i], adv_preds[i], adv_img=adv_imgs[i], noise=noise_grad[i])

_ = eval_model(test_dataloader_cscs, img_func=lambda x, y: fast_gradient_sign_method(model, x, y, epsilon=0.02)[0])



## Adverserial patch attack

def place_patch(img, patch):
    for i in range(img.shape[0]):
        h_offset = np.random.randint(0,img.shape[2]-patch.shape[1]-1)
        w_offset = np.random.randint(0,img.shape[3]-patch.shape[2]-1)
        img[i,:,h_offset:h_offset+patch.shape[1],w_offset:w_offset+patch.shape[2]] = patch_forward(patch)
    return img

NORM_MEAN = mean_cscs
NORM_STD = std_cscs
TENSOR_MEANS, TENSOR_STD = torch.FloatTensor(NORM_MEAN)[:,None,None], torch.FloatTensor(NORM_STD)[:,None,None]

def patch_forward(patch):
    # Map patch values from [-infty,infty] to ImageNet min and max
    patch = (torch.tanh(patch) + 1 - 2 * TENSOR_MEANS) / (2 * TENSOR_STD)
    return patch


def eval_patch(model, patch, val_loader, target_class):
    model.eval()
    tp, tp_5, counter = 0., 0., 0.
    with torch.no_grad():
        for img, img_labels in tqdm(val_loader, desc="Validating...", leave=False):
            # For stability, place the patch at 4 random locations per image, and average the performance
            for _ in range(4):
                patch_img = place_patch(img, patch)
                patch_img = patch_img.to(device)
                img_labels = img_labels.to(device)
                pred = model(patch_img)
                # In the accuracy calculation, we need to exclude the images that are of our target class
                # as we would not "fool" the model into predicting those
                tp += torch.logical_and(pred.argmax(dim=-1) == target_class, img_labels != target_class).sum()
                tp_5 += torch.logical_and((pred.topk(5, dim=-1)[1] == target_class).any(dim=-1), img_labels != target_class).sum()
                counter += (img_labels != target_class).sum()
    acc = tp/counter
    top5 = tp_5/counter
    return acc, top5


def patch_attack(model, target_class, patch_size=64, num_epochs=5):
    # Leave a small set of images out to check generalization.

    train_loader = train_dataloader
    val_loader = val_dataloader

    # Create parameter and optimizer
    if not isinstance(patch_size, tuple):
        patch_size = (patch_size, patch_size)
    patch = nn.Parameter(torch.zeros(3, patch_size[0], patch_size[1]), requires_grad=True)
    optimizer = torch.optim.AdamW([patch], lr=1e-4)
    loss_module = nn.CrossEntropyLoss(class_weights)
    loss_module = loss_module.cuda(0)

    # Training loop
    for epoch in range(num_epochs):
        t = tqdm(train_loader, leave=False)
        for img, _ in t:
            img = place_patch(img, patch)
            img = img.to(device)
            pred = model(img)
            # print(type(img))
            # print(type(pred))
            # print(type(int(target_class[0])))
            # print((target_class[0]))
            # print(int(target_class[0]))
            # target_class = int(target_class[0])
            # print(pred.device)
            labels = torch.zeros(img.shape[0], device=pred.device, dtype=torch.long).fill_(target_class)
            loss = loss_module(pred, labels)
            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()
            t.set_description(f"Epoch {epoch}, Loss: {loss.item():4.2f}")

    # Final validation
    acc, top5 = eval_patch(model, patch, val_loader, target_class)

    return patch.data, {"acc": acc.item(), "top5": top5.item()}



# criterion = LabelSmoothingCrossEntropy()
# criterion = nn.CrossEntropyLoss()
criterion = nn.CrossEntropyLoss(class_weights)

torch.cuda.set_device(gpu_id)
model.cuda(gpu_id)
criterion = criterion.cuda(gpu_id)

# Observe that all parameters are being optimized
# optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr,weight_decay=weight_decay)


# Load evaluation results of the pretrained patches
json_results_file = os.path.join(CHECKPOINT_PATH, "patch_results.json")
json_results = {}
if os.path.isfile(json_results_file):
    with open(json_results_file, "r") as f:
        json_results = json.load(f)

# If you train new patches, you can save the results via calling this function
def save_results(patch_dict):
    result_dict = {cname: {psize: [t.item() if isinstance(t, torch.Tensor) else t
                                   for t in patch_dict[cname][psize]["results"]]
                           for psize in patch_dict[cname]}
                   for cname in patch_dict}
    with open(os.path.join(CHECKPOINT_PATH, "patch_results.json"), "w") as f:
        json.dump(result_dict, f, indent=4)

def get_patches(class_names, patch_sizes):
    result_dict = dict()

    # Loop over all classes and patch sizes
    for name in class_names:
        result_dict[name] = dict()
        for patch_size in patch_sizes:
            # c = label_names.index(name)
            c = np.where(label_names == name)
            # c = functools.reduce(lambda sub, ele: sub * 10 + ele, c)
            c = [int(item) for t in c for item in t]
            c= c[0]
            # res = int("".join(map(str, c)))
            # print(c)
            file_name = os.path.join(CHECKPOINT_PATH, f"{name}_{patch_size}_patch.pt")
            # Load patch if pretrained file exists, otherwise start training
            if not os.path.isfile(file_name):
                patch, val_results = patch_attack(model, target_class=c, patch_size=patch_size, num_epochs=5)
                print(f"Validation results for {name} and {patch_size}:", val_results)
                torch.save(patch, file_name)
            else:
                patch = torch.load(file_name)
            # Load evaluation results if exist, otherwise manually evaluate the patch
            if name in json_results:
                results = json_results[name][str(patch_size)]
            else:
                results = eval_patch(model, patch, test_dataloader, target_class=c)

            # Store results and the patches in a dict for better access
            result_dict[name][patch_size] = {
                "results": results,
                "patch": patch
            }

    return result_dict

# class_names = ['cyclops', 'fragilaria', 'trichocerca', 'conochilus', 'bosmina']
# patch_sizes = [32, 48, 64]
class_names = ['cyclops', 'fragilaria']
patch_sizes = [32, 48]

train_dataloader = train_dataloader_cscs
val_dataloader = val_dataloader_cscs
test_dataloader = test_dataloader_cscs

patch_dict = get_patches(class_names, patch_sizes)
save_results(patch_dict)


def show_patches():
    fig, ax = plt.subplots(len(patch_sizes), len(class_names), figsize=(len(class_names)*2.2, len(patch_sizes)*2.2))
    for c_idx, cname in enumerate(class_names):
        for p_idx, psize in enumerate(patch_sizes):
            patch = patch_dict[cname][psize]["patch"]
            patch = (torch.tanh(patch) + 1) / 2 # Parameter to pixel values
            patch = patch.cpu().permute(1, 2, 0).numpy()
            patch = np.clip(patch, a_min=0.0, a_max=1.0)
            ax[p_idx][c_idx].imshow(patch)
            ax[p_idx][c_idx].set_title(f"{cname}, size {psize}")
            ax[p_idx][c_idx].axis('off')
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    plt.show()
show_patches()


def perform_patch_attack(patch):
    patch_batch = exmp_batch.clone()
    patch_batch = place_patch(patch_batch, patch)
    with torch.no_grad():
        patch_preds = model(patch_batch.to(device))
    for i in range(1,17,5):
        show_prediction(patch_batch[i], label_batch[i], patch_preds[i])



perform_patch_attack(patch_dict['fragilaria'][32]['patch'])










