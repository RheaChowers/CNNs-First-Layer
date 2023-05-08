from copy import deepcopy
from tqdm import tqdm
import torch
import torchvision
import os
from torchvision import transforms as T
from random import shuffle
from torchmetrics import Accuracy
from matplotlib import pyplot as plt
import numpy as np

cur_dir = os.getcwd()


def get_model(model_name):
    repo_name = "chenyaofo/pytorch-cifar-models"
    torch.hub.set_dir(cur_dir)
    model_list = torch.hub.list(repo_name, force_reload=True)
    model_name = "cifar10_" + model_name
    assert model_name in model_list
    model = torch.hub.load(repo_name, model_name, pretrained=False)
    return model


def get_first_layer_weights(model):
    if hasattr(model, "features"):  # suitable for vgg's
        layer = model.features[0].weight.detach()
    elif hasattr(model, "conv1"):  # suitable for resnets
        layer = model.conv1.weight.detach()
    elif hasattr(model, "transformer"):  # suitable for visual transformers
        layer = model.transformer.embeddings.patch_embeddings.weight.detach()
    else:
        raise Exception("Getting first layer of your model isn't implemented.")
    return layer


def fit_formula_to_model(
    model_energy_profile, lambdas, lambda_squared, num_images
):
    eta = 0.0000001  # learning rate
    eta_lambda_squared = (1 / (2 * num_images)) * eta * lambda_squared
    max_correlation, best_profile, best_iteration = 0, 0, 0
    pbar = tqdm(range(10000, 5000000, 250))
    # iterate over training steps of the linear model and calculate the 
    # theoretical energy profile for the corresponding step
    for gd_step in pbar:
        # calculate the theoretical energy profile
        vec = (1 - torch.pow((1 - eta_lambda_squared), gd_step)) / eta_lambda_squared
        solution = abs(vec * lambdas)
        solution = solution / solution.max()
        # calculate correlation with model energy profile
        corr = np.corrcoef(solution, model_energy_profile)[0, 1]
        # set aside if this step has higher correlation
        if corr > max_correlation:
            max_correlation = corr
            best_profile = solution
            best_iteration = gd_step
            pbar.set_postfix({"corr": max_correlation, "iter": best_iteration})

    return best_profile, max_correlation


def calc_energy_profile(
    first_layer, model_initialization, components, subtract_init=True, normalize=True
):
    if subtract_init:
        first_layer = first_layer.cpu() - model_initialization.cpu()

    first_layer = first_layer.to(torch.float64)
    weights_in_pca_basis = first_layer.flatten(1) @ components

    num_filters = weights_in_pca_basis.shape[0]

    energy_profile = ((1 / num_filters) * torch.sqrt((weights_in_pca_basis**2).sum(0)))

    if normalize:
        return energy_profile / energy_profile.max()

    return energy_profile


def get_dataloader(use_random_labels, train=True):
    transforms = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616)),
        ]
    )
    batch_size = 256
    dataset = torchvision.datasets.CIFAR10(cur_dir, train=train, transform=transforms)

    if use_random_labels:
        shuffle(dataset.targets)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size if train else 10000,
        num_workers=3,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )
    return loader


def train_model(
    model, train_loader, test_loader, num_epochs=100, use_cuda=False, lr=0.01
):
    use_cuda = torch.cuda.is_available()
    losses = []
    train_accs = []
    test_accs = []
    pbar = tqdm(range(num_epochs))

    if use_cuda:
        model = model.to("cuda")
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss = torch.nn.CrossEntropyLoss()

    acc_metric = Accuracy(task="multiclass", num_classes=10).to("cuda")
    test_acc = None
    for epoch in pbar:
        epoch_loss = 0
        epochlen = 0
        avgacc = 0
        for batch in train_loader:
            x, y = batch
            if use_cuda:
                x, y = x.to("cuda"), y.to("cuda")

            optimizer.zero_grad()
            outp = model(x)
            l = loss(outp, y)
            with torch.no_grad():
                avgacc += acc_metric(outp, y)
            losses.append(l.item())
            epoch_loss += l.item()
            epochlen += 1
            l.backward()
            optimizer.step()

        if epoch % 5 == 0:
            model.eval()
            avg_test_acc = 0
            test_epoch_len = 0
            with torch.no_grad():
                for batch in test_loader:
                    x, y = batch
                    if use_cuda:
                        x, y = x.to("cuda"), y.to("cuda")
                    outp = model(x)
                    l = loss(outp, y)
                    with torch.no_grad():
                        avg_test_acc += acc_metric(outp, y)
                    epoch_loss += l.item()
                    test_epoch_len += 1
            model.train()

        pbar.set_postfix(
            {
                "epoch": epoch,
                "train loss": epoch_loss / epochlen,
                "train acc": avgacc.item() / epochlen,
                "test_acc": (test_acc.item() / test_epoch_len)
                if test_acc is not None
                else None,
            }
        )

