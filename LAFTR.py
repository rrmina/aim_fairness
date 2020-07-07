import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import itertools

import aim_fairness
import aim_fairness.datasets as datasets
from aim_fairness.metrics import DemographicParity_gap

# Adult Dataset
INPUT_DIM = 102
HIDDEN_DIM = 8
Z_DIM = 8
BATCH_SIZE = 2000
NUM_EPOCHS = 1000
LEAKY_RELU_SLOPE = 0.2  # Default alpha of tf.nn.leaky_relu
                        # used by Madras et. al's implementation of LAFTR
LEARNING_RATE = 1e-3
DISCRIMINATOR_CRITERION = "BCE" # or "NormL1" for Normalized L1 Loss
DISC_STEP = 1

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim, activation='lrelu'):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, z_dim)

        if (activation == 'lrelu'):
            self.activation = nn.LeakyReLU(negative_slope=LEAKY_RELU_SLOPE, inplace=True)
        else:
            self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x

class Classifier(nn.Module):
    def __init__(self, z_dim, hidden_dim, output_dim=1, activation='lrelu'):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        if (activation == 'lrelu'):
            self.activation = nn.LeakyReLU(negative_slope=LEAKY_RELU_SLOPE, inplace=True)
        else:
            self.activation = nn.ReLU(inplace=True)

    def forward(self, z):
        z = self.activation(self.fc1(z))
        z = self.bn1(z)
        z = self.fc2(z)
        return z

class Discriminator(nn.Module):
    def __init__(self, z_dim, hidden_dim, output_dim=1, activation='lrelu'):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        if (activation == 'lrelu'):
            self.activation = nn.LeakyReLU(negative_slope=LEAKY_RELU_SLOPE, inplace=True)
        else:
            self.activation = nn.ReLU(inplace=True)

    def forward(self, z):
        z = self.activation(self.fc1(z))
        z = self.bn1(z)
        z = self.fc2(z)
        return z

def main():
    # Device
    device = ("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate Networks
    encoder = Encoder(INPUT_DIM, HIDDEN_DIM, Z_DIM).to(device)
    classifier = Classifier(Z_DIM, HIDDEN_DIM, 1).to(device)
    discriminator = Discriminator(Z_DIM, HIDDEN_DIM, 1).to(device)

    # Dataset and DataLoader
    x_train, y_train, a_train = datasets.adult.load_dataset(train=True, device=device)
    trainset = aim_fairness.TabularFairnessDataset(x_train, y_train, a_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)

    # Optimizers
    enc_cla_params = itertools.chain(encoder.parameters(), classifier.parameters())

    enc_cla_optimizer = optim.Adam(enc_cla_params, lr=LEARNING_RATE, betas=[0.9, 0.999], eps=1e-8)
    disc_optimizer = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=[0.9, 0.999], eps=1e-8)

    # Loss Criterions
    cla_criterion = nn.BCEWithLogitsLoss()
    if (DISCRIMINATOR_CRITERION == "BCE"):
        disc_criterion  = nn.BCEWithLogitsLoss()

    # Training Loop
    losses = {"cla": [], "dec": [], "enc_disc": [], "disc": []}
    for epoch in range(1, NUM_EPOCHS+1):

        epoch_losses = {"cla": [], "dec": [], "enc_disc": [], "disc": []}
        for x, y, a in trainloader:
            x.to(device), y.to(device), a.to(device)

            # Zero out gradients
            enc_cla_optimizer.zero_grad()
            disc_optimizer.zero_grad()

            # Forward Encoder - Classifier
            z = encoder(x)
            cla_out = classifier(z)
            cla_loss = cla_criterion(cla_out, y.float().unsqueeze(-1))
            epoch_losses["cla"].append(cla_loss.item())

            # Forward Encoder - Discriminator
            enc_disc_out = discriminator(z)
            enc_disc_loss = disc_criterion(enc_disc_out, a.float().unsqueeze(-1))
            epoch_losses["enc_disc"].append(cla_loss.item())

            (cla_loss - enc_disc_loss).backward()
            enc_cla_optimizer.step()

            for _ in range(DISC_STEP):
                # Forward Discriminator
                z = encoder(x)
                disc_out = discriminator(z)
                disc_loss = disc_criterion(disc_out, a.float().unsqueeze(-1))

                disc_loss.backward()
                disc_optimizer.step()
            epoch_losses["disc"].append(disc_loss.item())

        # Record the losses
        losses["cla"].append(np.mean(epoch_losses["cla"]))
        losses["enc_disc"].append(np.mean(epoch_losses["enc_disc"]))
        losses["disc"].append(np.mean(epoch_losses["disc"]))

        with torch.no_grad():
            y_hat = (torch.sigmoid(cla_out) > 0.5) * 1
            dp_gap = DemographicParity_gap(a.cpu().numpy(), y_hat.squeeze().cpu().numpy())

        # Print epoch losses
        print("[Epoch {}/{}]   cla:{:>5.6f}  |  enc_disc:{:>5.6f}  |  disc:{:>5.6f}  |  dp_gap:{:>5.6f}".format(epoch, NUM_EPOCHS, cla_loss.item(), enc_disc_loss.item(), disc_loss.item(), dp_gap))

        
main()