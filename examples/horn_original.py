#### DYNAMICS.PY

# example HORN network with manual weights
# Felix Effenberger, July 21, 2023

import torch
import matplotlib.pyplot as plt

from model import HORN

torch.manual_seed(1) # set random seed
torch.set_grad_enabled(False) # no gradient computation

# 1-dim time series as input
num_input = 1
num_hidden = 32
num_output = 10

# hyperparameters
h = 1.0
alpha = 0.04 # excitability
omega_base = 2. * torch.pi / 28. # natural frequency
gamma_base = 0.01 # damping

# oscillation parameters - varying across units
# uncomment to enable
omega_min = 0.5 * omega_base
omega_max = 2.0 * omega_base
omega = torch.rand(num_hidden) * (omega_max - omega_min) + omega_min
gamma_min = 0.5 * gamma_base
gamma_max = 2.0 * gamma_base
gamma = torch.rand(num_hidden) * (gamma_max - gamma_min) + gamma_min

# construct heterogeneous HORN
model = HORN(num_input, num_hidden, num_output, h, alpha, omega, gamma)
model.eval()

# set input weights
model.i2h.weight[:, :] = torch.randn(num_hidden, 1)
model.i2h.bias[:] = 0

# set recurrent weights
model.h2h.weight[:, :] = torch.randn(num_hidden, num_hidden) * 0.1
model.i2h.bias[:] = 0

# show weight matrix - diagonal terms are feedback connections
plt.matshow(model.h2h.weight)
plt.colorbar()
plt.title('W_hh')

# compute dynamics for 1000 time steps
domain = torch.arange(1000)

# stimulus shape: batch x input dimension x input length, i.e. 1 x 1 x 1000
stimulus = torch.sin(domain * torch.pi * 2 / 60).unsqueeze(0).unsqueeze(0)
stimulus[0, 0, 500:] = 0 # not stimulus for last 500 time steps
stimulus = stimulus.permute(2, 0, 1) # shape: time steps x batch x input dimension

# run model dynamics
random_init = 1.0 # set to None for x_0 and y_0 = 0
out = model.forward(stimulus, random_init = random_init, record = True)
x_t = out['rec_x_t'] # get amplitudes - shape batch x time steps x unit
y_t = out['rec_y_t'] # get velocities

# show amplitude dynamics
plt.figure()
for i in range(num_hidden):
    # plot amplitude dyn for unit i
    plt.plot(domain, x_t[0, :, i])
# plot stimulus
plt.plot(domain, stimulus[:, 0, 0], color = 'k', linewidth = 2)
plt.xlabel('time')
plt.ylabel('amplitude')

plt.show()



########## MODEL.py ##########

# HORN network module
# Felix Effenberger, July 21, 2023

import math
import torch

# HORN model
class HORN(torch.nn.Module):

    def __init__(self, num_input, num_nodes, num_output, h, alpha, omega, gamma):
        super().__init__()

        self.num_input = num_input
        self.num_nodes = num_nodes
        self.num_output = num_output

        # hyperparameters h, alpha, omega, gamma
        self.h = h
        self.alpha = alpha
        self.omega = omega
        self.gamma = gamma

        # precompute omega^2 for DHO equation
        self.omega_factor = self.omega * self.omega

        # precompute 2 * gamma for DHO equation
        self.gamma_factor = 2.0 * self.gamma

        # precompute recurrent gain factor
        self.gain_rec = 1. / math.sqrt(self.num_nodes)

        # input, recurrent and output layers
        self.i2h = torch.nn.Linear(num_input, num_nodes)
        self.h2h = torch.nn.Linear(num_nodes, num_nodes)
        self.h2o = torch.nn.Linear(num_nodes, num_output)

    def dynamics_step(self, x_t, y_t, input_t):
        # one discrete dynamics step based on sympletic Euler integration

        # 1. integrate y_t
        y_t = y_t + self.h * (
            # input (forcing) on y_t
            self.alpha * torch.tanh(
                self.i2h(input_t) # external input
                + self.gain_rec * self.h2h(y_t) # recurrent input from network
            )
            - self.omega_factor * x_t # natural frequency term
            - self.gamma_factor * y_t # damping term
        )

        # 2. integrate x_t with updated y_t, no input here
        x_t = x_t + self.h * y_t

        # return updated x_t, y_t
        return x_t, y_t

    def forward(self, batch, random_init = None, record = False):
        # batch has shape: (time steps, batch size, ...)
        batch_size = batch.size(1)
        num_timesteps = batch.size(0)

        ret = {}

        if record:
            # record dynamics of x_t, y_t
            rec_x_t = torch.zeros(batch_size, num_timesteps, self.num_nodes)
            rec_y_t = torch.zeros(batch_size, num_timesteps, self.num_nodes)
            ret['rec_x_t'] = rec_x_t
            ret['rec_y_t'] = rec_y_t

        # initial conditions for variables x, y for each DHO node
        if not random_init is None:
            # Gauss random initial condition
            x_0 = torch.randn(batch_size, self.num_nodes) * random_init
            y_0 = torch.randn(batch_size, self.num_nodes) * random_init
        else:
            # initial condition x=y=0
            x_0 = torch.zeros(batch_size, self.num_nodes)
            y_0 = torch.zeros(batch_size, self.num_nodes)

        # make x_t, y_t autograd variables for automatic differentiation
        x_t = torch.autograd.Variable(x_0)
        y_t = torch.autograd.Variable(y_0)

        # loop over all time steps, feeding one pixel per time step
        for t in range(num_timesteps):
            # compute one dynamics step and update x_t, y_t
            x_t, y_t = self.dynamics_step(x_t, y_t, batch[t])

            if record:
                rec_x_t[:, t, :] = x_t
                rec_y_t[:, t, :] = y_t

        # linear readout at last time step
        output = self.h2o(x_t)

        ret['output'] = output
        return ret



######### TRAINING.PY ##########

# MNIST train script for HORN
# Felix Effenberger, July 21, 2023

import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torchvision

from model import HORN

# command line arguments
parser = argparse.ArgumentParser(description='HORN training script')
parser.add_argument('--num-hidden', type=int, default=32, help='number of units')
parser.add_argument('--epochs', type=int, default=10, help='number of training epochs')
parser.add_argument('--batch-size', type=int, default=64, help='batch size')
parser.add_argument('--shuffle', action = 'store_true', help='whether to shuffle stimulus time steps')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
parser.add_argument('--h', type=float, default=1.0, help='microscopic time constant h (default: 1)')
parser.add_argument('--alpha', type=float, default=0.04, help='excitability coefficient alpha')
parser.add_argument('--omega', type=float, default=0.224, help='natural frequency omega') # 2 * pi / 28 for sMNIST
parser.add_argument('--gamma', type=float, default=0.01, help='damping coefficient gamma')

args = parser.parse_args()
print(args)

# fix seed
torch.manual_seed(args.seed)

# sMNIST as 1-dim time series
dim_input = 1

# 10 MNIST classes
dim_output = 10

# batch size of the test set
batch_size_train = args.batch_size
batch_size_test = 1000

# to shuffle mnist digits
if args.shuffle:
    perm = torch.randperm(784)

# load dataset
size_validation = 1000 # size of validation dataset
train_set = torchvision.datasets.MNIST(root='data', train=True, transform=torchvision.transforms.ToTensor(), download=True)
test_set = torchvision.datasets.MNIST(root='data', train=False, transform=torchvision.transforms.ToTensor(), download=True)
train_set, valid_set = torch.utils.data.random_split(train_set, [len(train_set) - size_validation, size_validation])

# data loaders
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size_train, shuffle=True)
valid_loader = torch.utils.data.DataLoader(dataset=valid_set, batch_size=batch_size_test, shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size_test, shuffle=False)

# instantiate homogeneous HORN model
# see dynamics.py for an example of a heterogeneous HORN
model = HORN(dim_input, args.num_hidden, dim_output, args.h, args.alpha, args.omega, args.gamma)

# can set input and recurrent weights
# model.i2h.weight = ... # n vector
# model.i2h.bias = ... # n vector
# model.h2h.weight = ... # n x n matrix
# model.h2h.bias = ... # n vector

# bce loss and optimizer for training
loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr = args.lr)

# make output directory and open log file
if not os.path.exists('out'):
    os.makedirs('out')

fh_log = open('out/log.txt', 'a')

# run inference on test set
def evaluate_model(data_loader, epoch = None, batch = None):
    model.eval()
    correct = 0
    test_loss = 0

    with torch.no_grad():
        # loop over batches in data loader
        for i, (images, labels) in enumerate(data_loader):
            # reshape batch
            images = images.reshape(batch_size_test, 1, 784)
            images = images.permute(2, 0, 1)

            if args.shuffle:
                images = images[perm, :, :]

            # run model inference - record true returns dynamics
            output = model(images, record = True)
            prediction = output['output']

            # compute loss + number of correct predictions
            test_loss += loss(prediction, labels).item()
            pred_label = prediction.data.max(1, keepdim=True)[1]
            correct += pred_label.eq(labels.data.view_as(pred_label)).sum()

            if i == 0 and not epoch is None and batch is None:
                # plot unit dynamics (amplitudes) for one sample
                plt.figure()

                # loop over all units
                for i in range(args.n_hidden):
                    plt.plot(output['rec_x_t'][0, :, i])
                plt.title(f'epoch {epoch}, batch {batch}')
                plt.xlabel('time')
                plt.ylabel('amplitude')
                plt.savefig(f'out/dynamics_epoch{epoch:02d}_batch{batch:03d}.png')

                plt.close()

    # compute loss and accuracy
    test_loss /= len(data_loader)
    accuracy = 100. * correct / len(data_loader.dataset)

    return accuracy.item()

# training loop
best_eval = 0.
for epoch in tqdm(range(args.epochs), total = args.epochs):
    tqdm.write(f'epoch {epoch}')

    # loop over batches for one epoch
    for batch, (images, labels) in tqdm(enumerate(train_loader), total = len(train_loader)):
        # set model into train mode
        model.train()

        # reshape samples
        images = images.reshape(-1, 1, 784)

        # dimensions: time x batch x 1
        images = images.permute(2, 0, 1)

        if args.shuffle:
            # shuffle if requested
            images = images[perm, :, :]

        # zero gradients
        optimizer.zero_grad()

        # predict
        output = model(images)
        prediction = output['output']

        # compute loss
        train_loss = loss(prediction, labels)

        # compute gradients
        train_loss.backward()

        # update parameters
        optimizer.step()

        if batch % 100 == 0:
            test_acc = evaluate_model(test_loader, epoch, batch)
            tqdm.write(f'epoch {epoch} batch {batch}: test acc {test_acc:.2f}')

    # compute validation and test accuracy
    valid_acc = evaluate_model(valid_loader)
    test_acc = evaluate_model(test_loader, epoch, batch)
    if valid_acc > best_eval:
        best_eval = valid_acc
        final_test_acc = test_acc

    # log accuracy
    msg = f'val: {valid_acc:.4f}, test: {test_acc:.4f}'
    fh_log.write(msg + '\n')
    tqdm.write(msg)

    # save checkpoint
    fn_checkpoint = f'out/epoch{epoch:02d}.pt'
    torch.save(model.state_dict(), fn_checkpoint)
    tqdm.write(f'wrote checkpoint {fn_checkpoint}')

# all done
msg = f'best test: {final_test_acc:.2f}'
fh_log.write(msg + '\n')
fh_log.close()
print(msg)
