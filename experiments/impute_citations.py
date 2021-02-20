#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional
import torch.utils.data as data
import numpy as np
import sys

sys.path.append('.')
import scnn.scnn
import scnn.chebyshev


class MySCNN(nn.Module):
    def __init__(self, colors=1, res_add=False):
        super().__init__()

        assert(colors > 0)
        self.colors = colors
        self.res_add = res_add

        num_filters = 30
        variance = 0.01

        # Degree 0 convolutions.
        self.C0_1 = scnn.scnn.SimplicialConvolution(5, self.colors, num_filters * self.colors, variance=variance)
        self.C0_2 = scnn.scnn.SimplicialConvolution(5, num_filters * self.colors, num_filters * self.colors, variance=variance)
        self.C0_3 = scnn.scnn.SimplicialConvolution(5, num_filters * self.colors, self.colors, variance=variance)

        # Degree 1 convolutions.
        self.C1_1 = scnn.scnn.SimplicialConvolution(5, self.colors, num_filters * self.colors, variance=variance)
        self.C1_2 = scnn.scnn.SimplicialConvolution(5, num_filters * self.colors, num_filters * self.colors, variance=variance)
        self.C1_3 = scnn.scnn.SimplicialConvolution(5, num_filters * self.colors, self.colors, variance=variance)

        # Degree 2 convolutions.
        self.C2_1 = scnn.scnn.SimplicialConvolution(5, self.colors, num_filters * self.colors, variance=variance)
        self.C2_2 = scnn.scnn.SimplicialConvolution(5, num_filters * self.colors, num_filters * self.colors, variance=variance)
        self.C2_3 = scnn.scnn.SimplicialConvolution(5, num_filters * self.colors, self.colors, variance=variance)

    def forward(self, Ls, Ds, adDs, xs):
        assert(len(xs) == 3) # The three degrees are fed together as a list.

        assert(len(Ls) == len(Ds))
        Ms = [L.shape[0] for L in Ls]
        Ns = [D.shape[0] for D in Ds]

        Bs = [x.shape[0] for x in xs]
        C_ins = [x.shape[1] for x in xs]
        Ms = [x.shape[2] for x in xs]

        assert(Ms == [D.shape[1] for D in Ds])
        assert(Ms == [L.shape[1] for L in Ls])
        assert([adD.shape[0] for adD in adDs] == [D.shape[1] for D in Ds])
        assert([adD.shape[1] for adD in adDs] == [D.shape[0] for D in Ds])

        assert(Bs == len(Bs)*[Bs[0]])
        assert(C_ins == len(C_ins)*[C_ins[0]])

        out0_1 = self.C0_1(Ls[0], xs[0]) #+ self.D10_1(xs[1])
        out1_1 = self.C1_1(Ls[1], xs[1]) #+ self.D01_1(xs[0]) + self.D21_1(xs[2])
        out2_1 = self.C2_1(Ls[2], xs[2]) #+ self.D12_1(xs[1])

        out0_2 = self.C0_2(Ls[0], nn.LeakyReLU()(out0_1)) #+ self.D10_2(nn.LeakyReLU()(out1_1))
        out1_2 = self.C1_2(Ls[1], nn.LeakyReLU()(out1_1)) #+ self.D01_2(nn.LeakyReLU()(out0_1)) + self.D21_2(nn.LeakyReLU()(out2_1))
        out2_2 = self.C2_2(Ls[2], nn.LeakyReLU()(out2_1)) #+ self.D12_2(nn.LeakyReLU()(out1_1))

        out0_3 = self.C0_3(Ls[0], nn.LeakyReLU()(out0_2)) #+ self.D10_3(nn.LeakyReLU()(out1_2))
        out1_3 = self.C1_3(Ls[1], nn.LeakyReLU()(out1_2)) #+ self.D01_3(nn.LeakyReLU()(out0_2)) + self.D21_2(nn.LeakyReLU()(out2_2))
        out2_3 = self.C2_3(Ls[2], nn.LeakyReLU()(out2_2)) #+ self.D12_3(nn.LeakyReLU()(out1_2))

        if self.res_add:
            return [
                out0_3 + 0.5 * xs[0],
                out1_3 + 0.5 * xs[1],
                out2_3 + 0.5 * xs[2],
            ]
        return [out0_3, out1_3, out2_3]


def load_cochains(path, batch_size=1, top_dim=2):
    """ Loads the cochains contained at path.

    Args:
        path: the string of the path to load
        batch_size: the batch size to load in with
        top_dim: the top dimension of simplices to consider

    Returns:
        A list containing elements consisting of batch_size cochains contained
        at path
    """
    cochains = []
    signal = np.load(path, allow_pickle=True)
    raw_data=[list(signal[i].values()) for i in range(len(signal))]

    for d in range(top_dim + 1):
        cochain_target = torch.zeros((batch_size, 1, len(raw_data[d])),
                                     dtype=torch.float, requires_grad=False)

        for i in range(0, batch_size):
            cochain_target[i, 0, :] = torch.tensor(raw_data[d],
                                                   dtype=torch.float,
                                                   requires_grad=False)
        cochains.append(cochain_target)
    return cochains


def construct_model_inputs(
        path_prefix, file_identifier, top_dim=2,
        batch_size=1, percentage_missing_values='30'):
    """ Returns the cochain inputs, targets, Laplacians, Ls, Ds, and adDs.

    Args:
        path_prefix: the directory to look under
        file_identifier: the unique file identifier for the data
        top_dim: the top dimension of simplices to consider
        batch_size: the batch size to use
        percentage_missing_values: the percentage of missing values to expect.
            Used for file lookup purposes.

    Returns:
        A tuple consisting of the following matrices: Laplacians, Ls, Ds, adDs
    """
    cochain_input = load_cochains(
        path=f'{path_prefix}/{file_identifier}_'
             f'percentage_{percentage_missing_values}_input_damaged.npy',
        batch_size=batch_size,
        top_dim=top_dim)
    cochain_target = load_cochains(
        path=f'{path_prefix}/{file_identifier}_cochains.npy',
        batch_size=batch_size,
        top_dim=top_dim)
    laplacians = np.load(
        '{}/{}_laplacians.npy'.format(
            path_prefix, file_identifier), allow_pickle=True)
    boundaries = np.load(
        '{}/{}_boundaries.npy'.format(
            path_prefix, file_identifier), allow_pickle=True)
    Ls = [
            scnn.scnn.coo2tensor(
                scnn.chebyshev.normalize(
                    laplacians[i],half_interval=True))
            for i in range(top_dim + 1)
    ]
    Ds = [
            scnn.scnn.coo2tensor(
                boundaries[i].transpose()) 
            for i in range(top_dim + 1)
    ]
    adDs = [
        scnn.scnn.coo2tensor(boundaries[i]) for i in range(top_dim + 1)
    ]
    return cochain_input, cochain_target, laplacians, Ls, Ds, adDs


class Hparams:
    def __init__(
            self,
            learning_rate: float,
            batch_size: int,
            train_steps: int,
            top_dim: int,
            optimizer: torch.optim.Optimizer,
            loss_criterion: torch.nn.Module,
            res_add: bool):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.train_steps = train_steps
        self.top_dim = top_dim
        self.optimizer = optimizer
        self.loss_criterion = loss_criterion
        self.res_add = res_add


def train_and_test(hparams: Hparams):
    torch.manual_seed(1337)
    np.random.seed(1337)

    path_prefix = sys.argv[1]  # Input
    logdir = sys.argv[2]  # Output
    file_identifier = sys.argv[3]
    percentage_missing_values = sys.argv[4]
    cuda = False
    batch_size = hparams.batch_size
    top_dim = hparams.top_dim
    network = MySCNN(colors=1, res_add=hparams.res_add)
    optimizer = hparams.optimizer(
        network.parameters(), lr=hparams.learning_rate)
    criterion = hparams.loss_criterion

    cochain_train_input, cochain_train_target,\
        laplacians, Ls, Ds, adDs = construct_model_inputs(
            path_prefix=path_prefix, file_identifier=file_identifier,
            top_dim=top_dim, batch_size=batch_size,
            percentage_missing_values=percentage_missing_values)
    num_params = 0
    print("Parameter counts:")
    for param in network.parameters():
        p = np.array(param.shape, dtype=int).prod()
        print(p)
        num_params += p
    print(f'Total number of parameters: {num_params}')

    masks_all_deg = np.load(
        '{}/{}_percentage_{}_known_values.npy'.format(
            path_prefix,file_identifier,percentage_missing_values),
        allow_pickle=True)
    masks = [list(
        masks_all_deg[i].values()) for i in range(len(masks_all_deg))]
    losslogf = open(f'{logdir}/loss.txt', "w")

    print(
        [
            len(masks[d]) / len(cochain_train_target[d][0,0,:])
                for d in range(top_dim + 1)
        ])

    for i in range(hparams.train_steps):
        xs = [cochain_input.clone(
            ) for cochain_input in cochain_train_input]

        optimizer.zero_grad()
        ys = network(Ls, Ds, adDs, xs)

        loss = torch.FloatTensor([0.0])
        for b in range(batch_size):
            for d in range(top_dim + 1):
                loss += criterion(ys[d][
                    b, 0, masks[d]], cochain_train_target[d][
                        b, 0, masks[d]])

        detached_ys = [ys[d].detach() for d in range(0, top_dim+1)]

        if np.mod(i, 10) == 0:
            for d in range(0,top_dim+1):
                np.savetxt(
                    f'{logdir}/output_{i}_{d}.txt', detached_ys[d][0,0,:])

        for d in range(0, top_dim+1):
            predictionlogf = open(f'{logdir}/prediction_{i}_{d}.txt', "w")
            actuallogf = open(f'{logdir}/actual_{i}_{d}.txt', "w")

            for b in range(0, batch_size):
                for y in detached_ys[d][b, 0, masks[d]]:
                    predictionlogf.write("%f " %(y))
                predictionlogf.write("\n")
                for x in cochain_train_target[d][b, 0, masks[d]]:
                    actuallogf.write("%f " %(x))
                actuallogf.write("\n")
            predictionlogf.close()
            actuallogf.close()

        losslogf.write("%d %f\n" %(i, loss.item()))
        losslogf.flush()
        loss.backward()
        optimizer.step()
    losslogf.close()

    # Test against the same set
    masks = [
        list(set(range(
            cochain_train_input[i].shape[-1])) - set(masks_all_deg[i].values()))
        for i in range(top_dim + 1)
    ]
    test_criterion = nn.L1Loss(reduction="sum")
    loss = 0
    ys = network(Ls, Ds, adDs, cochain_train_input)
    for d in range(top_dim + 1):
        loss += test_criterion(ys[d][
            0, 0, masks[d]], cochain_train_target[d][0, 0, masks[d]])
    print(f'Train set loss: {loss}')

    # Now test the model against our test set
    cochain_test_input, cochain_test_target,\
        laplacians, Ls, Ds, adDs = construct_model_inputs(
            path_prefix=path_prefix,
            file_identifier='test_' + file_identifier,
            top_dim=top_dim,
            batch_size=batch_size,
            percentage_missing_values=percentage_missing_values)

    test_masks = np.load('{}/test_{}_percentage_{}_known_values.npy'.format(
            path_prefix,file_identifier,percentage_missing_values),
            allow_pickle=True)
    masks = [
        list(set(range(
            cochain_test_input[i].shape[-1])) - set(test_masks[i].values()))
        for i in range(top_dim + 1)
    ]
    test_criterion = nn.L1Loss(reduction="sum")
    loss = 0
    ys = network(Ls, Ds, adDs, cochain_test_input)
    for d in range(top_dim + 1):
        loss += test_criterion(ys[d][
            0, 0, masks[d]], cochain_test_target[d][0, 0, masks[d]])
    print(f'Test set loss: {loss}')


if __name__ == "__main__":
    train_steps = int(sys.argv[5]) if len(sys.argv) > 5 else 30
    new_hparams = Hparams(
        learning_rate=0.01,
        batch_size=1,
        train_steps=train_steps,
        top_dim=2,
        optimizer=torch.optim.Adagrad,
        loss_criterion=nn.MSELoss(reduction="sum"),
        res_add=True,
    )
    original_hparams = Hparams(
        learning_rate=0.001,
        batch_size=1,
        train_steps=train_steps,
        top_dim=2,
        optimizer=torch.optim.Adam,
        loss_criterion=nn.L1Loss(reduction="sum"),
        res_add=False,
    )
    hparam_set = new_hparams if len(
        sys.argv) > 6 and sys.argv[6] == 'new' else original_hparams
    train_and_test(hparam_set)
