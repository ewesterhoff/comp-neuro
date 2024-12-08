import os
import neurogym as ngym
from neurogym.wrappers.reaction_time import ReactionTime
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
import math
import networkx as nx
import matplotlib.pyplot as plt
import torch.optim as optim
import numpy as np
import pickle
import pandas as pd
from itertools import product


class PosWLinear(nn.Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`

    Same as nn.Linear, except that weight matrix is constrained to be non-negative
    """
    __constants__ = ['bias', 'in_features', 'out_features']

    def __init__(self, in_features, out_features, bias=True):
        super(PosWLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        # weight is non-negative
        return F.linear(input, torch.abs(self.weight), self.bias)


class EIRecLinear(nn.Module):
    r"""Recurrent E-I Linear transformation.

    Args:
        hidden_size: int, layer size
        e_prop: float between 0 and 1, proportion of excitatory units
    """
    __constants__ = ['bias', 'hidden_size', 'e_prop']

    def __init__(self, hidden_size, e_prop=0.8, density=1, graph_type='er', ii_connectivity=1, bias=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.e_prop = e_prop
        self.e_size = int(e_prop * hidden_size)
        self.i_size = hidden_size - self.e_size
        self.weight = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        mask = np.tile([1]*self.e_size+[-1]*self.i_size, (hidden_size, 1))
        np.fill_diagonal(mask, 0)
        
        if graph_type == 'ws':
            # Generate Watts-Strogatz graph
            Ge = nx.newman_watts_strogatz_graph(self.e_size, math.ceil(density * self.e_size), 0.05)
            Gi = nx.newman_watts_strogatz_graph(self.i_size, math.ceil(density * self.i_size), 0.05)

        elif graph_type == 'er':
            # Generate Erdos-Renyi graph
            Ge = nx.erdos_renyi_graph(self.e_size, density)
            Gi = nx.erdos_renyi_graph(self.i_size, density)

        # Convert graphs to adjacency matrices
        Ae = nx.to_numpy_array(Ge, dtype=int)
        Ai = nx.to_numpy_array(Gi, dtype=int)

        if not ii_connectivity:
            Ai[:] = 0

        # Initialize the full adjacency matrix
        A = np.zeros((hidden_size, hidden_size), dtype=int)

        # Assign Ae and Ai directly to the corresponding submatrices of A
        A[:self.e_size, :self.e_size] = Ae  # Excitatory region
        A[self.e_size:, self.e_size:] = Ai  # Inhibitory region

        # Generate random connections between excitatory and inhibitory groups
        mask_ei = np.random.rand(self.e_size, self.i_size) < density
        mask_ie = np.random.rand(self.i_size, self.e_size) < density

        # Update A for cross-group connections
        A[:self.e_size, self.e_size:] = mask_ei.astype(int)  # E to I
        A[self.e_size:, :self.e_size] = mask_ie.astype(int)  # I to E

        G = nx.from_numpy_array(A)

        # Plot the graph
        plt.figure(figsize=(10, 8))
        nx.draw(G, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10)
        plt.title("Graph Representation of A")
        # plt.show()
        plt.close('all')

        mask = np.where(A == 0, 0, mask)
        self.mask = torch.tensor(mask, dtype=torch.float32)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(hidden_size))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # Scale E weight by E-I ratio
        if self.i_size == 0:
            self.weight.data[:, :self.e_size] /= self.e_size
        else:
            self.weight.data[:, :self.e_size] /= (self.e_size/self.i_size)
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def effective_weight(self):
        return torch.abs(self.weight) * self.mask

    def forward(self, input):
        # weight is non-negative
        return F.linear(input, self.effective_weight(), self.bias)


class EIRNN(nn.Module):
    """E-I RNN.

    Reference:
        Song, H.F., Yang, G.R. and Wang, X.J., 2016.
        Training excitatory-inhibitory recurrent neural networks
        for cognitive tasks: a simple and flexible framework.
        PLoS computational biology, 12(2).

    Args:
        input_size: Number of input neurons
        hidden_size: Number of hidden neurons

    Inputs:
        input: (seq_len, batch, input_size)
        hidden: (batch, hidden_size)
        e_prop: float between 0 and 1, proportion of excitatory neurons
    """

    def __init__(self, input_size, hidden_size, dt=None,
                 e_prop=0.8, sigma_rec=0, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.e_size = int(hidden_size * e_prop)
        self.i_size = hidden_size - self.e_size
        self.num_layers = 1
        self.tau = 100
        if dt is None:
            alpha = 1
        else:
            alpha = dt / self.tau
        self.alpha = alpha
        self.oneminusalpha = 1 - alpha
        # Recurrent noise
        self._sigma_rec = np.sqrt(2*alpha) * sigma_rec

        # self.input2h = PosWLinear(input_size, hidden_size)
        self.input2h = nn.Linear(input_size, hidden_size)
        self.h2h = EIRecLinear(hidden_size, e_prop=e_prop, **kwargs)

    def init_hidden(self, input):
        batch_size = input.shape[1]
        return (torch.zeros(batch_size, self.hidden_size).to(input.device),
                torch.zeros(batch_size, self.hidden_size).to(input.device))

    def recurrence(self, input, hidden):
        """Recurrence helper."""
        state, output = hidden
        total_input = self.input2h(input) + self.h2h(output)
        state = state * self.oneminusalpha + total_input * self.alpha
        state += self._sigma_rec * torch.randn_like(state)
        output = torch.relu(state)
        return state, output

    def forward(self, input, hidden=None):
        """Propogate input through the network."""
        if hidden is None:
            hidden = self.init_hidden(input)

        output = []
        steps = range(input.size(0))
        for i in steps:
            hidden = self.recurrence(input[i], hidden)
            output.append(hidden[1])

        output = torch.stack(output, dim=0)
        return output, hidden


class Net(nn.Module):
    """Recurrent network model.

    Args:
        input_size: int, input size
        hidden_size: int, hidden size
        output_size: int, output size
        rnn: str, type of RNN, lstm, rnn, ctrnn, or eirnn
    """
    def __init__(self, input_size, hidden_size, output_size, **kwargs):
        super().__init__()

        # Excitatory-inhibitory RNN
        self.rnn = EIRNN(input_size, hidden_size, **kwargs)
        # self.fc = PosWLinear(self.rnn.e_size, output_size)
        self.fc = nn.Linear(self.rnn.e_size, output_size)

    def forward(self, x):
        rnn_activity, _ = self.rnn(x)
        rnn_e = rnn_activity[:, :, :self.rnn.e_size]
        out = self.fc(rnn_e)
        return out, rnn_activity
    



# Instantiate the network and print information
def initialize_train_network(task_type, e_prop, density, dim_ring, hidden_size, graph_type, ii_connectivity):

    # Environment
    if task_type == 'PDMa':
        task = 'PerceptualDecisionMaking-v0'
        timing = {'fixation': ('choice', (50,100,200,400)),
            'stimulus': ('choice', (100,200,400,800)),
            }
    if task_type == 'PDMb':
        task = 'PerceptualDecisionMaking-v0'
        timing = {'fixation': ('choice', (50,100,200,400)),
            'stimulus': ('choice', (100,200,400,800)),
            }
    if task_type == 'MSI':
        task = 'MultiSensoryIntegration-v0'
        timing = {'fixation': ('choice', (50,100,200,400)),
            'stimulus': ('choice', (100,200,400,800)),
            'decision': ('choice', (50,75,100,125)),
            }

    

    kwargs = {'dt': 20, 'timing': timing, 'dim_ring': dim_ring}
    seq_len = 100

    # Make supervised dataset
    dataset = ngym.Dataset(task, env_kwargs=kwargs, batch_size=16,
                        seq_len=seq_len)

    # A sample environment from dataset
    env = dataset.env
    if task_type == 'PDMb':
        env = ReactionTime(env)
    # Visualize the environment with 2 sample trials
    # _ = ngym.utils.plot_env(env, num_trials=2)

    # Network input and output size
    input_size = env.observation_space.shape[0]
    output_size = env.action_space.n

    net = Net(input_size=input_size, hidden_size=hidden_size,
            output_size=output_size, dt=env.dt, sigma_rec=0.15,
            e_prop=e_prop, density=density, graph_type=graph_type, ii_connectivity=ii_connectivity)

    # Use Adam optimizer
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    running_loss = 0
    running_acc = 0
    print_step = 200
    for i in range(5000):
        inputs, labels = dataset()
        inputs = torch.from_numpy(inputs).type(torch.float)
        labels = torch.from_numpy(labels.flatten()).type(torch.long)

        # in your training loop:
        optimizer.zero_grad()   # zero the gradient buffers
        output, activity = net(inputs)

        output = output.view(-1, output_size)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()    # Does the update

        running_loss += loss.item()
        if i % print_step == (print_step - 1):
            running_loss /= print_step
            # print('Step {}, Loss {:0.4f}'.format(i+1, running_loss))
            running_loss = 0

    env.reset(no_step=True)
    env.timing.update({'fixation': ('constant', 500),
                    'stimulus': ('constant', 500)})
    perf = 0
    num_trial = 5000
    activity_dict = {}
    trial_infos = {}
    stim_activity = []  # response for ground-truth 0 and 1
    for _ in range(dim_ring):
        stim_activity.append([])

    for i in range(num_trial):
        env.new_trial()
        ob, gt = env.ob, env.gt
        inputs = torch.from_numpy(ob[:, np.newaxis, :]).type(torch.float)
        action_pred, rnn_activity = net(inputs)

        # Compute performance
        action_pred = action_pred.detach().numpy()
        choice = np.argmax(action_pred[-1, 0, :])
        correct = choice == gt[-1]

        # Log trial info
        trial_info = env.trial
        trial_info.update({'correct': correct, 'choice': choice})
        trial_infos[i] = trial_info

        # Log stimulus period activity
        rnn_activity = rnn_activity[:, 0, :].detach().numpy()
        activity_dict[i] = rnn_activity

        # Compute stimulus selectivity for all units
        # Compute each neuron's response in trials where ground_truth=0 and 1 respectively
        rnn_activity = rnn_activity[env.start_ind['stimulus']: env.end_ind['stimulus']]
        stim_activity[env.trial['ground_truth']].append(rnn_activity)
    
    
    e_size = net.rnn.e_size
    mean_activity = []
    std_activity = []
    for ground_truth in [0, 1]:
        activity = np.concatenate(stim_activity[ground_truth], axis=0)
        mean_activity.append(np.mean(activity, axis=0))
        std_activity.append(np.std(activity, axis=0))

    # Compute d'
    selectivity = (mean_activity[0] - mean_activity[1])
    selectivity /= np.sqrt((std_activity[0]**2+std_activity[1]**2+1e-7)/2)

    ind_sort = np.concatenate((np.argsort(selectivity[:e_size]),
                            np.argsort(selectivity[e_size:])+e_size))
    W = net.rnn.h2h.effective_weight().detach().numpy()
    # Sort by selectivity
    W = W[:, ind_sort][ind_sort, :]
    eigenvalues, eigenvectors = np.linalg.eig(W)
    
    results = {"W": W,
        "evals": eigenvalues,
        "evecs": eigenvectors,
        "performance": np.mean([val['correct'] for val in trial_infos.values()])
    }

    return results



if __name__ == "__main__":

    task_type = 'PDMa'

    e_props = [0.25, 0.5, 0.8]
    densities = [0.01, 0.05, 0.1, 0.5, 1]
    dim_rings = [2, 4, 8, 16]
    hidden_sizes = [25, 50, 75, 100]
    graph_types = ['er', 'ws']
    ii_connectivities = [0, 1]
    iterations = range(7)

    # Nested dictionary to store results
    pickle_file = f'temp_data1/{task_type}_training_results.pkl'
    os.makedirs('temp_data1', exist_ok=True)

    # Check if the file exists
    if os.path.exists(pickle_file):
        with open(pickle_file, 'rb') as f:
            data = pickle.load(f)
    else:
        data = []

    # Iterate over all parameter combinations and iterations
    counter = 0
    for _, e_prop, density, dim_ring, hidden_size, graph_type, ii_conn in product(iterations, e_props, densities, dim_rings, hidden_sizes, graph_types, ii_connectivities):        # Train the network and store the results
        # Train the network and store the results
        params = {
            'e_prop': e_prop,
            'density': density,
            'dim_ring': dim_ring,
            'hidden_size': hidden_size,
            'graph_type': graph_type,
            'ii_conn': ii_conn,
        }
        print(params)
        results = initialize_train_network(task_type, e_prop, density, hidden_size, dim_ring, graph_type, ii_conn)
        data.append((params, results))

        # Save data as a pickle file
        counter += 1
        if counter == 10:
            coutner = 0
            with open(pickle_file, 'wb') as f:
                pickle.dump(data, f)