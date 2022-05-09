import os
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

class MatrixGenerator:
    def __init__(self, input_matrix, batch_size=32, shuffle=True, device=None):
        super().__init__()
        self.input_matrix = input_matrix
        self._num_data = self.input_matrix.shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device
    
    def __len__(self):
        return int(np.ceil(self._num_data / self.batch_size))

    def __iter__(self):
        if self.shuffle:
            perm = np.random.permutation(self._num_data)
        else:
            perm = np.arange(self._num_data, dtype=np.int32)

        for b, st in enumerate(range(0, self._num_data, self.batch_size)):
            ed = min(st + self.batch_size, self._num_data)
            batch_idx = perm[st:ed]
            
            batch_input = torch.tensor(self.input_matrix[batch_idx].toarray(),
                                            dtype=torch.float32, device=self.device)
            
            yield batch_input


class VAE(nn.Module):
    def __init__(self,
                 train_data,
                 rating_mat,
                 q_dims=None,
                 q=0.5,
                 epochs=10,
                 lr=1e-3,
                 reg_1=0.,
                 reg_2=0.,
                 beta=0.5,
                 loss_type='CL',
                 gpuid='0',
                 device='gpu',
                 early_stop=True,
                 optimizer='adam',
                 initializer='xavier_normal'):
        """
        VAE Recommender Class
        Parameters
        ----------
        rating_mat : np.matrix,
        q_dims : List, Q-net dimension list
        q : float, drop out rate
        epochs : int, number of training epochs
        lr : float, learning rate
        reg_1 : float, first-order regularization term
        reg_2 : float, second-order regularization term
        loss_type : str, loss function type
        gpuid : str, GPU ID
        early_stop : bool, whether to activate early stop mechanism
        """
        super(VAE, self).__init__()

        self.epochs = epochs
        self.lr = lr
        self.reg_1 = reg_1
        self.reg_2 = reg_2
        self.beta = beta
        self.loss_type = loss_type
        self.early_stop = early_stop
        self.device = device
        self.optimizer = optimizer
        self.initializer = initializer

        if torch.cuda.is_available() and self.device == 'gpu':
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpuid)
        # torch.cuda.set_device(int(gpuid)) # if internal error, try this code instead

        cudnn.benchmark = True

        self.train_data = train_data
        user_num, item_num = rating_mat.shape
        self.user_num = user_num
        self.item_num = item_num
        self.rating_mat = rating_mat
        self.user_record = set()

        p_dims = [200, 600, item_num]
        self.p_dims = p_dims
        if q_dims:
            assert q_dims[0] == p_dims[-1], "In and Out dimensions must equal to each other"
            assert q_dims[-1] == p_dims[0], "Latent dimension for p- and q- network mismatches."
            self.q_dims = q_dims
        else:
            self.q_dims = p_dims[::-1]
        
        # Last dimension of q- network is for mean and variance
        tmp_q_dims = self.q_dims[:-1] + [self.q_dims[-1] * 2]  # essay setting only focus on 2 encoder
        self.q_layers = nn.ModuleList(
            [nn.Linear(d_in, d_out) for d_in, d_out in zip(tmp_q_dims[:-1], tmp_q_dims[1:])]
        )
        self.p_layers = nn.ModuleList(
            [nn.Linear(d_in, d_out) for d_in, d_out in zip(self.p_dims[:-1], self.p_dims[1:])]
        )
        self.drop = nn.Dropout(q)
        self._init_weights()

        self.prediction = None

    def _init_weights(self):

        for layer in self.q_layers:
            # Xavier Initialization for weights
            if self.initializer == 'xavier_normal':
                size = layer.weight.size()
                fan_out = size[0]
                fan_in = size[1]
                std = np.sqrt(2.0/(fan_in + fan_out))
                layer.weight.data.normal_(0.0, std)
            elif self.initializer == 'xavier_uniform':
                nn.init.xavier_uniform_(layer.weight)
            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)
        
        for layer in self.p_layers:
            # Xavier Initialization for weights
            if self.initializer == 'xavier_normal':
                size = layer.weight.size()
                fan_out = size[0]
                fan_in = size[1]
                std = np.sqrt(2.0/(fan_in + fan_out))
                layer.weight.data.normal_(0.0, std)
            elif self.initializer == 'xavier_uniform':
                nn.init.xavier_uniform_(layer.weight)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)
    
    def encode(self, input):
        h = F.normalize(input)
        h = self.drop(h)
        for i, layer in enumerate(self.q_layers):
            h = layer(h)
            if i != len(self.q_layers) - 1:
                h = F.tanh(h)
            else:
                mu = h[:, :self.q_dims[-1]]  
                logvar = h[:, self.q_dims[-1]:]
        
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)  # core calculation for predicting the real distribution
        else:
            return mu

    def decode(self, z):
        h = z
        for i, layer in enumerate(self.p_layers):
            h = layer(h)
            if i != len(self.p_layers) - 1:
                h = F.tanh(h)
        return h

    def forward(self, input):
        mu, logvar = self.encode(input)
        z = self.reparameterize(mu, logvar)
        z = self.decode(z)
        return z, mu, logvar

    def fit(self, batch_size):
        if torch.cuda.is_available() and self.device=='gpu':
            self.cuda()
        else:
            self.cpu()

        if self.loss_type == 'CL':
            criterion = nn.BCEWithLogitsLoss(reduction='mean')
        elif self.loss_type == 'SL':
            criterion = nn.MSELoss(reduction='mean')
        else:
            raise ValueError('Invalid loss type')

        if self.optimizer == 'adam':
            optimizer = optim.Adam(self.parameters(), lr=self.lr)
        elif self.optimizer == 'sgd':
            optimizer = optim.SGD(self.parameters(), lr=self.lr)

        last_loss = 0.
        train_matrix = self.train_data
        num_training = train_matrix.shape[0]
        num_batches = int(np.ceil(num_training / batch_size))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch_generator = MatrixGenerator(train_matrix, batch_size=batch_size, shuffle=True, device=device)

        for epoch in range(1, self.epochs + 1):
            self.train()
            current_loss = 0.
            pbar = tqdm(batch_generator)
            pbar.set_description(f'[Epoch {epoch:03d}]')
            for _, batch_matrix in pbar:
                self.zero_grad()
                pred, mu, logvar = self.forward(batch_matrix)
                loss = criterion(pred * batch_matrix, batch_matrix)
                KLD = -self.beta * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
                loss += KLD

                for layer in self.q_layers:
                    loss += self.reg_1 * layer.weight.norm(p=1)
                    loss += self.reg_2 * layer.weight.norm()
                for layer in self.p_layers:
                    loss += self.reg_1 * layer.weight.norm(p=1)
                    loss += self.reg_2 * layer.weight.norm()

                if torch.isnan(loss):
                    raise ValueError(f'Loss=Nan or Infinity: current settings does not fit the recommender')
                
                loss.backward()
                optimizer.step()
                
                pbar.set_postfix(loss=loss.item())
                current_loss += loss.item()

            self.eval()
            delta_loss = float(current_loss - last_loss)
            if (abs(delta_loss) < 1e-5) and self.early_stop:
                print('Satisfy early stop mechanism')
                break
            else:
                last_loss = current_loss


        
        # since there is not enough GPU memory to calculate, so we divide the data into batches, and then calculate them.
        print("calculate result")
        row_size = self.rating_mat.shape[0]
        row_batch_size = 2048 #100
        for i in range(row_size // row_batch_size + 1):
            tmp = self.rating_mat[i * row_batch_size : (i + 1) * row_batch_size, :].A.squeeze()
            tmp = torch.tensor(tmp).float()

            if torch.cuda.is_available() and self.device=='gpu':
                tmp = tmp.cuda()
            else:
                tmp = tmp.cpu()
            tmp_pred = self.forward(tmp)[0]
            tmp_pred.clamp_(min=0, max=5)
            tmp_pred = tmp_pred.cpu().detach().numpy()

            if i == 0:
                self.prediction = tmp_pred
            else:
                self.prediction = np.vstack((self.prediction, tmp_pred))
        print("vae result complete")


    def predict(self, u, i):
        return self.prediction[u, i]
