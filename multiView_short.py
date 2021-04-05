import pandas as pd
import numpy as np
import ciso8601
import os
import random as rand
import datetime as dt
from datetime import datetime
import multiprocessing as mp
from sklearn import preprocessing
import networkx as nx
from scipy import sparse
import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from deeprobust.graph import utils
from copy import deepcopy
from sklearn.metrics import f1_score
##### Utils import
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
import torch.sparse as ts
import warnings
import csv
import argparse
import time
import pickle
#### newly added in multi_view_new.py
from torch.utils.data import Dataset, DataLoader, TensorDataset
import psutil

G1_DATA_FILE = '../../thorgeir/inputData/large/g1calls.csv'
LABEL_FILE = '../data/healthy_sick_pair_uid_mapping.csv'
holdout_set_FILE = '../holdout/holdout_set.pkl'
viewsAndLabels_subset_FILE = '../pkl_files/viewsAndLabels_anchor50_k3_20090201To20100201_NoSelfLoop_localVar.pkl'
train_loss_data_FILE = '../pkl_files/multiView_train_loss_subset_batch.pkl'
val_loss_data_FILE = '../pkl_files/multiView_val_loss_subset_batch.pkl'
test_loss_data_FILE = '../pkl_files/multiView_test_loss_subset_batch.pkl'
lr_data_FILE = '../pkl_files/multiView_lr_subset_batch.pkl'
RAM_data_FILE = '../pkl_files/multiView_RAM_subset_batch.pkl'
networkSubset_FILE = '../pkl_files/networkSubset_nodeID.pkl'# list of node id's 20100
#anchor_set_FILE = '../pkl_files/anchor_49_1_nodeID.pkl'# list of node id's

# This is data loader -- you can ignore this class

class DealDataset(Dataset):

    def __init__(self, start_date, end_date, lap, multi_view, anchor_size, k_neighbor, selectSize, reload_data=False):

        super(DealDataset, self).__init__()
        
        if reload_data:
            # uncomment to load new datasets
            self.x_data, self.y_data = loadData(start_date, end_date, lap, multi_view, anchor_size, k_neighbor)
            a_file = open(viewsAndLabels_subset_FILE, "wb")
            pickle.dump([self.x_data, self.y_data], a_file)
            a_file.close()
            print("New viewsAndLabels_subset_FILE saved")
        
        a_file = open(viewsAndLabels_subset_FILE, "rb")
        data = pickle.load(a_file)
        a_file.close()
        
        total_tuple_idx = list(range(0, len(data[1])))
        selected_tuple_idx = rand.sample(total_tuple_idx, selectSize)
        self.x_data = [data[0][i] for i in selected_tuple_idx]
        self.y_data = [data[1][i] for i in selected_tuple_idx]
        
#        print("len(self.y_data)", len(self.y_data))
#        print("Head of labels:", self.y_data[0:5])

        
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len
     
# This is model layer class

class InteractiveGraphConvolution(Module):

    def __init__(self, in_features, out_features, with_bias=True):
        super(InteractiveGraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_self = Parameter(torch.FloatTensor(in_features, out_features))
        self.weight_view2 = Parameter(torch.FloatTensor(in_features, out_features))
        self.weight_view3 = Parameter(torch.FloatTensor(in_features, out_features))
        self.weight_all_views = Parameter(torch.FloatTensor(1,3))
        if with_bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # self.weight.data.fill_(1)
        # if self.bias is not None:
        #     self.bias.data.fill_(1)

        stdv = 1. / math.sqrt(self.weight_self.size(1))
        self.weight_self.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.weight_view2.size(1))
        self.weight_view2.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.weight_view3.size(1))
        self.weight_view3.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.weight_all_views.size(1))
        self.weight_all_views.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, self_input, self_adj, view2_input, view2_adj, view3_input, view3_adj):
        """ Graph Convolutional Layer forward function
        """
        # one self embedding
#        print("shapes")
#        print(self_input.shape,self.weight_self.shape,view2_input.shape, self.weight_view2.shape,view3_input.shape, self.weight_view3.shape)
        self_embedding = torch.spmm(self_input, self.weight_self)
        view2_embedding = torch.spmm(view2_input, self.weight_view2)
        view3_embedding = torch.spmm(view3_input, self.weight_view3)
        
        # three neighborhood embedding
        self_neighberhood_embedding = torch.spmm(self_adj, self_embedding) # all adj does not contain self loops
        view2_neighberhood_embedding = torch.spmm(view2_adj, view2_embedding)
        view3_neighberhood_embedding = torch.spmm(view3_adj, view3_embedding)
        
        neighbor_embds = torch.cat((self_neighberhood_embedding.unsqueeze(1), view2_neighberhood_embedding.unsqueeze(1), view3_neighberhood_embedding.unsqueeze(1)), 1)
        
        agg_neighberhood_embedding = torch.squeeze(torch.matmul(self.weight_all_views, neighbor_embds))
        
        output = self_embedding + 1.01 * agg_neighberhood_embedding
        
        if self.bias is not None:
            output = output + self.bias
            
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

# This is the model

class Embedding(nn.Module):
    """ 2 Layer Graph Convolutional Network.

    Parameters
    ----------
    nfeat : int
        size of input feature dimension
    nhid : int
        number of hidden units
    nclass : int
        size of output dimension
    dropout : float
        dropout rate for GCN
    lr : float
        learning rate for GCN
    weight_decay : float
        weight decay coefficient (l2 normalization) for GCN. When `with_relu` is True, `weight_decay` will be set to 0.
    with_relu : bool
        whether to use relu activation function. If False, GCN will be linearized.
    with_bias: bool
        whether to include bias term in GCN weights.
    device: str
        'cpu' or 'cuda'.
    """
    def __init__(self, nfeat, nhid, nclass, dropout=0.4, lr=0.0001, weight_decay=5e-4, with_relu=True, with_bias=True, device=None):
    
        super(Embedding, self).__init__()
        
        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.nfeat = nfeat
        self.hidden_sizes = [nhid] # ?
        self.nclass = nclass

        self.gc1_1 = InteractiveGraphConvolution(nfeat, nhid, with_bias=with_bias)
        self.gc1_2 = InteractiveGraphConvolution(nfeat, nhid, with_bias=with_bias)
        self.gc1_3 = InteractiveGraphConvolution(nfeat, nhid, with_bias=with_bias)
        self.gc2_1 = InteractiveGraphConvolution(nhid, nclass, with_bias=with_bias)
        self.gc2_2 = InteractiveGraphConvolution(nhid, nclass, with_bias=with_bias)
        self.gc2_3 = InteractiveGraphConvolution(nhid, nclass, with_bias=with_bias)


        self.dropout = dropout
        self.lr = lr
        if not with_relu:
            self.weight_decay = 0
        else:
            self.weight_decay = weight_decay
        self.with_relu = with_relu
        self.with_bias = with_bias
        self.output = None
        self.best_model = None
        self.best_output = None
        self.views = None
        self.loss = nn.MSELoss()
        

    def initialize(self):
        """Initialize parameters of GCN.
        """
        self.gc1_1.reset_parameters()
        self.gc1_2.reset_parameters()
        self.gc1_3.reset_parameters()
        self.gc2_1.reset_parameters()
        self.gc2_2.reset_parameters()
        self.gc2_3.reset_parameters()
        
    def forward(self, VIEWS, Batch=True):
        
        output = []
        
        for views in VIEWS:
            x = self.pred_forward(views, dropout=True)
            output.append(x)
        
        output = torch.cat(output, dim=0) # concat all outputs
        
        return output
        
        
    def pred_forward(self, views, dropout=False):
        
        x1 = F.relu(self.gc1_1(views[0][0], views[0][1], views[1][0], views[1][1], views[2][0], views[2][1]))
        x2 = F.relu(self.gc1_2(views[1][0], views[1][1],views[0][0], views[0][1],views[2][0], views[2][1]))
        x3 = F.relu(self.gc1_3(views[2][0], views[2][1],views[0][0], views[0][1],views[1][0], views[1][1]))
        if dropout: # no dropout for prediction task
            x1 = F.dropout(x1, self.dropout, training=self.training) # N*16
            x2 = F.dropout(x2, self.dropout, training=self.training)
            x3 = F.dropout(x3, self.dropout, training=self.training)
            
        x1_1 = F.relu(self.gc2_1(x1, views[0][1], x2, views[1][1], x3, views[2][1]))
        x2_1 = F.relu(self.gc2_2(x2, views[1][1], x1, views[0][1], x3, views[2][1]))
        x3_1 = F.relu(self.gc2_3(x3, views[2][1], x1, views[0][1], x2, views[1][1])) # N*16
        
        x = (x1_1 + x2_1 + x3_1) / 3
        output = torch.abs(x)
        
        return output


    def fit(self, views, labels, idx_train=None, idx_val=None, train_iters=25, initialize=False, verbose=True, normalize=True, patience=1000, **kwargs):
        
        print("initialize =", initialize)
        
        self.device = self.gc1_1.weight_self.device
        if initialize:
            self.initialize()

        self.views = views
        self.labels = labels

        loss_train = self._train_without_val(labels, train_iters, verbose)

        return loss_train
        
    def _train_without_val(self, labels, train_iters, verbose):

        self.train()
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        for i in range(train_iters): # epoch
            optimizer.zero_grad()
            
            #############################################################################
            ##################      WATCH HERE                        ###################
            #############################################################################
            # The following line eats a lot of memory
            output = self.forward(self.views)# should be a list of sets of views

            loss_train = self.loss(output, torch.cat(labels, dim=0)) # use cat to convert a list of tensors to tensor
            loss_train.backward()
            optimizer.step()
            
            if verbose and i % 5 == 0:
                print('Epoch {}, training loss: {:.4e}'.format(i, loss_train.item()))

        self.eval()
        output = self.forward(self.views)
        self.output = output

        loss_train = self.loss(output, torch.cat(labels, dim=0)) # mse loss
        
        return loss_train

    def predict(self, views):
    
        return self.pred_forward(views)
        
# This is a class that is used to train Embedding model

class GlobalVar(nn.Module):
    def __init__(self, Embedding_nfeat, Embedding_nhid, Embedding_nclass, lr, batch_size, Embedding_device=None):
    
        super(GlobalVar, self).__init__()
        
        assert Embedding_device is not None, "Please specify 'device'!"
        self.device = Embedding_device
        self.Embedding_nfeat = Embedding_nfeat
        self.Embedding_nhid = Embedding_nhid
        self.Embedding_nclass = Embedding_nclass
        self.batch_size = batch_size

        self.VIEWS = None
        self.LABELS = None
        self.Embedding_model = Embedding(self.Embedding_nfeat, self.Embedding_nhid, self.Embedding_nclass, device=self.device, lr=lr)
        self.loss = nn.MSELoss() # mse loss
        self.weights = None
        self.train_loss_data = None
        self.val_loss_data = None
        self.lr_data = None
        self.RAM_data = None
        self.initialize = True
        self.inner_nEpoch = None
        
    def forward(self, idx_train):
        # this is one epoch
        batch_size = self.batch_size
        batch_cnt = len(idx_train)//batch_size
        start_i = 0
        train_loss_ls = []
        
        for i in range(batch_cnt):
        
            if (self.initialize == True and i == 0):
                self.initialize = True
            else:
                self.initialize = False
            
            print("Batch", i+1)
            
            # get batch index
            batch_idx = idx_train[start_i:start_i+batch_size]
            start_i += batch_size
            
            VIEWS_batch = [self.VIEWS[i] for i in batch_idx]
            LABELS_batch = [self.LABELS[i] for i in batch_idx]
            
            #############################################################################
            ##################      WATCH HERE                        ###################
            #############################################################################
            # fit a batch of views
            loss_train = self.Embedding_model.fit(VIEWS_batch, LABELS_batch, train_iters=self.inner_nEpoch, initialize=self.initialize) # this line eats a lot of RAM

            train_loss_ls.append(loss_train)
            
        weights = deepcopy(self.Embedding_model.state_dict())
        
        return weights, train_loss_ls

    def fit(self, VIEWS, LABELS, idx_train, idx_val, train_iters=50, inner_nEpoch=1, verbose=True, normalize=True, **kwargs):
        print("Fitting data. Might take some time. :)")
        self.inner_nEpoch = inner_nEpoch
        # to_tensor
        for i in range(len(LABELS)):
            if type(LABELS[i]) is not torch.Tensor:
                LABELS[i] = LABELS[i].to_numpy().T[1:3].T
#                print("LABELS[i].shape: ", LABELS[i].shape)
                LABELS[i] = torch.FloatTensor(LABELS[i]).to(self.device)
            for j in range(len(VIEWS[i])): # go through each view in this tuple
                if sp.issparse(VIEWS[i][j][1]): # adj
                    VIEWS[i][j][1] = sparse_mx_to_torch_sparse_tensor(VIEWS[i][j][1]).to(self.device)
                else:
                    VIEWS[i][j][1] = torch.FloatTensor(VIEWS[i][j][1]).to(self.device)
                if sp.issparse(VIEWS[i][j][0]): # features
                    VIEWS[i][j][0] = sparse_mx_to_torch_sparse_tensor(VIEWS[i][j][0]).to(self.device)
                else:
                    VIEWS[i][j][0] = torch.FloatTensor(VIEWS[i][j][0]).to(self.device)
                if normalize:
                    if is_sparse_tensor(VIEWS[i][j][1]):
                        VIEWS[i][j][1] = normalize_adj_tensor(VIEWS[i][j][1], sparse=True)
                    else:
                        VIEWS[i][j][1] = normalize_adj_tensor(VIEWS[i][j][1])

                                    
        self.VIEWS = VIEWS
        self.LABELS = LABELS

        self._train_with_val(idx_train, idx_val, train_iters, verbose)

    def _train_with_val(self, idx_train, idx_val, train_iters, verbose):
        
        if verbose:
            print('=== training GlobalVar model ===')

        start_time = time.time()
    
        views_val = [self.VIEWS[i] for i in idx_val]
        lables_val = [self.LABELS[i] for i in idx_val]
        
        best_loss_val_avg = 10**9
        train_loss_data = []
        val_loss_data = []
        lr_data = []
        RAM_data = []
        
        fst_lr_shrink = True
        snd_lr_shrink = True
        
        for i in range(train_iters):
            if i == 0:
                self.initialize = True
            else:
                self.initialize = False
            print("****** GlobalVar Epoch {} ******".format(i+1))
            epoch_start_time = time.time()
            
            rand.shuffle(idx_train) # shuffle idx_train (inplace)

            weights, train_loss_ls = self.forward(idx_train)

            train_loss_data.append(train_loss_ls)
            self.Embedding_model.load_state_dict(weights)
            
            loss_val = 0
            for j in range(len(idx_val)):
                output = self.Embedding_model.predict(views_val[j])
                loss_val += self.loss(output, lables_val[j]).item()

            loss_val_avg = loss_val/len(idx_val)
            
            print("loss_val_avg: ", loss_val_avg)
            
            cpu_used, ram_used = print_used_memory()
            val_loss_data.append(loss_val_avg)
            lr_data.append(self.Embedding_model.lr)
            RAM_data.append([cpu_used, ram_used])
            
            print("self.Embedding_model.lr", self.Embedding_model.lr)
            
            if (fst_lr_shrink and loss_val_avg < 10**3):
                print("First Update lr")
                self.Embedding_model.lr /= 10
                print("New lr: ", self.Embedding_model.lr)
                fst_lr_shrink = False

            if (snd_lr_shrink and loss_val_avg < 10**2):
                print("Second Update lr")
                self.Embedding_model.lr /= 10
                print("New lr: ", self.Embedding_model.lr)
                snd_lr_shrink = False
                
            
            if best_loss_val_avg > loss_val_avg:
                best_loss_val_avg = loss_val_avg
                
                print("Updata best_loss_val_avg: ", best_loss_val_avg)
                
                weights = deepcopy(self.Embedding_model.state_dict())
                
            time1 = (time.time() - start_time) / 60
            time2 = (time.time() - epoch_start_time) / 60
            epoch_remain = train_iters-i-1
            print("Total time past: %.4s mins." % time1, "Epoch time: %.4s mins." % time2)
            print("Might need {:.2f} more mins.".format(float(epoch_remain)*time2),"{} more epochs to go.".format(epoch_remain))
        
            
        if verbose:
            print('=== picking the best model according to the performance on validation ===')
        self.weights = weights
        self.Embedding_model.load_state_dict(weights)
        
        self.train_loss_data = train_loss_data
        self.val_loss_data = val_loss_data
        self.lr_data = lr_data
        self.RAM_data = RAM_data
        
    def predict(self, VIEWS):
    
        output = []
        for i in range(len(VIEWS)):
            output.append(self.Embedding_model.predict(VIEWS[i]))
            
        return output
        
    def test(self, idx_test):

        self.Embedding_model.load_state_dict(self.weights)
        
        views_test = [self.VIEWS[i] for i in idx_test]
        lables_test = [self.LABELS[i] for i in idx_test]
        
        OUTPUT = self.predict(views_test) # OUTPUT is a list of tensor-type output
        
        loss = []
        for i in range(len(idx_test)):
            loss.append(self.loss(OUTPUT[i], lables_test[i]).item())
            
#        print("Prediction: ", torch.cat(OUTPUT).tolist())
#        print("Ground Truth: ", torch.cat(lables_test).tolist())
        print("Loss:", loss)
            
        return loss, OUTPUT, lables_test


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_date', type=str, default="2009,2,1,0,0,0", help='start date of network')
    parser.add_argument('--end_date', type=str, default="2010,2,1,0,0,0", help='end date of network')
    parser.add_argument('--bin', type=float, default=7, help='daily network range')
    parser.add_argument('--lap', type=float, default=1, help='daily network lap')
    parser.add_argument('--nhid', type=int, default=16, help='hidden layer size')
    parser.add_argument('--nEpoch', type=int, default=50)
    parser.add_argument('--inner_nEpoch', type=int, default=1, help='num of epochs for each embedding model')
    parser.add_argument('--multi_view', nargs='+', type=int, default=[7,3,1], help='multi view bin')
    parser.add_argument('--anchor_size', type=int, default=50)
    parser.add_argument('--k_neighbor', type=int, default=3)
    parser.add_argument('--nTrain', type=int, default=200)
    parser.add_argument('--nVal', type=int, default=50)
    parser.add_argument('--nTest', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--reload_data', action='store_true')
    parser.add_argument('--loss_FILE_suffix', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=4)
    args = parser.parse_args()
    
    print("Parameters: ", '--nhid', args.nhid, '--nEpoch', args.nEpoch, '--inner_nEpoch', args.inner_nEpoch, '--multi_view', args.multi_view, '--anchor_size', args.anchor_size, '--k_neighbor', args.k_neighbor, '--nTrain', args.nTrain, '--nVal', args.nVal, '--nTest', args.nTest, '--lr', args.lr, '--start_date', args.start_date,'--end_date', args.end_date,'--bin', args.bin,'--lap', args.lap, '--reload_data', args.reload_data, '--batch_size', args.batch_size, '--loss_FILE_suffix', args.loss_FILE_suffix)
    
    start_time = datetime.now()
    
    Data = DealDataset(args.start_date, args.end_date, args.lap, args.multi_view, args.anchor_size, args.k_neighbor, selectSize=args.nTrain+args.nVal+args.nTest, reload_data=args.reload_data)
    views, labels = Data.x_data, Data.y_data
    del Data
        
    print("len(views): ", len(views))
    print("len(labels): ", len(labels))
#    print("Features include: ", 'tariftype', 'call_count', 'total_call_len', 'lat', 'lon', 'unique_locations_visited','avg_call_len','sub_or_ob','missing')
       

    nfeat = views[0][0][0].shape[1]
    nhid = args.nhid
    nclass = labels[0].shape[1]-1 # labels contain many df, each df has shape N*3. # local vars = 2. "-1" because the first column only indicate the node new_id, so delete.
    print("nfeat =", nfeat)
    print("nclass =", nclass)
    print("==================== Start training model ====================")
    
    # 请从这里开始向下看

    model = GlobalVar(nfeat, nhid, nclass, args.lr, args.batch_size, Embedding_device=torch.device("cpu"))
    train_idx, val_idx, test_idx = train_val_test_split(len(views), args.nTrain, args.nVal, args.nTest)

    model.fit(views, labels, idx_train = train_idx, idx_val = val_idx, train_iters=args.nEpoch, inner_nEpoch = args.inner_nEpoch)
    loss, prediction_tensor, groud_truth_tensor = model.test(test_idx)
    
    # save loss files
    global train_loss_data_FILE,val_loss_data_FILE,test_loss_data_FILE, lr_data_FILE, RAM_data_FILE
    train_loss_data_FILE = train_loss_data_FILE[:-4] + args.loss_FILE_suffix + '.pkl'
    val_loss_data_FILE = val_loss_data_FILE[:-4] + args.loss_FILE_suffix + '.pkl'
    test_loss_data_FILE = test_loss_data_FILE[:-4] + args.loss_FILE_suffix + '.pkl'
    lr_data_FILE = lr_data_FILE[:-4] + args.loss_FILE_suffix + '.pkl'
    RAM_data_FILE = RAM_data_FILE[:-4] + args.loss_FILE_suffix + '.pkl'
    a_file = open(train_loss_data_FILE, "wb")
    pickle.dump(model.train_loss_data, a_file)
    a_file.close()
    a_file = open(val_loss_data_FILE, "wb")
    pickle.dump(model.val_loss_data, a_file)
    a_file.close()
    a_file = open(test_loss_data_FILE, "wb")
    pickle.dump([loss, prediction_tensor, groud_truth_tensor], a_file)
    a_file.close()
    a_file = open(lr_data_FILE, "wb")
    pickle.dump(model.lr_data, a_file)
    a_file.close()
    a_file = open(RAM_data_FILE, "wb")
    pickle.dump(model.RAM_data, a_file)
    a_file.close()
    
    print("==================== Everything is Done ====================")
    print("Outputs are :", viewsAndLabels_subset_FILE, "\n",
    train_loss_data_FILE, "\n",
    val_loss_data_FILE, "\n",
    test_loss_data_FILE, "\n",
    lr_data_FILE, "\n",
    RAM_data_FILE)
    
    print("Parameters: ", '--nhid', args.nhid, '--nEpoch', args.nEpoch, '--inner_nEpoch', args.inner_nEpoch, '--multi_view', args.multi_view, '--anchor_size', args.anchor_size, '--k_neighbor', args.k_neighbor, '--nTrain', args.nTrain, '--nVal', args.nVal, '--nTest', args.nTest, '--lr', args.lr, '--start_date', args.start_date,'--end_date', args.end_date,'--bin', args.bin,'--lap', args.lap, '--reload_data', args.reload_data, '--batch_size', args.batch_size, '--loss_FILE_suffix', args.loss_FILE_suffix)
    
    now = datetime.now()
    start_time_str = start_time.strftime("%d/%m/%Y %H:%M:%S")
    now_str = now.strftime("%d/%m/%Y %H:%M:%S")
    print("The whole rocess starts from [{}] to [{}]".format(start_time_str, now_str))
        
if __name__ == "__main__":
    main()
