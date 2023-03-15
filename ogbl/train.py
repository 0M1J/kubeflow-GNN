import os
import pathlib
import argparse
import random
import time
import networkx as nx
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch.distributed as dist # communication 
import torch_geometric.transforms as T
from tensorboardX import SummaryWriter
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.utils import negative_sampling, convert, to_dense_adj
from torch_geometric.nn import SAGEConv
from torch_geometric.nn.conv import MessagePassing
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator

WORLD_SIZE = int(os.environ.get('WORLD_SIZE', 1))

print(f"WORLD SIZE {WORLD_SIZE}")

# PyTorch support and communication primitives for multiprocess parallelism across several computation nodes
# running on one or more machines.


# torch geometric represents the graph data using two things
# 1 the nodes/vertices list (labels)
# 2 edge_index matrix/tensor with 2d dimension
# edge_index example [[0,1,2,3][1,2,3,0]] where 0,1,2,3 are positions in vertices list



# the sage model 
# a stack of sageconv layers from torch_geometric
# two ops concat and aggregate
# aggregate emb of neighbours and then concat with node emb from prev layer
# other option for aggr = mean or max
class SAGE(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers,
        dropout,
        aggr="add",
    ):
        super(SAGE, self).__init__()

        # init convs list and append seq of sageconv layers
        self.convs = torch.nn.ModuleList()
        self.convs.append(
            SAGEConv(in_channels, hidden_channels, normalize=True, aggr=aggr)
        )
        for _ in range(num_layers - 2):
            self.convs.append(
                SAGEConv(hidden_channels, hidden_channels, normalize=True, aggr=aggr)
            )
        self.convs.append(
            SAGEConv(hidden_channels, out_channels, normalize=True, aggr=aggr)
        )

        # add dropout to prevent overfit 
        self.dropout = dropout

    # use this before training to reset the params
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    # the forward pass
    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x


# a simple dotproduct of two node emb to predict the possibility of an edge
class DotProductLinkPredictor(torch.nn.Module):
    def __init__(self):
        super(DotProductLinkPredictor, self).__init__()

    # the non-linear sigmoid func will give prediction
    def forward(self, x_i, x_j):
        out = (x_i * x_j).sum(-1)
        return torch.sigmoid(out)

    def reset_parameters(self):
        pass


# a custom batch creator
# sampling over edges
def create_train_batch(all_pos_train_edges, perm, edge_index, num_nodes, device):
    # First, we get our positive edges, reshaping them to the form (2, hidden_dimension)
    pos_edges = all_pos_train_edges[perm].t().to(device)

    # We then sample the negative edges using PyG functionality
    neg_edges = negative_sampling(
        edge_index, num_nodes=num_nodes, num_neg_samples=perm.shape[0], method="dense"
    ).to(device)

    # Our training batch is just the positive edges concatanted with the negative ones
    train_edge = torch.cat([pos_edges, neg_edges], dim=1)

    # Our labels are all 1 for the positive edges and 0 for the negative ones
    pos_label = torch.ones(
        pos_edges.shape[1],
    )
    neg_label = torch.zeros(
        neg_edges.shape[1],
    )
    train_label = torch.cat([pos_label, neg_label], dim=0).to(device)

    return train_edge, train_label

# training function
# takes the model along with other args and then runs a single epoch
# also uses tensorboard writer for logging customscaler

def train(
    args,
    model,
    device,
    predictor,
    x,
    adj_t,
    split_edge,
    num_nodes,
    loss_fn,
    optimizer,
    batch_size,
    epoch,
    writer,
    local_rank,
    edge_model=False,
    spd=None,
):
    # adj_t isn't used everywhere in PyG yet, so we switch back to edge_index for negative sampling
    # and get the edge_index tensor from the sparse tensor
    row, col, edge_attr = adj_t.t().coo()
    edge_index = torch.stack([row, col], dim=0)

    model.train()
    predictor.train()

    # use this when you want to train over some fix epoch first and then want to predict after that.    
    # model.reset_parameters()
    # for conv in model.convs:
    #     conv.reset_parameters()
    # predictor.reset_parameters()

    # train edge will be positive edges
    all_pos_train_edges = split_edge["train"]["edge"]
    epoch_total_loss = 0

    # batch wise sampling of the edges and training
    # to get random positive edge sample first define the length range and then use random sampler.
    # use the distributed sampler to divide the batch into number of workers (defined by DDP/MPI)
    edge_range = range(all_pos_train_edges.shape[0])
    for perm in DataLoader(
        edge_range, batch_size, shuffle=False, sampler=DistributedSampler(edge_range)
    ):
        print("rank ", local_rank)
        print("batch data to be processed - edge indexes - ", perm)
        
        # reset the gradients
        optimizer.zero_grad()

        train_edge, train_label = create_train_batch(
            all_pos_train_edges, perm, edge_index, num_nodes, device
        )

        # Use the GNN to generate node embeddings
        if edge_model:
            h = model(x, edge_index, spd)
        else:
            h = model(x, adj_t)

        # Get predictions for our batch and compute the loss
        preds = predictor(h[train_edge[0]], h[train_edge[1]])
        loss = loss_fn(preds, train_label)

        epoch_total_loss += loss.item()

        # Update our parameters
        loss.backward()

        # perform normalization to avoid exploding grad
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
        optimizer.step()

    print(f"Epoch {epoch} has loss {round(epoch_total_loss, 4)}")
    return epoch_total_loss
        
        


def accuracy(pred, label):
    pred_rounded = torch.round(pred)
    accu = torch.eq(pred_rounded, label).sum() / label.shape[0]
    accu = round(accu.item(), 4)
    return accu


@torch.no_grad()
def test(
    args,
    model,
    device,
    predictor,
    x,
    adj_t,
    split_edge,
    evaluator,
    batch_size,
    epoch,
    writer,
    edge_model=False,
    spd=None,
):
    model.eval()
    predictor.eval()

    if edge_model:
        # adj_t isn't used everywhere in PyG yet, so we switch back to edge_index
        row, col, edge_attr = adj_t.t().coo()
        edge_index = torch.stack([row, col], dim=0)
        h = model(x, edge_index, spd)
    else:
        h = model(x, adj_t)

    pos_eval_edge = split_edge["edge"].to(device)
    neg_eval_edge = split_edge["edge_neg"].to(device)

    pos_eval_preds = []
    pos_edge_range = range(pos_eval_edge.size(0))
    for perm in DataLoader(pos_edge_range, batch_size, sampler=DistributedSampler(pos_edge_range)):
        edge = pos_eval_edge[perm].t()
        pos_eval_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_eval_pred = torch.cat(pos_eval_preds, dim=0)

    neg_eval_preds = []
    neg_edge_range = range(neg_eval_edge.size(0))
    for perm in DataLoader(neg_edge_range, batch_size, sampler=DistributedSampler(neg_edge_range)):
        edge = neg_eval_edge[perm].t()
        neg_eval_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    neg_eval_pred = torch.cat(neg_eval_preds, dim=0)

    total_preds = torch.cat((pos_eval_pred, neg_eval_pred), dim=0)
    labels = torch.cat(
        (torch.ones_like(pos_eval_pred), torch.zeros_like(neg_eval_pred)), dim=0
    )

    # acc = accuracy(total_preds, labels)

    # use mrr accuracy
    results = {}
    results = evaluator.eval(
        {
            "y_pred_pos": pos_eval_pred,
            "y_pred_neg": neg_eval_pred,
        }
    )

    # mrr - mean reciprocal rank list
    acc = np.mean(results["mrr_list"])
    # print(acc)
    print(f"Epoch {epoch} has accuracy {round(acc, 4)}")
    return results




# to check if the pytorch is using distributed 
def should_distribute():
    print(f"should_distribute = {dist.is_available() and WORLD_SIZE > 1}")
    return dist.is_available() and WORLD_SIZE > 1


# to check if the pytorch is using distributed and initiated
def is_distributed():
    print(f"is_distributed = {dist.is_available() and dist.is_initialized()}")
    return dist.is_available() and dist.is_initialized()




def main():
    # parse the setting for training
    parser = argparse.ArgumentParser(description='GNN - Link Prediction - DATA: ogbl | PyTorch distributed + kubeflow ')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--exp-name', type=str, default="ogbl_example", metavar='E', dest="exp_name",
                        help='experiment name (default: ogbl_example)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--dir', default='logs', metavar='L',
                        help='directory where summary logs are stored')

    print("Check if distributed trianing is possible")
    print("If yes then append backend arg variable from [GLOO, NCLL, MPI]")
    # these backend will act as communication channels between processes in distributed system.
    if dist.is_available():
        parser.add_argument('--backend', type=str, help='Distributed backend',
                            choices=[dist.Backend.GLOO, dist.Backend.NCCL, dist.Backend.MPI],
                            default=dist.Backend.GLOO)
    args = parser.parse_args()

    print("Check if GPU is available or not")
    print("If yes then use CUDA/GPU as device else use CPU as device")
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if use_cuda:
        device = torch.device("cuda")
        print('Using CUDA')
    else:
        device = torch.device("cpu")
        print('Using CPU')

    print("Initiate TensorBoardX summary writer")
    pathlib.Path(f"./tensorboard/logs/{args.exp_name}").mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(f"./tensorboard/logs/{args.exp_name}")

    torch.manual_seed(args.seed)

    print("Check if distributed training is possible")
    print("If yes then init multi-processing using torch.distributed.init_process_group()")
    if should_distribute():
        print('Using distributed PyTorch with {} backend'.format(args.backend))
        dist.init_process_group(backend=args.backend)
        local_rank = torch.distributed.get_rank()
    else: local_rank = 0
    print(f"Local Rank {local_rank}")

    # kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    print("Load data and calculate number of nodes and edges...")
    dataset = PygLinkPropPredDataset(name="ogbl-citation2")
    data = dataset[0]
    G = convert.to_networkx(data, to_undirected=True)

    num_nodes, num_edges = G.number_of_nodes(), G.number_of_edges()
    print(
        f"ogbl has {num_nodes} nodes and {num_edges} edges, with an average node degree of {round(2 * num_edges / num_nodes)}"
    )

    dataset = PygLinkPropPredDataset(name='ogbl-citation2',
                                        transform=T.ToSparseTensor())
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data = dataset[0]
    adj_t = data.adj_t.to(device)
    split_edge = dataset.get_edge_split()
    emb = torch.ones(num_nodes, 1).to(device)


    # Initialize our model and LinkPredictor
    hidden_dimension = 256
    model = SAGE(1, hidden_dimension, hidden_dimension, 5, 0.5).to(device)
    predictor = DotProductLinkPredictor().to(device)
    
    print(f"SAGE model \n {model}")
    print(f"predictor model \n {predictor}")

    if is_distributed():
        Distributor = nn.parallel.DistributedDataParallel if use_cuda else nn.parallel.DistributedDataParallelCPU
        model = Distributor(model)


    # Run our initial "node features" through the GNN to get node embeddings
    model.eval()
    predictor.eval()
    h = model(emb, adj_t)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(predictor.parameters()), lr=0.01
    )

    for epoch in range(1, args.epochs + 1):
        loss = train(args, model, device, predictor, emb, adj_t, split_edge, num_nodes, torch.nn.BCELoss(), optimizer, 64*1024, epoch, writer, local_rank)
        writer.add_scalar(f'{args.exp_name}-ogbl-worker-{local_rank}-loss', round(loss, 4), epoch)
        acc = test(args, model, device, predictor, emb, adj_t, split_edge["valid"], Evaluator(name="ogbl-citation2"), 64*1024, epoch, writer)
        writer.add_scalar(f'{args.exp_name}-ogbl-worker-{local_rank}-accuracy', round(acc["Accuracy"], 4), epoch)
    
    pathlib.Path(f"./tensorboard/models/{args.exp_name}").mkdir(parents=True, exist_ok=True)
    if (args.save_model):
        torch.save(model.state_dict(),f"./tensorboard/models/{args.exp_name}_ogbl.pt")


if __name__ == '__main__':
    """
    Pytorch script for training GNN for link prediction
    using SAGEConv module from torch_geometric as base
    distributing training data using DistributedSampler
    replicating model copy on each worker using DDP
    """
    s_time = time.time()
    print("starting timer ... ")
    main()
    e_time = time.time()
    print("ending timer ... ")
    print(f"Total time taken : {e_time - s_time:.3} second(s) or {(e_time - s_time)/60:.3} minute(s)")