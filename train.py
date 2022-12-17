import os
import argparse
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.distributed as dist
import networkx as nx
import random
import torch_geometric.transforms as T

from tensorboardX import SummaryWriter
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.utils import negative_sampling, convert, to_dense_adj
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.nn.conv import MessagePassing
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator

WORLD_SIZE = int(os.environ.get('WORLD_SIZE', 1))

print(f"WORLD SIZE {WORLD_SIZE}")


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

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x


class DotProductLinkPredictor(torch.nn.Module):
    def __init__(self):
        super(DotProductLinkPredictor, self).__init__()

    def forward(self, x_i, x_j):
        out = (x_i * x_j).sum(-1)
        return torch.sigmoid(out)

    def reset_parameters(self):
        pass


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
    edge_model=False,
    spd=None,
):
    # adj_t isn't used everywhere in PyG yet, so we switch back to edge_index for negative sampling
    row, col, edge_attr = adj_t.t().coo()
    edge_index = torch.stack([row, col], dim=0)

    model.train()
    predictor.train()

    # model.reset_parameters()
    # for conv in model.convs:
    #     conv.reset_parameters()
    # predictor.reset_parameters()

    all_pos_train_edges = split_edge["train"]["edge"]
    epoch_total_loss = 0
    for perm in DataLoader(
        range(all_pos_train_edges.shape[0]), batch_size, shuffle=True
    ):
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
    for perm in DataLoader(range(pos_eval_edge.shape[0]), batch_size):
        edge = pos_eval_edge[perm].t()
        pos_eval_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_eval_pred = torch.cat(pos_eval_preds, dim=0)

    neg_eval_preds = []
    for perm in DataLoader(range(neg_eval_edge.size(0)), batch_size):
        edge = neg_eval_edge[perm].t()
        neg_eval_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    neg_eval_pred = torch.cat(neg_eval_preds, dim=0)

    total_preds = torch.cat((pos_eval_pred, neg_eval_pred), dim=0)
    labels = torch.cat(
        (torch.ones_like(pos_eval_pred), torch.zeros_like(neg_eval_pred)), dim=0
    )
    acc = accuracy(total_preds, labels)

    results = {}
    for K in [10, 20, 30, 40, 50]:
        evaluator.K = K
        valid_hits = evaluator.eval(
            {
                "y_pred_pos": pos_eval_pred,
                "y_pred_neg": neg_eval_pred,
            }
        )[f"hits@{K}"]
        results[f"Hits@{K}"] = valid_hits
    results["Accuracy"] = acc

    # print(acc)
    print(f"Epoch {epoch} has accuracy {round(acc, 4)}")
    return results




def should_distribute():
    print(f"should_distribute = {dist.is_available() and WORLD_SIZE > 1}")
    return dist.is_available() and WORLD_SIZE > 1


def is_distributed():
    print(f"is_distributed = {dist.is_available() and dist.is_initialized()}")
    return dist.is_available() and dist.is_initialized()




def main():
    # parse the setting for training
    parser = argparse.ArgumentParser(description='GNN - Link Prediction - DATA: ogbl-ddi | PyTorch distributed + kubeflow ')
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
    if dist.is_available():
        parser.add_argument('--backend', type=str, help='Distributed backend',
                            choices=[dist.Backend.GLOO, dist.Backend.NCCL, dist.Backend.MPI],
                            default=dist.Backend.GLOO)
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if use_cuda:
        print('Using CUDA')

    writer = SummaryWriter(args.dir)

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    if should_distribute():
        print('Using distributed PyTorch with {} backend'.format(args.backend))
        dist.init_process_group(backend=args.backend)

    # kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    dataset = PygLinkPropPredDataset(name="ogbl-ddi")
    data = dataset[0]
    G = convert.to_networkx(data, to_undirected=True)


    num_nodes, num_edges = G.number_of_nodes(), G.number_of_edges()
    print(
        f"ogbl-ddi has {num_nodes} nodes and {num_edges} edges, with an average node degree of {round(2 * num_edges / num_nodes)}"
    )

    dataset = PygLinkPropPredDataset(name='ogbl-ddi',
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
    # train(
    #     model,
    #     predictor,
    #     emb,
    #     adj_t,
    #     split_edge,
    #     torch.nn.BCELoss(),
    #     optimizer,
    #     64 * 1024,
    #     30,
    # )

    # test(
    #     model,
    #     predictor,
    #     emb,
    #     adj_t,
    #     split_edge["valid"],
    #     Evaluator(name="ogbl-ddi"),
    #     64 * 1024,
    # )

    for epoch in range(1, args.epochs + 1):
        loss = train(args, model, device, predictor, emb, adj_t, split_edge, num_nodes, torch.nn.BCELoss(), optimizer, 64*1024, epoch, writer)
        writer.add_scalar('loss', round(loss, 4), epoch)
        acc = test(args, model, device, predictor, emb, adj_t, split_edge["valid"], Evaluator(name="ogbl-ddi"), 64*1024, epoch, writer)
        writer.add_scalar('accuracy', round(acc["Accuracy"], 4), epoch)

    if (args.save_model):
        torch.save(model.state_dict(),"ogbl-ddi-1.pt")


if __name__ == '__main__':
    main()