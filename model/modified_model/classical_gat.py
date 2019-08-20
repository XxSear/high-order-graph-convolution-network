import os.path as osp

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GATConv
from data_loader import Cora
from model_component.utils.mask import index_2_mask

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data = Cora()
data.get_data_loader([2000, 300, 400], mode='numerical', shuffle=True)
nodes = data.data_loader.all_node_feature.to(device)
edges = data.data_loader.edge_index.to(device)
node_num = nodes.size()[0]

all_label = data.data_loader.all_node_label
train_index = data.data_loader.train_index
test_index = data.data_loader.test_index
valid_index = data.data_loader.valid_index

train_mask = index_2_mask(node_num, train_index)
test_mask = index_2_mask(node_num, test_index)
valid_mask = index_2_mask(node_num, valid_index)




class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GATConv(1433, 8, heads=10, dropout=0.6)
        self.conv2 = GATConv(8 * 10, 7, dropout=0.6)

    def forward(self):
        x = F.dropout(nodes, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edges))

        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edges)
        return F.log_softmax(x, dim=-1)


model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)


def train():
    model.train()
    optimizer.zero_grad()
    F.nll_loss(model()[train_index], all_label[train_index]).backward()
    optimizer.step()


def test():
    model.eval()
    logits, accs = model(), []

    for index in [train_index, valid_index, test_index]:
        pred = logits[index].max(1)[1]
        # print(pred.eq(all_label)[index].sum(), float(index.size()[0]))
        acc = pred.eq(all_label[index]).sum() / torch.tensor( float(index.size()[0]) )
        accs.append(acc)
    return accs


for epoch in range(1, 200):
    train()
    log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    print(log.format(epoch, *test()))
    # break

# print(data.train_mask.sum())
# print(all_label[train_mask])
# for i in range(7):
#     print( (all_label[train_mask] == i).sum() )