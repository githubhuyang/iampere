import torch
import torch.nn.functional as F
from torch.nn import LayerNorm
import torch_geometric
from torch_geometric.nn.norm import GraphNorm
from torch_geometric.nn.conv.gatv2_conv import GATv2Conv
from torch_geometric.nn.conv.gat_conv import GATConv


class GNNNet1(torch.nn.Module):
    def __init__(self, input_dim, edge_dim, output_dim):
        super(GNNNet1, self).__init__()
        self.gat1 = GATv2Conv(in_channels=input_dim, out_channels=128, edge_dim=edge_dim, heads=2, bias=True, add_self_loops=False) # share_weights=True, 
        self.norm1 = GraphNorm(256)
        self.gat2 = GATv2Conv(in_channels=256, out_channels=128, edge_dim=edge_dim, heads=2, bias=True, add_self_loops=False) # share_weights=True, 
        self.norm2 = GraphNorm(256)
        self.gat3 = GATv2Conv(in_channels=256, out_channels=128, edge_dim=edge_dim, heads=2, bias=True, add_self_loops=False) # share_weights=True, 
        self.norm3 = GraphNorm(256)
        self.gat4 = GATv2Conv(in_channels=256, out_channels=128, edge_dim=edge_dim, heads=2, bias=True, add_self_loops=False) # share_weights=True, 
        self.norm4 = GraphNorm(256)
        self.gat5 = GATv2Conv(in_channels=256, out_channels=128, edge_dim=edge_dim, heads=2, bias=True, add_self_loops=False) # share_weights=True, 
        self.norm5 = GraphNorm(256)
        self.gat6 = GATv2Conv(in_channels=256, out_channels=128, edge_dim=edge_dim, heads=2, bias=True, add_self_loops=False) # share_weights=True, 
        self.norm6 = GraphNorm(256)

        self.l1 = torch.nn.Linear(512 + 2 * input_dim, 256)
        self.l2 = torch.nn.Linear(256, 128)
        self.l3 = torch.nn.Linear(128, output_dim)


    def forward(self, x, edge_index, edge_attr):
        x0 = x
        x1 = self.gat1(x=x, edge_index=edge_index, edge_attr=edge_attr)
        x1 = self.norm1(x1)

        x2 = self.gat2(x=x1, edge_index=edge_index, edge_attr=edge_attr)
        x2 = self.norm2(x2)

        x2 = x1 + x2

        x3 = self.gat3(x=x2, edge_index=edge_index, edge_attr=edge_attr)
        x3 = self.norm3(x3)

        x3 = x2 + x3

        x4 = self.gat4(x=x3, edge_index=edge_index, edge_attr=edge_attr)
        x4 = self.norm4(x4)

        x4 = x3 + x4

        x5 = self.gat5(x=x4, edge_index=edge_index, edge_attr=edge_attr)
        x5 = self.norm5(x5)

        x5 = x4 + x5

        x6 = self.gat6(x=x5, edge_index=edge_index, edge_attr=edge_attr)
        x6 = self.norm6(x6)

        x6 = x5 + x6

        x_from = torch.index_select(x6, dim=0, index=edge_index[0]) # x2/x1
        x_from_0 = torch.index_select(x0, dim=0, index=edge_index[0])
        x_from = torch.cat([x_from, x_from_0], dim=1)

        x_to = torch.index_select(x6, dim=0, index=edge_index[1])   # x2/x1
        x_to_0 = torch.index_select(x0, dim=0, index=edge_index[1])
        x_to = torch.cat([x_to, x_to_0], dim=1)

        assert(x_from.shape[0] == edge_index.shape[1])
        assert(x_to.shape[0] == edge_index.shape[1])

        x = torch.cat([x_from, x_to], dim=1) # edge_attr

        assert(x.shape[0] == edge_index.shape[1])

        x = F.gelu(self.l1(x))
        x = F.gelu(self.l2(x))
        x = torch.sigmoid(self.l3(x))
        return x


class PiApproximationWithNN:
    def __init__(self, lr, cuda=True):
        self.gnn = GNNNet1(3, 6, 1) # 128
        self.gnn.train()

        if cuda:
            self.gnn = self.gnn.cuda()

        self.optimizer = torch.optim.AdamW(self.gnn.parameters(), lr=lr)


    def save(self, fn_path):
        torch.save(self.gnn.state_dict(), fn_path)

    def load(self, fn_path):
        self.gnn.load_state_dict(torch.load(fn_path))

    def load_cpu(self, fn_path):
        self.gnn.load_state_dict(torch.load(fn_path, map_location="cpu"))

    def parameters(self):
        return self.gnn.parameters()

    def get_dist(self, s, action_space):
        p = self.gnn(s.x, s.edge_index, s.edge_attr).flatten()
        
        masks = torch.ones(s.edge_index.shape[1]).cuda()

        ignore_idx_lst = [i for i in range(s.edge_index.shape[1]) if i not in action_space]
        masks[ignore_idx_lst] = 0.0
        p = p * masks

        eps = torch.finfo(torch.float64).eps
        p = torch.clamp(p, min=eps, max=1-eps)
        return p