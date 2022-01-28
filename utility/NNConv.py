import torch, pdb
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, CGConv, global_max_pool, global_mean_pool, global_add_pool, GMMConv, GatedGraphConv, GraphConv, LEConv, SGConv, TAGConv, TransformerConv
from torch.nn import Linear, LayerNorm, ReLU
import torch_geometric.transforms as T
from torch_geometric.utils import normalized_cut
from torch_geometric.nn import NNConv, graclus, max_pool, max_pool_x

#### edge attr ######
class NNConvNet(nn.Module):
    def __init__(self, args):
        super(NNConvNet, self).__init__()
        self.log_softmax = args['log_softmax']
        in_channels, out_channels = args['node_features_size'], 32

        nn1 = nn.Sequential(nn.Linear(1, 25), nn.ReLU(), nn.Linear(25, in_channels * out_channels))
        self.conv1 = NNConv(in_channels, out_channels, nn1, aggr='max')

        nn2 = nn.Sequential(nn.Linear(1, 25), nn.ReLU(), nn.Linear(25, 2048))
        self.conv2 = NNConv(32, 64, nn2, aggr='max')

        self.fc1 = torch.nn.Linear(64, 128)
        self.fc2 = torch.nn.Linear(128, args['num_classes'])

        self.reset_weights()

    def reset_weights(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, data):
        x = F.elu(self.conv1(data.x, data.edge_index, data.edge_attr))
        x = F.elu(self.conv2(x, data.edge_index, data.edge_attr))
        x = F.elu(self.fc1(x))
        x = self.fc2(x)

        if self.log_softmax:
            return F.log_softmax(x, dim=1)
        else:
            return x
    
    def get_features(self, data):
        x = F.elu(self.conv1(data.x, data.edge_index, data.edge_attr))
        x = F.elu(self.conv2(x, data.edge_index, data.edge_attr))
        x = F.elu(self.fc1(x))
        x = self.fc2(x)
        return x

class NNConvNet_1layer(nn.Module):
    def __init__(self, args):
        super(NNConvNet_1layer, self).__init__()
        self.log_softmax = args['log_softmax']
        in_channels, out_channels = args['node_features_size'], 32

        nn1 = nn.Sequential(nn.Linear(1, 25), nn.ReLU(), nn.Linear(25, in_channels * out_channels))
        self.conv1 = NNConv(in_channels, out_channels, nn1, aggr='max')

        self.fc1 = torch.nn.Linear(32, args['num_classes'])

        self.reset_weights()

    def reset_weights(self):
        self.conv1.reset_parameters()
        torch.nn.init.xavier_uniform_(self.fc1.weight)

    def forward(self, data):
        x = F.elu(self.conv1(data.x, data.edge_index, data.edge_attr))
        x = self.fc1(x)

        if self.log_softmax:
            return F.log_softmax(x, dim=1)
        else:
            return x
    
    def get_features(self, data):
        x = F.elu(self.conv1(data.x, data.edge_index, data.edge_attr))
        x = F.elu(self.conv2(x, data.edge_index, data.edge_attr))
        x = F.elu(self.fc1(x))
        x = self.fc2(x)
        return x

class TransformerConvNet(nn.Module):
    def __init__(self, args):
        super(TransformerConvNet, self).__init__()
        self.log_softmax = args['log_softmax']
        in_channels, out_channels = args['node_features_size'], 32
        edge_dim = 1
        aggregator = "max"
        kernel_size = 3
        K_hops = 3

        self.conv1 = TransformerConv(in_channels, out_channels, edge_dim=edge_dim)

        self.conv2 = TransformerConv(32, 64, edge_dim=edge_dim)

        self.fc1 = torch.nn.Linear(64, 128)
        self.fc2 = torch.nn.Linear(128, args['num_classes'])
        
        self.reset_weights()

    def reset_weights(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        torch.nn.init.kaiming_normal_(self.fc1.weight)
        torch.nn.init.kaiming_normal_(self.fc2.weight)

    def forward(self, data):
        x = F.elu(self.conv1(data.x, data.edge_index, edge_attr=data.edge_attr))
        x = F.elu(self.conv2(x, data.edge_index, edge_attr=data.edge_attr))
        x = F.elu(self.fc1(x))
        
        x = self.fc2(x)
        if self.log_softmax:
            return F.log_softmax(x, dim=1)
        else:
            return x

class GMMConvNet(nn.Module):
    def __init__(self, args):
        super(GMMConvNet, self).__init__()
        self.log_softmax = args['log_softmax']
        in_channels = args['node_features_size'] if args['node_features_size'] > 16 else 16
        edge_dim = 1
        aggregator = "max"
        kernel_size = 3

        self.conv1 = GMMConv(in_channels, in_channels*2, edge_dim, kernel_size=kernel_size, aggr=aggregator)

        self.conv2 = GMMConv(in_channels*2, in_channels*4, edge_dim, kernel_size=kernel_size, aggr=aggregator)

        self.fc1 = torch.nn.Linear(in_channels*4, in_channels*8)
        self.fc2 = torch.nn.Linear(in_channels*8, args['num_classes'])
        
        self.reset_weights()

    def reset_weights(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = F.elu(self.conv1(x, edge_index, edge_attr))
        x = F.elu(self.conv2(x, edge_index, edge_attr))
        x = F.elu(self.fc1(x))
        
        x = self.fc2(x)
        if self.log_softmax:
            return F.log_softmax(x, dim=1)
        else:
            return x



##### edge weight ########
class GCNNet_3conv(torch.nn.Module):
    def __init__(self, args):
        super(GCNNet_3conv, self).__init__()
        self.log_softmax = args['log_softmax']
        in_channels = args['node_features_size']
        out_channels = args['node_features_size'] if args['node_features_size'] > args['GNN_dim'] else args['GNN_dim']
        self.conv1 = GCNConv(in_channels, out_channels*2)
        self.conv2 = GCNConv(out_channels*2, out_channels*4)
        self.conv3 = GCNConv(out_channels*4, out_channels*8)
        self.fc1 = torch.nn.Linear(out_channels*8, out_channels*8)
        self.fc2 = torch.nn.Linear(out_channels*8, args['num_classes'])
        

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.relu(self.conv1(x, edge_index, edge_weight=data.edge_attr.squeeze()))
        # x = F.dropout(x, training=self.training)
        x = F.relu(self.conv2(x, edge_index, edge_weight=data.edge_attr.squeeze()))
        x = F.relu(self.conv3(x, edge_index, edge_weight=data.edge_attr.squeeze()))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)


class GCNNet_2conv(torch.nn.Module):
    def __init__(self, args):
        super(GCNNet_2conv, self).__init__()
        self.log_softmax = args['log_softmax']
        in_channels = args['node_features_size']
        out_channels = args['node_features_size'] if args['node_features_size'] > args['GNN_dim'] else args['GNN_dim']
        self.conv1 = GCNConv(in_channels, out_channels*4)
        # self.conv2 = GCNConv(out_channels*2, out_channels*4)
        self.conv3 = GCNConv(out_channels*4, out_channels*8)
        self.fc1 = torch.nn.Linear(out_channels*8, out_channels*8)
        self.fc2 = torch.nn.Linear(out_channels*8, args['num_classes'])
        

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.relu(self.conv1(x, edge_index, edge_weight=data.edge_attr.squeeze()))
        # x = F.dropout(x, training=self.training)
        x = F.relu(self.conv3(x, edge_index, edge_weight=data.edge_attr.squeeze()))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)

class GCNNet_1conv(torch.nn.Module):
    def __init__(self, args):
        super(GCNNet_1conv, self).__init__()
        self.log_softmax = args['log_softmax']
        in_channels = args['node_features_size']
        out_channels = args['node_features_size'] if args['node_features_size'] > args['GNN_dim'] else args['GNN_dim']
        self.conv1 = GCNConv(in_channels, out_channels*4)
        # self.conv2 = GCNConv(out_channels*2, out_channels*4)
        # self.conv3 = GCNConv(out_channels*4, out_channels*8)
        self.fc1 = torch.nn.Linear(out_channels*4, out_channels*4)
        self.fc2 = torch.nn.Linear(out_channels*4, args['num_classes'])
        

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.relu(self.conv1(x, edge_index, edge_weight=data.edge_attr.squeeze()))
        # x = F.dropout(x, training=self.training)
        # x = F.relu(self.conv3(x, edge_index, edge_weight=data.edge_attr.squeeze()))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)

class GCNNet(torch.nn.Module):
    def __init__(self, args):
        super(GCNNet, self).__init__()
        self.log_softmax = args['log_softmax']
        in_channels = args['node_features_size'] if args['node_features_size'] > 16 else 16
        self.conv1 = GCNConv(in_channels, in_channels*2)
        self.conv2 = GCNConv(in_channels*2, in_channels*4)
        self.fc1 = torch.nn.Linear(in_channels*4, in_channels*8)
        self.fc2 = torch.nn.Linear(in_channels*8, args['num_classes'])
        

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.relu(self.conv1(x, edge_index, edge_weight=data.edge_attr.squeeze()))
        # x = F.dropout(x, training=self.training)
        x = F.relu(self.conv2(x, edge_index, edge_weight=data.edge_attr.squeeze()))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        if self.log_softmax:
            return F.log_softmax(x, dim=1)
        else:
            return x

class GatedGraphConvNet(nn.Module):
    def __init__(self, args):
        super(GatedGraphConvNet, self).__init__()
        self.log_softmax = args['log_softmax']
        in_channels, out_channels = args['node_features_size'], 32
        edge_dim = 1
        aggregator = "max"
        kernel_size = 3
        num_layers = 3

        self.conv1 = GatedGraphConv(out_channels = out_channels, num_layers = num_layers, aggr=aggregator)
        self.conv2 = GatedGraphConv(64, num_layers, aggr=aggregator)

        self.fc1 = torch.nn.Linear(64, 128)
        self.fc2 = torch.nn.Linear(128, args['num_classes'])
        
        self.reset_weights()

    def reset_weights(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, data):
        x = F.elu(self.conv1(data.x, data.edge_index, edge_weight=data.edge_attr.squeeze()))
        x = F.elu(self.conv2(x, data.edge_index, edge_weight=data.edge_attr.squeeze()))
        x = F.elu(self.fc1(x))
        x = self.fc2(x)

        if self.log_softmax:
            return F.log_softmax(x, dim=1)
        else:
            return x

class GraphConvNet(nn.Module):
    def __init__(self, args):
        super(GraphConvNet, self).__init__()
        self.log_softmax = args['log_softmax']
        in_channels, out_channels = args['node_features_size'], 32
        edge_dim = 1
        aggregator = "max"

        self.conv1 = GraphConv(in_channels, out_channels, aggr=aggregator)
        self.conv2 = GraphConv(32, 64, aggr=aggregator)

        self.fc1 = torch.nn.Linear(64, 128)
        self.fc2 = torch.nn.Linear(128, args['num_classes'])
        
        self.reset_weights()

    def reset_weights(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        torch.nn.init.kaiming_normal_(self.fc1.weight)
        torch.nn.init.kaiming_normal_(self.fc2.weight)

    def forward(self, data):
        x = F.elu(self.conv1(data.x, data.edge_index, data.edge_attr))
        x = F.elu(self.conv2(x, data.edge_index, data.edge_attr))
        x = F.elu(self.fc1(x))
        
        x = self.fc2(x)
        if self.log_softmax:
            return F.log_softmax(x, dim=1)
        else:
            return x

class LEConvNet(nn.Module):
    def __init__(self, args):
        super(LEConvNet, self).__init__()
        self.log_softmax = args['log_softmax']
        in_channels, out_channels = args['node_features_size'], 32

        self.conv1 = LEConv(in_channels, out_channels)

        self.conv2 = LEConv(32, 64)

        self.fc1 = torch.nn.Linear(64, 128)
        self.fc2 = torch.nn.Linear(128, args['num_classes'])
        
        self.reset_weights()

    def reset_weights(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        torch.nn.init.kaiming_normal_(self.fc1.weight)
        torch.nn.init.kaiming_normal_(self.fc2.weight)

    def forward(self, data):
        x = F.elu(self.conv1(data.x, data.edge_index, edge_weight=data.edge_attr.squeeze()))
        x = F.elu(self.conv2(x, data.edge_index, edge_weight=data.edge_attr.squeeze()))
        x = F.elu(self.fc1(x))
        
        x = self.fc2(x)
        if self.log_softmax:
            return F.log_softmax(x, dim=1)
        else:
            return x

class TAGConvNet(nn.Module):
    def __init__(self, args):
        super(TAGConvNet, self).__init__()
        self.log_softmax = args['log_softmax']
        in_channels = args['node_features_size']
        out_channels = args['node_features_size'] if args['node_features_size'] > args['GNN_dim'] else args['GNN_dim']
        K_hops = 3
        self.conv1 = TAGConv(in_channels, out_channels*2, K_hops)
        self.conv2 = TAGConv(out_channels*2, out_channels*4, K_hops)
        self.conv3 = TAGConv(out_channels*4, out_channels*8, K_hops)
        self.fc1 = torch.nn.Linear(out_channels*8, out_channels*8)
        self.fc2 = torch.nn.Linear(out_channels*8, args['num_classes'])
        
        self.reset_weights()

    def reset_weights(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        torch.nn.init.kaiming_normal_(self.fc1.weight)
        torch.nn.init.kaiming_normal_(self.fc2.weight)

    def forward(self, data):
        x = F.elu(self.conv1(data.x, data.edge_index, edge_weight=data.edge_attr.squeeze()))
        x = F.elu(self.conv2(x, data.edge_index, edge_weight=data.edge_attr.squeeze()))
        x = F.elu(self.conv3(x, data.edge_index, edge_weight=data.edge_attr.squeeze()))
        x = F.elu(self.fc1(x))
        x = self.fc2(x)

        if self.log_softmax:
            return F.log_softmax(x, dim=1)
        else:
            return x


            
class CGConvNet(nn.Module):
    def __init__(self, args):
        super(CGConvNet, self).__init__()
        self.log_softmax = args['log_softmax']
        in_channels, out_channels = args['node_features_size'], 32
        edge_dim = 1
        aggregator = "max"

        self.conv1 = CGConv(in_channels, edge_dim, aggr=aggregator)

        self.conv2 = CGConv(in_channels, edge_dim,  aggr=aggregator)

        self.fc1 = torch.nn.Linear(in_channels, 128)
        self.fc2 = torch.nn.Linear(128, args['num_classes'])
        
        self.reset_weights()

    def reset_weights(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, data):
        x = F.elu(self.conv1(data.x, data.edge_index, data.edge_attr))
        x = F.elu(self.conv2(x, data.edge_index, data.edge_attr))
        x = F.elu(self.fc1(x))
        
        x = self.fc2(x)
        if self.log_softmax:
            return F.log_softmax(x, dim=1)
        else:
            return x






class LEConvNet_1layer(nn.Module):
    def __init__(self, args):
        super(LEConvNet_1layer, self).__init__()
        self.log_softmax = args['log_softmax']
        in_channels, out_channels = args['node_features_size'], 32

        self.conv1 = LEConv(in_channels, out_channels)

        self.fc1 = torch.nn.Linear(32, args['num_classes'])
        
        self.reset_weights()

    def reset_weights(self):
        self.conv1.reset_parameters()
        torch.nn.init.kaiming_normal_(self.fc1.weight)

    def forward(self, data):
        x = F.elu(self.conv1(data.x, data.edge_index, edge_weight=data.edge_attr.squeeze()))
        x = self.fc1(x)

        if self.log_softmax:
            return F.log_softmax(x, dim=1)
        else:
            return x

class SGConvNet_1layer(nn.Module):
    def __init__(self, args):
        super(SGConvNet_1layer, self).__init__()
        self.log_softmax = args['log_softmax']
        in_channels, out_channels = args['node_features_size'], 32
        K_hops = 3

        self.conv1 = SGConv(in_channels, out_channels, K_hops)

        self.fc1 = torch.nn.Linear(32, args['num_classes'])
        
        self.reset_weights()

    def reset_weights(self):
        self.conv1.reset_parameters()
        torch.nn.init.kaiming_normal_(self.fc1.weight)

    def forward(self, data):
        x = F.elu(self.conv1(data.x, data.edge_index, edge_weight=data.edge_attr.squeeze()))
        
        x = self.fc1(x)
        if self.log_softmax:
            return F.log_softmax(x, dim=1)
        else:
            return x

class SGConvNet(nn.Module):
    def __init__(self, args):
        super(SGConvNet, self).__init__()
        self.log_softmax = args['log_softmax']
        in_channels, out_channels = args['node_features_size'], 32
        edge_dim = 1
        aggregator = "max"
        kernel_size = 3
        K_hops = 3

        self.conv1 = SGConv(in_channels, out_channels, K_hops)

        self.conv2 = SGConv(32, 64, K_hops)

        self.fc1 = torch.nn.Linear(64, 128)
        self.fc2 = torch.nn.Linear(128, args['num_classes'])
        
        self.reset_weights()

    def reset_weights(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        torch.nn.init.kaiming_normal_(self.fc1.weight)
        torch.nn.init.kaiming_normal_(self.fc2.weight)

    def forward(self, data):
        x = F.elu(self.conv1(data.x, data.edge_index, edge_weight=data.edge_attr.squeeze()))
        x = F.elu(self.conv2(x, data.edge_index, edge_weight=data.edge_attr.squeeze()))
        x = F.elu(self.fc1(x))
        
        x = self.fc2(x)
        if self.log_softmax:
            return F.log_softmax(x, dim=1)
        else:
            return x

