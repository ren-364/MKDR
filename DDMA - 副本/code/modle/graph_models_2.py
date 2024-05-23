import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect

from torch_geometric.utils import softmax, add_self_loops
from torch_geometric.nn.inits import glorot, zeros
from torch_scatter import scatter_add
from build_tree import build_stage_one_edges

from util import bigru_attention, BiLSTM_Attention

class OntologyEmbedding(nn.Module):
    def __init__(self, voc, build_tree_func, med_adj=None, ddi_adj=None,
                 in_channels=128, out_channels=32, heads=4):
        super(OntologyEmbedding, self).__init__()

        # initial tree edges
        res, graph_voc = build_tree_func(list(voc.idx2word.values()))
        stage_one_edges = build_stage_one_edges(res, graph_voc)


        self.edges1 = torch.tensor(stage_one_edges)
        self.graph_voc = graph_voc

        # construct model
        assert in_channels == heads * out_channels

        # tree embedding
        num_nodes = len(graph_voc.word2idx)
        self.embedding = nn.Parameter(torch.Tensor(num_nodes, in_channels))

        # idx mapping: FROM leaf node in graphvoc TO voc
        self.idx_mapping = [[graph_voc.word2idx[key] for key in sublist] for sublist in res]

        self.init_params()
        self.biLSTM_Attention = BiLSTM_Attention(in_channels, in_channels, in_channels)

    def get_all_graph_emb(self):
        emb = self.embedding
        emb = self.g(self.g(emb, self.edges1.to(emb.device)),
                     self.edges2.to(emb.device))
        return emb

    def forward(self):
        """
        :param idxs: [N, L]
        :return:
        """
        emb = self.embedding
        final_emb, _ = self.biLSTM_Attention(emb[self.idx_mapping])
        return final_emb

    def init_params(self):
        glorot(self.embedding)


class MessagePassing(nn.Module):
    r"""Base class for creating message passing layers

    .. math::
        \mathbf{x}_i^{\prime} = \gamma_{\mathbf{\Theta}} \left( \mathbf{x}_i,
        \square_{j \in \mathcal{N}(i)} \, \phi_{\mathbf{\Theta}}
        \left(\mathbf{x}_i, \mathbf{x}_j,\mathbf{e}_{i,j}\right) \right),

    where :math:`\square` denotes a differentiable, permutation invariant
    function, *e.g.*, sum, mean or max, and :math:`\gamma_{\mathbf{\Theta}}`
    and :math:`\phi_{\mathbf{\Theta}}` denote differentiable functions such as
    MLPs.
    See `here <https://rusty1s.github.io/pytorch_geometric/build/html/notes/
    create_gnn.html>`__ for the accompanying tutorial.

    """

    def __init__(self, aggr='add'):
        super(MessagePassing, self).__init__()

        self.message_args = inspect.getargspec(self.message)[0][1:]
        self.update_args = inspect.getargspec(self.update)[0][2:]

    def propagate(self, aggr, edge_index, **kwargs):
        r"""The initial call to start propagating messages.
        Takes in an aggregation scheme (:obj:`"add"`, :obj:`"mean"` or
        :obj:`"max"`), the edge indices, and all additional data which is
        needed to construct messages and to update node embeddings."""

        assert aggr in ['add', 'mean', 'max']
        kwargs['edge_index'] = edge_index   #kwargs 字典类型 其中包含 节点初始编码x、节点数量、边的二维tuble

        size = None
        message_args = []
        for arg in self.message_args:
            if arg[-2:] == '_i':    #该步骤的目的是message_args中加入
                tmp = kwargs[arg[:-2]]  #tmp :x 所有节点初始编码
                size = tmp.size(0)  #节点数量
                message_args.append(tmp[edge_index[0]])
            elif arg[-2:] == '_j':  #获取所有有向边的终点节点特征
                tmp = kwargs[arg[:-2]]
                size = tmp.size(0)
                message_args.append(tmp[edge_index[1]])
            else:
                message_args.append(kwargs[arg])

        update_args = [kwargs[arg] for arg in self.update_args]

        out = self.message(*message_args)
        out = scatter_add(out, edge_index[0], dim=0, dim_size=size)
        out = self.update(out, *update_args)

        return out

    def message(self, x_j):  # pragma: no cover
        r"""Constructs messages in analogy to :math:`\phi_{\mathbf{\Theta}}`
        for each edge in :math:`(i,j) \in \mathcal{E}`.
        Can take any argument which was initially passed to :meth:`propagate`.
        In addition, features can be lifted to the source node :math:`i` and
        target node :math:`j` by appending :obj:`_i` or :obj:`_j` to the
        variable name, *.e.g.* :obj:`x_i` and :obj:`x_j`."""

        return x_j

    def update(self, aggr_out):  # pragma: no cover
        r"""Updates node embeddings in analogy to
        :math:`\gamma_{\mathbf{\Theta}}` for each node
        :math:`i \in \mathcal{V}`.
        Takes in the output of aggregation as first argument and any argument
        which was initially passed to :meth:`propagate`."""

        return aggr_out


class GATConv(MessagePassing):
    r"""The graph attentional operator from the `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \alpha_{i,i}\mathbf{\Theta}\mathbf{x}_{j} +
        \sum_{j \in \mathcal{N}(i)} \alpha_{i,j}\mathbf{\Theta}\mathbf{x}_{j},

    where the attention coefficients :math:`\alpha_{i,j}` are computed as

    .. math::
        \alpha_{i,j} =
        \frac{
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_j]
        \right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_k]
        \right)\right)}.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions. (default:
            :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
        attentions are averaged instead of concatenated. (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 heads=1,
                 concat=True,
                 negative_slope=0.2,
                 dropout=0,
                 bias=True):
        super(GATConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.weight = nn.Parameter(
            torch.Tensor(in_channels, heads * out_channels))
        self.att = nn.Parameter(torch.Tensor(1, heads, 2 * out_channels))

        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index):
        """"""
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))#为所有节点添加自循环 结果变成了个长度为2的tuble类型且一个tuble中存放边的对，一个里面为空
        #用于存储有序的元素集合。元组的每个元素可以是不同的数据类型，并且一旦创建，元组的内容不能被修改
        x = torch.mm(x, self.weight).view(-1, self.heads, self.out_channels)#2584 5 20 x原本是2584 100，这里相当于加一矩阵变换并分割为5heads
        ans = self.propagate('add', edge_index, x=x, num_nodes=x.size(0))#三个参数 分别为：信息聚合类型， 边id 节点编码 节点数量
        return ans

    def message(self, x_i, x_j, edge_index, num_nodes):
        # Compute attention coefficients.
        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)#x_i 2,5217,5,20 x_j 1.2584.5.20 self.att 1,5,40
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index[0], num_nodes = num_nodes)

        alpha = F.dropout(alpha, p=self.dropout)

        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        if self.concat is True:
            aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
        else:
            aggr_out = aggr_out.mean(dim=1)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)

