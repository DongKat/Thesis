import torch
from torch import nn as nn
from torch.nn import functional as F
import math
import copy
import numpy as np

from basicsr.utils.registry import ARCH_REGISTRY
from .arch_util import default_init_weights

def grid_sample(input, grid, canvas = None):
  output = F.grid_sample(input, grid)
  if canvas is None:
    return output
  else:
    input_mask = input.data.new(input.size()).fill_(1)
    output_mask = F.grid_sample(input_mask, grid)
    padded_output = output * output_mask + canvas * (1 - output_mask)
    return padded_output

# phi(x1, x2) = r^2 * log(r), where r = ||x1 - x2||_2
def compute_partial_repr(input_points, control_points):
  N = input_points.size(0)
  M = control_points.size(0)
  pairwise_diff = input_points.view(N, 1, 2) - control_points.view(1, M, 2)
  # original implementation, very slow
  # pairwise_dist = torch.sum(pairwise_diff ** 2, dim = 2) # square of distance
  pairwise_diff_square = pairwise_diff * pairwise_diff
  pairwise_dist = pairwise_diff_square[:, :, 0] + pairwise_diff_square[:, :, 1]
  repr_matrix = 0.5 * pairwise_dist * torch.log(pairwise_dist)
  # fix numerical error for 0 * log(0), substitute all nan with 0
  mask = repr_matrix != repr_matrix
  repr_matrix.masked_fill_(mask, 0)
  return repr_matrix

# output_ctrl_pts are specified, according to our task.
def build_output_control_points(num_control_points, margins):
  margin_x, margin_y = margins
  num_ctrl_pts_per_side = num_control_points // 2
  ctrl_pts_x = np.linspace(margin_x, 1.0 - margin_x, num_ctrl_pts_per_side)
  ctrl_pts_y_top = np.ones(num_ctrl_pts_per_side) * margin_y
  ctrl_pts_y_bottom = np.ones(num_ctrl_pts_per_side) * (1.0 - margin_y)
  ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
  ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
  # ctrl_pts_top = ctrl_pts_top[1:-1,:]
  # ctrl_pts_bottom = ctrl_pts_bottom[1:-1,:]
  output_ctrl_pts_arr = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)
  output_ctrl_pts = torch.Tensor(output_ctrl_pts_arr)
  return output_ctrl_pts

# TPS
class TPSSpatialTransformer(nn.Module):

  def __init__(self, output_image_size=None, num_control_points=None, margins=None):
    super(TPSSpatialTransformer, self).__init__()
    self.output_image_size = output_image_size
    self.num_control_points = num_control_points
    self.margins = margins

    self.target_height, self.target_width = output_image_size
    target_control_points = build_output_control_points(num_control_points, margins)
    N = num_control_points
    # N = N - 4

    # create padded kernel matrix
    forward_kernel = torch.zeros(N + 3, N + 3)
    target_control_partial_repr = compute_partial_repr(target_control_points, target_control_points)
    forward_kernel[:N, :N].copy_(target_control_partial_repr)
    forward_kernel[:N, -3].fill_(1)
    forward_kernel[-3, :N].fill_(1)
    forward_kernel[:N, -2:].copy_(target_control_points)
    forward_kernel[-2:, :N].copy_(target_control_points.transpose(0, 1))
    # compute inverse matrix
    inverse_kernel = torch.inverse(forward_kernel)

    # create target cordinate matrix
    HW = self.target_height * self.target_width
    target_coordinate = list(itertools.product(range(self.target_height), range(self.target_width)))
    target_coordinate = torch.Tensor(target_coordinate) # HW x 2
    Y, X = target_coordinate.split(1, dim = 1)
    Y = Y / (self.target_height - 1)
    X = X / (self.target_width - 1)
    target_coordinate = torch.cat([X, Y], dim = 1) # convert from (y, x) to (x, y)
    target_coordinate_partial_repr = compute_partial_repr(target_coordinate, target_control_points)
    target_coordinate_repr = torch.cat([
      target_coordinate_partial_repr, torch.ones(HW, 1), target_coordinate
    ], dim = 1)

    # register precomputed matrices
    self.register_buffer('inverse_kernel', inverse_kernel)
    self.register_buffer('padding_matrix', torch.zeros(3, 2))
    self.register_buffer('target_coordinate_repr', target_coordinate_repr)
    self.register_buffer('target_control_points', target_control_points)

  def forward(self, input, source_control_points):
    assert source_control_points.ndimension() == 3
    assert source_control_points.size(1) == self.num_control_points
    assert source_control_points.size(2) == 2
    batch_size = source_control_points.size(0)

    Y = torch.cat([source_control_points, self.padding_matrix.expand(batch_size, 3, 2)], 1)
    mapping_matrix = torch.matmul(self.inverse_kernel, Y)
    source_coordinate = torch.matmul(self.target_coordinate_repr, mapping_matrix)

    grid = source_coordinate.view(-1, self.target_height, self.target_width, 2)
    grid = torch.clamp(grid, 0, 1) # the source_control_points may be out of [0, 1].
    # the input to grid_sample is normalized [-1, 1], but what we get is [0, 1]
    grid = 2.0 * grid - 1.0
    output_maps = grid_sample(input, grid, canvas=None)
    return output_maps, source_coordinate

# STN
def conv3x3_block(in_planes, out_planes, stride=1):
  """3x3 convolution with padding"""
  conv_layer = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1)

  block = nn.Sequential(
    conv_layer,
    nn.BatchNorm2d(out_planes),
    nn.ReLU(inplace=True),
  )
  return block

class STNHead(nn.Module):
  def __init__(self, in_planes, num_ctrlpoints, activation='none'):
    super(STNHead, self).__init__()

    self.in_planes = in_planes
    self.num_ctrlpoints = num_ctrlpoints
    self.activation = activation
    self.stn_convnet = nn.Sequential(
                          conv3x3_block(in_planes, 32), # 32*64
                          nn.MaxPool2d(kernel_size=2, stride=2),
                          conv3x3_block(32, 64), # 16*32
                          nn.MaxPool2d(kernel_size=2, stride=2),
                          conv3x3_block(64, 128), # 8*16
                          nn.MaxPool2d(kernel_size=2, stride=2),
                          conv3x3_block(128, 256), # 4*8
                          nn.MaxPool2d(kernel_size=2, stride=2),
                          conv3x3_block(256, 256), # 2*4,
                          nn.MaxPool2d(kernel_size=2, stride=2),
                          conv3x3_block(256, 256)) # 1*2

    self.stn_fc1 = nn.Sequential(
                      nn.Linear(2*256, 512),
                      nn.BatchNorm1d(512),
                      nn.ReLU(inplace=True))
    self.stn_fc2 = nn.Linear(512, num_ctrlpoints*2)

    self.init_weights(self.stn_convnet)
    self.init_weights(self.stn_fc1)
    self.init_stn(self.stn_fc2)

  def init_weights(self, module):
    for m in module.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
          m.bias.data.zero_()
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
      elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.001)
        m.bias.data.zero_()

  def init_stn(self, stn_fc2):
    margin = 0.01
    sampling_num_per_side = int(self.num_ctrlpoints / 2)
    ctrl_pts_x = np.linspace(margin, 1.-margin, sampling_num_per_side)
    ctrl_pts_y_top = np.ones(sampling_num_per_side) * margin
    ctrl_pts_y_bottom = np.ones(sampling_num_per_side) * (1-margin)
    ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
    ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
    ctrl_points = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0).astype(np.float32)
    if self.activation is 'none':
      pass
    elif self.activation == 'sigmoid':
      ctrl_points = -np.log(1. / ctrl_points - 1.)
    stn_fc2.weight.data.zero_()
    stn_fc2.bias.data = torch.Tensor(ctrl_points).view(-1)

  def forward(self, x):
    x = self.stn_convnet(x)
    batch_size, _, h, w = x.size()
    x = x.view(batch_size, -1)
    # embed()
    img_feat = self.stn_fc1(x)
    x = self.stn_fc2(0.1 * img_feat)
    if self.activation == 'sigmoid':
      x = F.sigmoid(x)
    x = x.view(-1, self.num_ctrlpoints, 2)
    return img_feat, x


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        # print(features)
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe

class FeatureEnhancer(nn.Module):

    def __init__(self):
        super(FeatureEnhancer, self).__init__()

        self.multihead = MultiHeadedAttention(h=4, d_model=128, dropout=0.1)
        self.mul_layernorm1 = LayerNorm(features=128)

        self.pff = PositionwiseFeedForward(128, 128)
        self.mul_layernorm3 = LayerNorm(features=128)

        self.linear = nn.Linear(128,64)

    def forward(self, conv_feature):
        '''
        text : (batch, seq_len, embedding_size)
        global_info: (batch, embedding_size, 1, 1)
        conv_feature: (batch, channel, H, W)
        '''
        batch = conv_feature.shape[0]
        position2d = positionalencoding2d(64,16,64).float().cuda().unsqueeze(0).view(1,64,1024)
        position2d = position2d.repeat(batch,1,1)
        conv_feature = torch.cat([conv_feature, position2d],1) # batch, 128(64+64), 32, 128
        result = conv_feature.permute(0, 2, 1).contiguous()
        origin_result = result
        result = self.mul_layernorm1(origin_result + self.multihead(result, result, result, mask=None)[0])
        origin_result = result
        result = self.mul_layernorm3(origin_result + self.pff(result))
        result = self.linear(result)
        return result.permute(0, 2, 1).contiguous()


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1, compress_attention=False):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.compress_attention = compress_attention
        self.compress_attention_linear = nn.Linear(h, 1)

    def forward(self, query, key, value, mask=None, align=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attention_map = attention(query, key, value, mask=mask,
                                     dropout=self.dropout, align=align)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)

        return self.linears[-1](x), attention_map


def attention(query, key, value, mask=None, dropout=None, align=None):
    "Compute 'Scaled Dot Product Attention'"

    d_k = query.size(-1)

    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        # print(mask)
        scores = scores.masked_fill(mask == 0, float('-inf'))
    else:
        pass

    p_attn = F.softmax(scores, dim=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class TBSRN(nn.Module):
    def __init__(self, scale_factor=2, width=128, height=32, STN=True, srb_nums=5, mask=False, hidden_units=32, input_channel=3):
        super(TBSRN, self).__init__()

        self.conv = nn.Conv2d(input_channel, 3,3,1,1)
        self.bn = nn.BatchNorm2d(3)
        self.relu = nn.ReLU()

        in_planes = 3
        if mask:
            in_planes = 4
        assert math.log(scale_factor, 2) % 1 == 0
        upsample_block_num = int(math.log(scale_factor, 2))
        self.block1 = nn.Sequential(
            nn.Conv2d(in_planes, 2 * hidden_units, kernel_size=9, padding=4),
            nn.PReLU()
            # nn.ReLU()
        )
        self.srb_nums = srb_nums
        for i in range(srb_nums):
            setattr(self, 'block%d' % (i + 2), RecurrentResidualBlock(2 * hidden_units))

        setattr(self, 'block%d' % (srb_nums + 2),
                nn.Sequential(
                    nn.Conv2d(2 * hidden_units, 2 * hidden_units, kernel_size=3, padding=1),
                    nn.BatchNorm2d(2 * hidden_units)
                ))

        # self.non_local = NonLocalBlock2D(64, 64)
        block_ = [UpsampleBLock(2 * hidden_units, 2) for _ in range(upsample_block_num)]
        block_.append(nn.Conv2d(2 * hidden_units, in_planes, kernel_size=9, padding=4))
        setattr(self, 'block%d' % (srb_nums + 3), nn.Sequential(*block_))
        self.tps_inputsize = [height // scale_factor, width // scale_factor]
        tps_outputsize = [height // scale_factor, width // scale_factor]
        num_control_points = 20
        tps_margins = [0.05, 0.05]
        self.stn = STN
        if self.stn:
            self.tps = TPSSpatialTransformer(
                output_image_size=tuple(tps_outputsize),
                num_control_points=num_control_points,
                margins=tuple(tps_margins))

            self.stn_head = STNHead(
                in_planes=in_planes,
                num_ctrlpoints=num_control_points,
                activation='none')

    def forward(self, x):
        if self.stn and self.training:
            # x = F.interpolate(x, self.tps_inputsize, mode='bilinear', align_corners=True)
            _, ctrl_points_x = self.stn_head(x)
            x, _ = self.tps(x, ctrl_points_x)
        block = {'1': self.block1(x)}
        for i in range(self.srb_nums + 1):
            block[str(i + 2)] = getattr(self, 'block%d' % (i + 2))(block[str(i + 1)])

        block[str(self.srb_nums + 3)] = getattr(self, 'block%d' % (self.srb_nums + 3)) \
            ((block['1'] + block[str(self.srb_nums + 2)]))
        output = torch.tanh(block[str(self.srb_nums + 3)])
        return output


class RecurrentResidualBlock(nn.Module):
    def __init__(self, channels):
        super(RecurrentResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.gru1 = GruBlock(channels, channels)
        # self.prelu = nn.ReLU()
        self.prelu = mish()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.gru2 = GruBlock(channels, channels)
        self.feature_enhancer = FeatureEnhancer()

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        size = residual.shape
        residual = residual.view(size[0],size[1],-1)
        residual = self.feature_enhancer(residual)
        residual = residual.resize(size[0], size[1], size[2], size[3])
        return x + residual



class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)

        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        # self.prelu = nn.ReLU()
        self.prelu = mish()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x


class mish(nn.Module):
    def __init__(self, ):
        super(mish, self).__init__()
        self.activated = True

    def forward(self, x):
        if self.activated:
            x = x * (torch.tanh(F.softplus(x)))
        return x


class GruBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GruBlock, self).__init__()
        assert out_channels % 2 == 0
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.gru = nn.GRU(out_channels, out_channels // 2, bidirectional=True, batch_first=True)

    def forward(self, x):
        # x: b, c, w, h
        x = self.conv1(x)
        x = x.permute(0, 2, 3, 1).contiguous()  # b, w, h, c
        b = x.size()
        x = x.view(b[0] * b[1], b[2], b[3])  # b*w, h, c
        x, _ = self.gru(x)
        # x = self.gru(x)[0]
        x = x.view(b[0], b[1], b[2], b[3])
        x = x.permute(0, 3, 1, 2).contiguous()
        return x