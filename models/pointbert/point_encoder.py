import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath

from models.pointbert.dvae import Group
from models.pointbert.dvae import Encoder
from models.pointbert.logger import print_log
from models.pointbert.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message

from .pointnet2_utils import farthest_point_sample, index_points, PointNetFeaturePropagation, DGCNN_Propagation


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    """ Transformer Encoder without hierarchical structure
    """

    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
            )
            for i in range(depth)])

    def forward(self, x, pos, task='cls'):
        feature_list = []
        fetch_idx = [3, 7, 11]

        for i, block in enumerate(self.blocks):
            x = block(x + pos)
            if task == 'partseg' and i in fetch_idx:
                feature_list.append(x)

        if task == 'partseg':
            return feature_list
        else:
            return x


class PointTransformer(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.args = kwargs["args"]

        self.trans_dim = config.trans_dim
        self.depth = config.depth
        self.drop_path_rate = config.drop_path_rate
        self.cls_dim = config.cls_dim
        self.num_heads = config.num_heads

        self.group_size = config.group_size
        self.num_group = config.num_group
        # grouper
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
        # define the encoder
        self.encoder_dims = config.encoder_dims
        self.encoder = Encoder(encoder_channel=self.encoder_dims)
        # bridge encoder and transformer
        self.reduce_dim = nn.Linear(self.encoder_dims, self.trans_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads
        )

        self.norm = nn.LayerNorm(self.trans_dim)

        # self.head_type = self.args.head_type
        # if self.head_type == 0: # prompt only
        #     self.cls_head = nn.Identity()
        # elif self.head_type == 1:   # linear head
        #     self.cls_head = nn.Sequential(
        #         nn.BatchNorm1d(2*self.trans_dim), 
        #         nn.ReLU(),
        #         nn.Linear(2*self.trans_dim, 2*self.trans_dim))
        # elif self.head_type == 2:   # 2-layer mlp head
        #     self.cls_head = nn.Sequential(
        #         nn.BatchNorm1d(2*self.trans_dim), 
        #         nn.ReLU(),
        #         nn.Linear(2*self.trans_dim, 2*self.trans_dim),
        #         nn.BatchNorm1d(2*self.trans_dim), 
        #         nn.ReLU(),
        #         nn.Linear(2*self.trans_dim, 2*self.trans_dim))
        # elif self.head_type == 3:   # 3-layer mlp head
        #     self.cls_head = nn.Sequential(
        #         nn.BatchNorm1d(2*self.trans_dim), 
        #         nn.ReLU(),
        #         nn.Linear(2*self.trans_dim, 2*self.trans_dim),
        #         nn.BatchNorm1d(2*self.trans_dim), 
        #         nn.ReLU(),
        #         nn.Linear(2*self.trans_dim, 2*self.trans_dim),
        #         nn.BatchNorm1d(2*self.trans_dim), 
        #         nn.ReLU(),
        #         nn.Linear(2*self.trans_dim, 2*self.trans_dim))

    def build_loss_func(self):
        self.loss_ce = nn.CrossEntropyLoss()

    def get_loss_acc(self, pred, gt, smoothing=True):
        # import pdb; pdb.set_trace()
        gt = gt.contiguous().view(-1).long()

        if smoothing:
            eps = 0.2
            n_class = pred.size(1)

            one_hot = torch.zeros_like(pred).scatter(1, gt.view(-1, 1), 1)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
            log_prb = F.log_softmax(pred, dim=1)

            loss = -(one_hot * log_prb).sum(dim=1).mean()
        else:
            loss = self.loss_ce(pred, gt.long())

        pred = pred.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))

        return loss, acc * 100

    def load_model_from_ckpt(self, bert_ckpt_path):
        # 加载到 CPU 即可，因为这不属于训练参数 
        ckpt = torch.load(bert_ckpt_path, map_location=torch.device('cpu'))
        base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}
        for k in list(base_ckpt.keys()):
            if k.startswith('transformer_q') and not k.startswith('transformer_q.cls_head'):
                base_ckpt[k[len('transformer_q.'):]] = base_ckpt[k]
            elif k.startswith('base_model'):
                base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
            del base_ckpt[k]

        incompatible = self.load_state_dict(base_ckpt, strict=False)

        if incompatible.missing_keys:
            print_log('missing_keys', logger='Transformer')
            print_log(
                get_missing_parameters_message(incompatible.missing_keys),
                logger='Transformer'
            )
        if incompatible.unexpected_keys:
            print_log('unexpected_keys', logger='Transformer')
            print_log(
                get_unexpected_parameters_message(incompatible.unexpected_keys),
                logger='Transformer'
            )

        print_log(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}', logger='Transformer')

    def forward(self, pts):
        # divide the point cloud in the same form. This is important
        neighborhood, center = self.group_divider(pts)
        # encoder the input cloud blocks
        group_input_tokens = self.encoder(neighborhood)  # B G N
        group_input_tokens = self.reduce_dim(group_input_tokens)
        # prepare cls
        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)
        # add pos embedding
        pos = self.pos_embed(center)
        # final input
        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)
        # transformer
        x = self.blocks(x, pos, task='cls')
        x = self.norm(x)
        concat_f = torch.cat([x[:, 0], x[:, 1:].max(1)[0]], dim=-1)
        # --- option 1
        out = concat_f
        # --- option 2
        # out = self.cls_head(concat_f)
        
        return out
    

class PointTransformer_partseg(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config

        self.trans_dim = config.trans_dim
        self.depth = config.depth 
        self.drop_path_rate = config.drop_path_rate 
        self.cls_dim = config.cls_dim 
        self.num_heads = config.num_heads 

        self.group_size = config.group_size
        self.num_group = config.num_group
        # grouper
        self.group_divider = Group(num_group = self.num_group, group_size = self.group_size)
        # define the encoder
        self.encoder_dims =  config.encoder_dims
        self.encoder = Encoder(encoder_channel = self.encoder_dims)
        # bridge encoder and transformer，感觉没必要做这个转换，上面encoder出来的维度直接映射到trans_dim即可
        self.reduce_dim = nn.Linear(self.encoder_dims,  self.trans_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )  

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim = self.trans_dim,
            depth = self.depth,
            drop_path_rate = dpr,
            num_heads = self.num_heads
        )

        self.norm = nn.LayerNorm(self.trans_dim)

        self.propagation_2 = PointNetFeaturePropagation(in_channel= self.trans_dim + 3, mlp = [self.trans_dim * 4, self.trans_dim])
        self.propagation_1= PointNetFeaturePropagation(in_channel= self.trans_dim + 3, mlp = [self.trans_dim * 4, self.trans_dim])
        self.propagation_0 = PointNetFeaturePropagation(in_channel= self.trans_dim + 3 + 16, mlp = [self.trans_dim * 4, self.trans_dim])
        self.dgcnn_pro_1 = DGCNN_Propagation(k = 4)
        self.dgcnn_pro_2 = DGCNN_Propagation(k = 4)

        self.conv1 = nn.Conv1d(self.trans_dim, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, self.cls_dim, 1)

        self.build_loss_func()
        
    def build_loss_func(self):
        self.loss_ce = nn.CrossEntropyLoss()
    
    def get_loss_acc(self, ret, gt):
        loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def load_model_from_ckpt(self, bert_ckpt_path):
        ckpt = torch.load(bert_ckpt_path)
        base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}
        for k in list(base_ckpt.keys()):
            if k.startswith('transformer_q') and not k.startswith('transformer_q.cls_head'):
                base_ckpt[k[len('transformer_q.'):]] = base_ckpt[k]
            elif k.startswith('base_model'):
                base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
            del base_ckpt[k]

        incompatible = self.load_state_dict(base_ckpt, strict=False)

        if incompatible.missing_keys:
            print('missing_keys')
            print(
                get_missing_parameters_message(incompatible.missing_keys)
            )
        if incompatible.unexpected_keys:
            print('unexpected_keys')
            print(
                get_unexpected_parameters_message(incompatible.unexpected_keys)
            )

        print(f'[PointTransformer] Successful Loading the ckpt from {bert_ckpt_path}')

    def forward(self, pts, cls_label):
        # pts: [batch, npoints, channels]
        B,N,_ = pts.shape

        # divide the point cloud in the same form. This is important
        # neighbor: [batch, ngroups, group_size, channels]
        # center: [batch, ngroups, channels]
        neighborhood, center = self.group_divider(pts)
        # encoder the input cloud blocks
        # group_input_tokens: [batch, ngroups, dim_encoder]
        group_input_tokens = self.encoder(neighborhood)  #  B G N
        # group_input_tokens: [batch, ngroups, dim_trans]
        group_input_tokens = self.reduce_dim(group_input_tokens)
        # prepare cls
        # cls_tokens: [batch, 1, trans_dim]
        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)  
        # cls_pos: [batch, 1, trans_dim]
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)  
        # add pos embedding
        # pos: [batch, ngroups, dim_encoder]
        pos = self.pos_embed(center)
        # final input
        # x: [batch, 1+ngroups, dim_trans]
        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        # pos: [batch, 1+ngroups, dim_trans]
        pos = torch.cat((cls_pos, pos), dim=1)
        # transformer
        # feature_list: [3, batch, 1+ngroups, dim_trans]
        feature_list = self.blocks(x, pos, task='partseg')
        # [3, batch, dim_trans, ngroups]
        feature_list = [self.norm(x)[:,1:].transpose(-1, -2).contiguous() for x in feature_list]    # 这里把cls_token扔了

        # cls_label_one_hot: [batch, 16, N]
        cls_label_one_hot = cls_label.view(B, 16, 1).repeat(1, 1, N)
        # center_level_0: [batch, point_channels, npoints]
        center_level_0 = pts.transpose(-1, -2).contiguous()
        # f_level_0: [batch, 16+point_channels, npoints]
        f_level_0 = torch.cat([cls_label_one_hot, center_level_0], 1)

        # center_level_1: [batch, point_channels, 512]
        pts_idx_512 = farthest_point_sample(pts, 512)
        center_level_1 = index_points(pts, pts_idx_512).transpose(-1, -2).contiguous()
        # f_level_1: [batch, point_channels, 512]
        f_level_1 = center_level_1
        # center_level_2: [batch, point_channels, 256]
        pts_idx_256 = farthest_point_sample(pts, 256)            
        center_level_2 = index_points(pts, pts_idx_256).transpose(-1, -2).contiguous()
        # f_level_2: [batch, point_channels, 256]
        f_level_2 = center_level_2
        # center_level_3: [batch, point_channels, ngroups]
        center_level_3 = center.transpose(-1, -2).contiguous()

        # init the feature by 3nn propagation
        # f_level_3: [batch, dim_trans, ngroups]
        f_level_3 = feature_list[2]

        # f_level_2: [batch, dim_trans, 256]
        f_level_2 = self.propagation_2(center_level_2, center_level_3, f_level_2, feature_list[1])
        # f_level_1: [batch, dim_trans, 512]
        f_level_1 = self.propagation_1(center_level_1, center_level_3, f_level_1, feature_list[0])

        # f_level_2: [batch, dim_trans, 256]
        f_level_2 = self.dgcnn_pro_2(center_level_3, f_level_3, center_level_2, f_level_2)
        # f_level_1: [batch, dim_trans, 512]
        f_level_1 = self.dgcnn_pro_1(center_level_2, f_level_2, center_level_1, f_level_1)
        # f_level_0: [batch, dim_trans, npoints]
        f_level_0 =  self.propagation_0(center_level_0, center_level_1, f_level_0, f_level_1)

        # ------ 下面把 partseg head 替换成返回每点特征
        # feat: [batch, 128, npoints]
        feat =  self.drop1(F.relu(self.bn1(self.conv1(f_level_0))))
        # x: [batch, npoints, 128]
        x = feat.permute(0, 2, 1)
        return x
