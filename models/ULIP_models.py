'''
 * Adapted from ULIP (https://github.com/salesforce/ULIP)
 * By Hongyu Sun
'''
from collections import OrderedDict

import os
import numpy as np
import torch
from torch import nn
from models.pointnet2.pointnet2 import Pointnet2_Ssg, Pointnet2_Msg
from data.dataset_3d import  *

from models import losses
from torch.nn.parameter import Parameter
from easydict import EasyDict

from utils.tokenizer import SimpleTokenizer


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class PromptLearner(nn.Module):
    def __init__(self, token_embeding, kwargs):
        super().__init__()

        self.class_name_position = kwargs.class_name_position
        self.classnames = kwargs.classnames
        self.num_learnable_prompt_tokens = kwargs.num_learnable_prompt_tokens
        self.transformer_width = kwargs.transformer_width
        self.device = kwargs.device

        if kwargs.template_init != '':
            template_init = kwargs.template_init.replace("_", " ")  # 这一步很重要，把带下划线的短语进行切分
            self.num_learnable_prompt_tokens = len(template_init.split(' '))
            prompt_prefix = template_init
        else:
            self.num_learnable_prompt_tokens = kwargs.num_learnable_prompt_tokens
            prompt_prefix = " ".join(["X"] * self.num_learnable_prompt_tokens)

        self.learnable_tokens = nn.Parameter(torch.empty(self.num_learnable_prompt_tokens, self.transformer_width))

        classnames = [name.replace("_", " ") for name in self.classnames]   # "nigth_stand" -> "night stand"
        tokenizer = SimpleTokenizer()
        # [num_classes]
        self.name_lengths = [len(tokenizer.encode(name)) for name in classnames]   # 每个classname bpe编码的长度
        # [num_classes]
        prompts = [prompt_prefix + " " + name + "." for name in classnames] # 相当于每个类对应一个 prompt

        # tokenizer(p) -> [context_length]
        # [tokenizer(p) for p in prompts]   ->  是一个list，共有 `n_cls` 个tensor，每个tensor维度：[1, context_length]
        # tokenized_prompts -> [num_classes, context_length]
        self.tokenized_prompts = torch.stack([tokenizer(p) for p in prompts])
        # [num_classes, context_length, transformer_width]
        self.embedding = token_embeding(self.tokenized_prompts).cuda(device=self.device)    # 一次性转移到指定设备上

    def forward(self):

        num_classes = self.embedding.shape[0]
        if self.learnable_tokens.dim() == 2:
            learnable_tokens = self.learnable_tokens.unsqueeze(0).repeat(num_classes, 1, 1)

        prefix = self.embedding[:, :1, :]   # 句子开头标记
        suffix = self.embedding[:, 1+self.num_learnable_prompt_tokens:, :]  # 句子结尾标记

        if self.class_name_position == "front":
            prompts = []
            for i in range(num_classes):
                shape_name_len = self.name_lengths[i]

                prefix_i = prefix[i : i+1, :1, :]
                class_i = suffix[i : i+1, :shape_name_len, :]
                learnable_tokens_i = learnable_tokens[i : i+1, :, :]
                suffix_i = suffix[i : i+1, shape_name_len:, :]

                prompt = torch.cat([prefix_i, class_i, learnable_tokens_i, suffix_i], dim=1)
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_name_position == "middle":
            prompts = []
            half_len = self.num_learnable_prompt_tokens // 2
            for i in range(num_classes):
                shape_name_len = self.name_lengths[i]

                prefix_i = prefix[i : i+1, :, :]
                learnable_tokens_i_half1 = learnable_tokens[i : i+1, :half_len, :]
                class_i = suffix[i : i+1, :shape_name_len, :]
                learnable_tokens_i_half2 = learnable_tokens[i : i+1, half_len:, :]
                suffix_i = suffix[i : i+1, shape_name_len:, :]

                prompt = torch.cat([prefix_i, learnable_tokens_i_half1, class_i, 
                                    learnable_tokens_i_half2, suffix_i], dim=1)
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_name_position == "end":
            prompts = torch.cat([prefix, learnable_tokens, suffix], dim=1)

        else:
            raise ValueError(f'`class_name_position`: {self.class_name_position} not in supported modes ["front", "middle", "end"]')

        # [num_classes, context_length, transformer_width]
        return prompts


class ULIP_WITH_IMAGE(nn.Module):
    def __init__(self, point_encoder, **kwargs):
        '''
            NOTE 
                1. text model defined in __init__ function
                2. image encoder is not necessary for my situation
        '''
        # super().__init__(ssl_mlp_dim, ssl_emb_dim, **kwargs)
        super().__init__()
        kwargs = EasyDict(kwargs)
        self.task = kwargs.task
        self.context_length = kwargs.context_length
        # self.vision_width = kwargs.vision_width
        # self.visual = kwargs.vision_model
        self.device = kwargs.device

        self.transformer = Transformer(
            width=kwargs.transformer_width,
            layers=kwargs.transformer_layers,
            heads=kwargs.transformer_heads,
            attn_mask=self.build_attention_mask(),
        )

        self.vocab_size = kwargs.vocab_size
        self.token_embedding = nn.Embedding(kwargs.vocab_size, kwargs.transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, kwargs.transformer_width))
        self.ln_final = LayerNorm(kwargs.transformer_width)

        # self.image_projection = nn.Parameter(torch.empty(kwargs.vision_width, kwargs.embed_dim))
        self.text_projection = nn.Parameter(torch.empty(kwargs.transformer_width, kwargs.embed_dim))
        self.pc_projection = nn.Parameter(torch.empty(kwargs.pc_feat_dims, kwargs.embed_dim))
        # --- where is it from?
        #   把 logit_scale 设成了可学习参数，读一下论文，看看是怎么讲的
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.point_encoder = point_encoder

        # --- learning to prompt --- 
        self.prompt_learner = PromptLearner(self.token_embedding, kwargs)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts

        self.initialize_parameters()

    # def encode_image(self, image):
    #     x = self.visual(image)
    #     x = x @ self.image_projection

    #     return x

    def encode_text(self, prompts, tokenized_prompts):
        '''
            prompts: [num_classes, context_length, transformer_width]
            tokenized_prompts: [num_classes, context_length]
        '''

        x = prompts + self.positional_embedding.unsqueeze(dim=0).repeat(prompts.shape[0], 1, 1)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # 用末尾的 token embedding 代表整个 prompt 的特征，这在多个论文都提到过我记得，比如 learning to prompt for vision-language models
        #   x[torch.arange(x.shape[0]), text.argmax(dim=-1)] -> [num_classes, transformer_width]
        #   self.text_projection                             -> [transformer_width, embed_dim]
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        # x: [num_classes, embed_dim]
        return x

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        nn.init.normal_(self.prompt_learner.learnable_tokens, std=0.02)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        # nn.init.normal_(self.image_projection, std=self.vision_width ** -0.5)
        nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)
        nn.init.normal_(self.pc_projection, std=512 ** -0.5)

    def encode_pc(self, pc, cls_label=None):
        if self.task == 'partseg':
            # [batch, npoints, num_parts]
            pc_feat = self.point_encoder(pc, cls_label)
        else:
            # [batch, num_classes]
            pc_feat = self.point_encoder(pc)
        pc_embed = pc_feat @ self.pc_projection
        return pc_embed

    def forward(self, pc, cls_label=None):
        if self.task == 'partseg':
            # pc:    (batch, n_points, 3)
            # pc_embed: [batch, embed_dim]
            pc_embed = self.encode_pc(pc, cls_label)
        else:
            # pc:    (batch, n_points, 3)
            # pc_embed: [batch, embed_dim]
            pc_embed = self.encode_pc(pc)

        # prompts: [num_classes, context_length, transformer_width]
        prompts = self.prompt_learner() # forward pass
        # tokenized_prompts: [num_classes, context_length]
        tokenized_prompts = self.tokenized_prompts

        # text_embed: [num_classes, embed_dim]
        text_embed = self.encode_text(prompts, tokenized_prompts)
        text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
        
        logit_scale = self.logit_scale.exp()
        # logits: [batch, num_classes]
        logits = logit_scale * pc_embed @ text_embed.t()

        return logits


def get_loss():
    return losses.ULIPWithImageLoss()


def get_metric_names():
    return ['loss', 'acc']


def ULIP_PN_SSG(args):
    # vision_model = timm.create_model('vit_base_patch16_224', num_classes=0)

    # =====================================================================
    # import the 3D backbone and specify the output point cloud feature dimension
    point_encoder = Pointnet2_Ssg()
    pc_feat_dims = 256
    # =====================================================================

    # 前向传播返回的是一个字典，包含文本特征、点云特征、图像特征，外加一个归一化因子
    model = ULIP_WITH_IMAGE(embed_dim=512, point_encoder=point_encoder, context_length=77, 
            vocab_size=49408, classnames=args.classnames, template_init=args.template_init, class_name_position=args.class_name_position, 
            num_learnable_prompt_tokens=args.num_learnable_prompt_tokens, transformer_width=512, transformer_heads=8, transformer_layers=12, 
            pc_feat_dims=pc_feat_dims, device=args.gpu, task=args.task)

    if not args.evaluate_3d:
        # load the pretrained model
        pretrain_point_model = torch.load('./data/pretrained_models/pointnet2_ssg.pt', map_location=torch.device(f'cpu'))
        pretrain_point_model_params = pretrain_point_model['state_dict']
        pretrain_point_model_params = {param_name.replace('module.', ''): param for param_name, param in
                                      pretrain_point_model_params.items()}

        pretrain_slip_model = torch.load('./data/initialize_models/slip_base_100ep.pt', map_location=torch.device(f'cpu'))
        pretrain_slip_model_params = pretrain_slip_model['state_dict']
        pretrain_slip_model_params = {param_name.replace('module.', ''): param for param_name, param in
                                      pretrain_slip_model_params.items()}

        # 首先这里用的是 named_parameters() 而不是 parameters，表明 model 的参数都是有对应名称的
        for name, param in model.named_parameters():
            if name == 'prompt_learner.learnable_tokens': 
                continue

            if name in pretrain_point_model_params:
                if isinstance(pretrain_point_model_params[name], Parameter):
                    param_new = pretrain_point_model_params[name].data
                else:
                    param_new = pretrain_point_model_params[name]
            else:
                if isinstance(pretrain_slip_model_params[name], Parameter):
                    param_new = pretrain_slip_model_params[name].data
                else:
                    param_new = pretrain_slip_model_params[name]

            param.requires_grad = False
            print('load {} and freeze'.format(name))
            param.data.copy_(param_new)

    return model


def ULIP_PN_MSG(args):
    # vision_model = timm.create_model('vit_base_patch16_224', num_classes=0)

    # =====================================================================
    # import the 3D backbone and specify the output point cloud feature dimension
    point_encoder = Pointnet2_Msg()
    pc_feat_dims = 256
    # =====================================================================

    # 前向传播返回的是一个字典，包含文本特征、点云特征、图像特征，外加一个归一化因子
    model = ULIP_WITH_IMAGE(embed_dim=512, point_encoder=point_encoder, context_length=77, 
            vocab_size=49408, classnames=args.classnames, template_init=args.template_init, class_name_position=args.class_name_position, 
            num_learnable_prompt_tokens=args.num_learnable_prompt_tokens, transformer_width=512, transformer_heads=8, transformer_layers=12, 
            pc_feat_dims=pc_feat_dims, device=args.gpu, task=args.task)

    if not args.evaluate_3d:
        # load the pretrained model
        pretrain_point_model = torch.load('./data/pretrained_models/pointnet2_msg_1kpts.pt', map_location=torch.device('cpu'))
        pretrain_point_model_params = pretrain_point_model['state_dict']
        pretrain_point_model_params = {param_name.replace('module.', ''): param for param_name, param in
                                      pretrain_point_model_params.items()}

        pretrain_slip_model = torch.load('./data/initialize_models/slip_base_100ep.pt', map_location=torch.device('cpu'))
        pretrain_slip_model_params = pretrain_slip_model['state_dict']
        pretrain_slip_model_params = {param_name.replace('module.', ''): param for param_name, param in
                                      pretrain_slip_model_params.items()}

        # 首先这里用的是 named_parameters() 而不是 parameters，表明 model 的参数都是有对应名称的
        for name, param in model.named_parameters():
            if name == 'prompt_learner.learnable_tokens': 
                continue

            if name in pretrain_point_model_params:
                if isinstance(pretrain_point_model_params[name], Parameter):
                    param_new = pretrain_point_model_params[name].data
                else:
                    param_new = pretrain_point_model_params[name]
            else:
                if isinstance(pretrain_slip_model_params[name], Parameter):
                    param_new = pretrain_slip_model_params[name].data
                else:
                    param_new = pretrain_slip_model_params[name]

            param.requires_grad = False
            print('load {} and freeze'.format(name))
            param.data.copy_(param_new)

    return model


def ULIP_PN_MLP(args):
    # vision_model = timm.create_model('vit_base_patch16_224', num_classes=0)

    # =====================================================================
    # import the 3D backbone and specify the output point cloud feature dimension
    from models.pointmlp.pointMLP import pointMLP
    point_encoder = pointMLP()
    pc_feat_dims = 256
    # =====================================================================

    model = ULIP_WITH_IMAGE(embed_dim=512, point_encoder=point_encoder, context_length=77, 
            vocab_size=49408, classnames=args.classnames, template_init=args.template_init, class_name_position=args.class_name_position, 
            num_learnable_prompt_tokens=args.num_learnable_prompt_tokens, transformer_width=512, transformer_heads=8, transformer_layers=12, 
            pc_feat_dims=pc_feat_dims, device=args.gpu, task=args.task)

    if not args.evaluate_3d:
        # load the pretrained model
        pretrain_point_model = torch.load('./data/pretrained_models/pointmlp.pt', map_location=torch.device('cpu'))
        pretrain_point_model_params = pretrain_point_model['state_dict']
        pretrain_point_model_params = {param_name.replace('module.', ''): param for param_name, param in
                                      pretrain_point_model_params.items()}
        
        pretrain_slip_model = torch.load('./data/initialize_models/slip_base_100ep.pt', map_location=torch.device('cpu'))
        pretrain_slip_model_params = pretrain_slip_model['state_dict']
        pretrain_slip_model_params = {param_name.replace('module.', ''): param for param_name, param in
                                      pretrain_slip_model_params.items()}

        for name, param in model.named_parameters():
            if name == 'prompt_learner.learnable_tokens': 
                continue

            if name in pretrain_point_model_params:
                if isinstance(pretrain_point_model_params[name], Parameter):
                    param_new = pretrain_point_model_params[name].data
                else:
                    param_new = pretrain_point_model_params[name]
            else:
                if isinstance(pretrain_slip_model_params[name], Parameter):
                    param_new = pretrain_slip_model_params[name].data
                else:
                    param_new = pretrain_slip_model_params[name]

            param.requires_grad = False
            print('load {} and freeze'.format(name))
            param.data.copy_(param_new)

    return model


def ULIP_PointBERT(args):
    # NOTE for prompting ULIP, we do not need image encoder. Text and point encoder is enough.
    # vision_model = timm.create_model('vit_base_patch16_224', num_classes=0)

    # =====================================================================
    # import the 3D backbone and specify the output point cloud feature dimension
    from models.pointbert.point_encoder import PointTransformer
    config_addr = './models/pointbert/PointTransformer_8192point.yaml'
    config = cfg_from_yaml_file(config_addr)
    point_encoder = PointTransformer(config.model, args=args)
    pc_feat_dims = 768
    # =====================================================================

    model = ULIP_WITH_IMAGE(embed_dim=512, point_encoder=point_encoder, context_length=77, 
            vocab_size=49408, classnames=args.classnames, template_init=args.template_init, class_name_position=args.class_name_position, 
            num_learnable_prompt_tokens=args.num_learnable_prompt_tokens, transformer_width=512, transformer_heads=8, transformer_layers=12, 
            pc_feat_dims=pc_feat_dims, device=args.gpu, task=args.task)

    unfreeze_modules = [] # prompt_only
    if args.head_type > 0: # linear
        unfreeze_modules += ['point_encoder.blocks.blocks.11.norm2.weight', 'point_encoder.blocks.blocks.11.norm2.bias',
                        'point_encoder.blocks.blocks.11.mlp.fc2.weight', 'point_encoder.blocks.blocks.11.mlp.fc2.bias']
    if args.head_type > 1:  # mlp
        unfreeze_modules += ['point_encoder.blocks.blocks.11.norm1.weight', 'point_encoder.blocks.blocks.11.norm1.bias', 
                        'point_encoder.blocks.blocks.11.mlp.fc1.weight', 'point_encoder.blocks.blocks.11.mlp.fc1.bias']
    if args.head_type > 2: # atten_block
        unfreeze_modules += ['point_encoder.blocks.blocks.11.attn.qkv.weight', 'point_encoder.blocks.blocks.11.attn.proj.weight',
                            'point_encoder.blocks.blocks.11.attn.proj.bias']

    if not args.evaluate_3d:
        # load the pretrained model
        if args.ulip2:
            pretrain_point_model = torch.load('./data/pretrained_models/pointbert_ulip2.pt', map_location=torch.device('cpu'))
        else:
            pretrain_point_model = torch.load('./data/pretrained_models/pointbert.pt', map_location=torch.device('cpu'))
        pretrain_point_model_params = pretrain_point_model['state_dict']
        pretrain_point_model_params = {param_name.replace('module.', ''): param for param_name, param in
                                      pretrain_point_model_params.items()}
        
        pretrain_slip_model = torch.load('./data/initialize_models/slip_base_100ep.pt', map_location=torch.device('cpu'))
        pretrain_slip_model_params = pretrain_slip_model['state_dict']
        pretrain_slip_model_params = {param_name.replace('module.', ''): param for param_name, param in
                                      pretrain_slip_model_params.items()}

        for name, param in model.named_parameters():    # `model` does not have parameter for image encoder
            if name == 'prompt_learner.learnable_tokens' or 'point_encoder.cls_head' in name: 
                continue

            if name in unfreeze_modules:  # 打开 transformer 最后一个 block，让参数更新
                continue

            if name in pretrain_point_model_params:
                if isinstance(pretrain_point_model_params[name], Parameter):
                    param_new = pretrain_point_model_params[name].data
                else:
                    param_new = pretrain_point_model_params[name]
            else:
                if isinstance(pretrain_slip_model_params[name], Parameter):
                    param_new = pretrain_slip_model_params[name].data
                else:
                    param_new = pretrain_slip_model_params[name]

            param.requires_grad = False
            print('load {} and freeze'.format(name))
            param.data.copy_(param_new)

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('\n====================\n\tNumber of learnable params:', params, '\n====================\n')

    return model


def ULIP_PointBERT_partseg(args):
    CUR_DIR = os.path.dirname(os.path.abspath(__file__)) # current dir
    PROJ_DIR = os.path.dirname(CUR_DIR) # project dir
    # vision_model = timm.create_model('vit_base_patch16_224', num_classes=0)

    # =====================================================================
    # import the 3D backbone and specify the output point cloud feature dimension
    from models.pointbert.point_encoder import PointTransformer_partseg
    config_addr = os.path.join(PROJ_DIR, 'models/pointbert/PointTransformer_8192point.yaml')
    config = cfg_from_yaml_file(config_addr)
    point_encoder = PointTransformer_partseg(config.model, args=args)
    pc_feat_dims = 128
    # =====================================================================

    model = ULIP_WITH_IMAGE(embed_dim=512, point_encoder=point_encoder, context_length=77, 
            vocab_size=49408, classnames=args.classnames, template_init=args.template_init, class_name_position=args.class_name_position, 
            num_learnable_prompt_tokens=args.num_learnable_prompt_tokens, transformer_width=512, transformer_heads=8, transformer_layers=12, 
            pc_feat_dims=pc_feat_dims, device=args.gpu, task=args.task)

    count = 0
    if not args.evaluate_3d:
        # load the pretrained model
        if args.ulip2:
            pretrain_point_model = torch.load(os.path.join(PROJ_DIR, 'data/pretrained_models/pointbert_ulip2.pt'), map_location=torch.device('cpu'))
        else:
            pretrain_point_model = torch.load(os.path.join(PROJ_DIR, 'data/pretrained_models/pointbert.pt'), map_location=torch.device('cpu'))
        pretrain_point_model_params = pretrain_point_model['state_dict']
        pretrain_point_model_params = {param_name.replace('module.', ''): param for param_name, param in
                                      pretrain_point_model_params.items()}
        
        pretrain_slip_model = torch.load(os.path.join(PROJ_DIR, 'data/initialize_models/slip_base_100ep.pt'), map_location=torch.device('cpu'))
        pretrain_slip_model_params = pretrain_slip_model['state_dict']
        pretrain_slip_model_params = {param_name.replace('module.', ''): param for param_name, param in
                                      pretrain_slip_model_params.items()}

        for name, param in model.named_parameters():
            if name.startswith('prompt_learner'):
                count += 1
                print('------ prompt_learner params:', name)

            elif name.startswith('point_encoder.'):
                if name in pretrain_point_model_params:
                    param.requires_grad = False
                    print('load {} and freeze'.format(name))
                else:
                    count += 1
                    print('------ pointbert_partseg params:', name)

            else:   # image and text encoder
                param.requires_grad = False
                print('load {} and freeze'.format(name))

        print(f'>>>>>> {count}')

    return model


def ULIP_PN_NEXT(args):
    # vision_model = timm.create_model('vit_base_patch16_224', num_classes=0)

    # =====================================================================
    # import the 3D backbone and specify the output point cloud feature dimension
    from models.pointnext.pointnext import PointNEXT
    point_encoder = PointNEXT()
    pc_feat_dims = 256
    # =====================================================================

    model = ULIP_WITH_IMAGE(embed_dim=512, point_encoder=point_encoder, context_length=77, 
            vocab_size=49408, classnames=args.classnames, template_init=args.template_init, class_name_position=args.class_name_position, 
            num_learnable_prompt_tokens=args.num_learnable_prompt_tokens, transformer_width=512, transformer_heads=8, transformer_layers=12, 
            pc_feat_dims=pc_feat_dims, device=args.gpu, task=args.task)

    if not args.evaluate_3d:
        # load the pretrained model
        pretrain_point_model = torch.load('./data/pretrained_models/pointnext.pt', map_location=torch.device('cpu'))
        pretrain_point_model_params = pretrain_point_model['state_dict']
        pretrain_point_model_params = {param_name.replace('module.', ''): param for param_name, param in
                                      pretrain_point_model_params.items()}
        
        pretrain_slip_model = torch.load('./data/initialize_models/slip_base_100ep.pt', map_location=torch.device('cpu'))
        pretrain_slip_model_params = pretrain_slip_model['state_dict']
        pretrain_slip_model_params = {param_name.replace('module.', ''): param for param_name, param in
                                      pretrain_slip_model_params.items()}

        for name, param in model.named_parameters():
            if name == 'prompt_learner.learnable_tokens': 
                continue

            if name in pretrain_point_model_params:
                if isinstance(pretrain_point_model_params[name], Parameter):
                    param_new = pretrain_point_model_params[name].data
                else:
                    param_new = pretrain_point_model_params[name]
            else:
                if isinstance(pretrain_slip_model_params[name], Parameter):
                    param_new = pretrain_slip_model_params[name].data
                else:
                    param_new = pretrain_slip_model_params[name]

            param.requires_grad = False
            print('load {} and freeze'.format(name))
            param.data.copy_(param_new)

    return model


def ULIP_CUSTOMIZED(args):
    # vision_model = timm.create_model('vit_base_patch16_224', num_classes=0)

    # =====================================================================
    # This is a sample template to pre-train your customized 3D backbones, please modify this part accordingly!
    from models.customized_backbone.customized_backbone import CUSTOMIZED_BACKBONE
    point_encoder = CUSTOMIZED_BACKBONE()
    # We assume you might have different point cloud output feature dimension,
    # we added a projecting layer to unify the point cloud output dimension before doing the multimodal alignment,
    # please change the output feature dimension here.
    pc_feat_dims = 512
    # =====================================================================

    model = ULIP_WITH_IMAGE(embed_dim=512, point_encoder=point_encoder, 
                            context_length=77, vocab_size=49408, template_init=args.template_init, 
                            class_name_position=args.class_name_position, num_learnable_prompt_tokens=args.num_learnable_prompt_tokens,
                            transformer_width=512, transformer_heads=8, transformer_layers=12, pc_feat_dims=pc_feat_dims)

    if not args.evaluate_3d:
        # load the pretrained model
        pretrain_slip_model = torch.load('./data/initialize_models/slip_base_100ep.pt', map_location=torch.device('cpu'))
        pretrain_slip_model_params = pretrain_slip_model['state_dict']
        pretrain_slip_model_params = {param_name.replace('module.', ''): param for param_name, param in
                                      pretrain_slip_model_params.items()}

        for name, param in model.named_parameters():
            if name not in pretrain_slip_model_params:
                continue

            if isinstance(pretrain_slip_model_params[name], Parameter):
                param_new = pretrain_slip_model_params[name].data
            else:
                param_new = pretrain_slip_model_params[name]

            param.requires_grad = False
            print('load {} and freeze'.format(name))
            param.data.copy_(param_new)

    return model