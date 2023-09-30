'''
 * Copyright (c) 2023, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Le Xue
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import utils

class ULIPWithImageLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.labels = None
        self.last_local_batch_size = None

    def forward(self, outputs):
        # pc_embed:    [batch, embed_dim]
        pc_embed = outputs['pc_embed']
        # text_embed:  [batch, embed_dim]
        text_embed = outputs['text_embed']
        # image_embed: [batch, embed_dim]
        image_embed = outputs['image_embed']
        logit_scale = outputs['logit_scale']
        local_batch_size = pc_embed.size(0)

        if local_batch_size != self.last_local_batch_size:
            # 不明白 self.labels 为什么还和 utils.get_rank() 有关
            #   local_batch_size * utils.get_rank() 算出来的是什么？把 local_batch_size 看成是一个 interval，
            #       再看下面 utils.all_gather_batch 返回值的维度，就能理解含义了
            # --- PyTorch 对 dist.get_rank() 的解释 ---
            #   Returns the rank of the current process in the provided group or the default group if none was provided.
            #   Rank is a unique identifier assigned to each process within a distributed process group. 
            #   They are always consecutive integers ranging from 0 to world_size.
            #   self.labels: [batch]
            # --- option 1 ---
            # self.labels = local_batch_size * utils.get_rank() + torch.arange(   # 这里还用两个值相加吗？因为我下面不用所有GPU的batch算loss了，仅用当前GPU的batch
                # local_batch_size, device=pc_embed.device
            # )
            # --- option 2 ---
            self.labels = torch.arange(local_batch_size, device=pc_embed.device)
            self.last_local_batch_size = local_batch_size

        # normalized features，怎么又做一次归一化，文本已经做过一次了
        pc_embed = F.normalize(pc_embed, dim=-1, p=2)
        text_embed = F.normalize(text_embed, dim=-1, p=2)
        image_embed = F.normalize(image_embed, dim=-1, p=2)

        # --- option 1: gather features from all GPUs ---
        #   感觉这样做显存消耗太大了，可以仅考虑当前batch内 (pc, text, image)
        #   pc_embed_all:    [world_size*batch, embed_dim]
        #   text_embed_all:  [world_size*batch, embed_dim]
        #   image_embed_all: [world_size*batch, embed_dim]
        # pc_embed_all, text_embed_all, image_embed_all = \
            # utils.all_gather_batch([pc_embed, text_embed, image_embed])

        # --- option 2: gather features from all GPUs ---
        #   pc_embed_all:    [batch, embed_dim]
        #   text_embed_all:  [batch, embed_dim]
        #   image_embed_all: [batch, embed_dim]
        pc_embed_all, text_embed_all, image_embed_all = pc_embed, text_embed, image_embed

        # cosine similarity as logits
        #   理解一下这里为什么要正反方向算两次呢？而且一次用 pc_embed，一次用 text_embed_all，而不是 text_embed
        #   从下面 `logits_per_pc_text` 的维度就可以看出，是在算当前 batch 和 所有 batch 样本的特征相似度
        #   logits_per_pc_text: [batch, world_size*batch]   和所有batch乘是真的费计算，怪不得8块A100
        logits_per_pc_text = logit_scale * pc_embed @ text_embed_all.t()
        logits_per_text_pc = logit_scale * text_embed @ pc_embed_all.t()
        logits_per_pc_image = logit_scale * pc_embed @ image_embed_all.t()
        logits_per_image_pc = logit_scale * image_embed @ pc_embed_all.t()

        # 不理解为啥用的 cross entropy loss，显然应该是对比 loss 啊
        loss = (F.cross_entropy(logits_per_pc_text, self.labels) + F.cross_entropy(logits_per_text_pc, self.labels)) / 2 + \
                (F.cross_entropy(logits_per_pc_image, self.labels) + F.cross_entropy(logits_per_image_pc, self.labels)) / 2

        # compute accuracy
        #   为啥还要算准确性？后面返回值用到了
        with torch.no_grad():
            pred = torch.argmax(logits_per_pc_text, dim=-1)
            correct = pred.eq(self.labels).sum()
            pc_text_acc = 100 * correct / local_batch_size

            pred = torch.argmax(logits_per_pc_image, dim=-1)
            correct = pred.eq(self.labels).sum()
            pc_image_acc = 100 * correct / local_batch_size

        return {'loss': loss, 'ulip_loss': loss, 'ulip_pc_image_acc': pc_image_acc, 'ulip_pc_text_acc': pc_text_acc}
