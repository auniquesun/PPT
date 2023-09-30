'''
 * Adapted from ULIP (https://github.com/salesforce/ULIP)
 * By Hongyu Sun
'''
import os, sys
import random
import numpy as np

import h5py
from collections import defaultdict

import yaml
from easydict import EasyDict

import torch
import torch.utils.data as data

from utils.io import IO
from utils.build import DATASETS
from utils.logger import *
from utils.build import build_dataset_from_cfg
import json
import pickle
from PIL import Image


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

def rotate_point_cloud(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data

def random_point_dropout(batch_pc, max_dropout_ratio=0.875):
    ''' batch_pc: BxNx3 '''
    for b in range(batch_pc.shape[0]):
        dropout_ratio =  np.random.random()*max_dropout_ratio # 0~0.875
        drop_idx = np.where(np.random.random((batch_pc.shape[1]))<=dropout_ratio)[0]
        if len(drop_idx)>0:
            batch_pc[b,drop_idx,:] = batch_pc[b,0,:] # set to the first point
    return batch_pc

def random_scale_point_cloud(batch_data, scale_low=0.8, scale_high=1.25):
    """ Randomly scale the point cloud. Scale is per point cloud.
        Input:
            BxNx3 array, original batch of point clouds
        Return:
            BxNx3 array, scaled batch of point clouds
    """
    B, N, C = batch_data.shape
    scales = np.random.uniform(scale_low, scale_high, B)
    for batch_index in range(B):
        batch_data[batch_index,:,:] *= scales[batch_index]
    return batch_data

def shift_point_cloud(batch_data, shift_range=0.1):
    """ Randomly shift point cloud. Shift is per point cloud.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, shifted batch of point clouds
    """
    B, N, C = batch_data.shape
    shifts = np.random.uniform(-shift_range, shift_range, (B,3))
    for batch_index in range(B):
        batch_data[batch_index,:,:] += shifts[batch_index,:]
    return batch_data

def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
    jittered_data += batch_data
    return jittered_data

def rotate_perturbation_point_cloud(batch_data, angle_sigma=0.06, angle_clip=0.18):
    """ Randomly perturb the point clouds by small rotations
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        angles = np.clip(angle_sigma*np.random.randn(3), -angle_clip, angle_clip)
        Rx = np.array([[1,0,0],
                       [0,np.cos(angles[0]),-np.sin(angles[0])],
                       [0,np.sin(angles[0]),np.cos(angles[0])]])
        Ry = np.array([[np.cos(angles[1]),0,np.sin(angles[1])],
                       [0,1,0],
                       [-np.sin(angles[1]),0,np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]),-np.sin(angles[2]),0],
                       [np.sin(angles[2]),np.cos(angles[2]),0],
                       [0,0,1]])
        R = np.dot(Rz, np.dot(Ry,Rx))
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), R)
    return rotated_data

def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
def load_modelnet_data(data_path):
    all_data = []
    all_label = []
    with open(data_path, "r") as f:
        for h5_name in f.readlines():
            f = h5py.File(h5_name.strip(), 'r')
            data = f['data'][:].astype('float32')
            label = f['label'][:].astype('int64')
            f.close()
            all_data.append(data)
            all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)

    return all_data, all_label

def load_scanobjectnn_data(root, type, partition):
    all_data = []
    all_label = []

    if type != 'hardest':   # obj_only or obj_bg
        h5_name = os.path.join(root, type, f'{partition}_objectdataset.h5')
    else:   
        h5_name = os.path.join(root, type, f'{partition}_objectdataset_augmentedrot_scale75.h5')

    f = h5py.File(h5_name, mode="r")
    data = f['data'][:].astype('float32')
    label = f['label'][:].astype('int64')
    f.close()
    all_data.append(data)
    all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label

def read_mn_so_data(classnames, points, labels):
    items = []

    for i, pc in enumerate(points):
        label = int(labels[i])
        classname = classnames[label]

        item = {'pc': pc, 'label': label, 'classname': classname}
        items.append(item)
    
    return items

def generate_fewshot_dataset(data_source, num_shots=-1, repeat=True):
    """Generate a few-shot dataset (typically for the training set).

    This function is useful when one wants to evaluate a model
    in a few-shot learning setting where each class only contains
    a few number of images.

    Args:
        data_source: a list of Datum objects.
        num_shots (int): number of instances per class to sample.
        repeat (bool): repeat images if needed.
    """
    if num_shots < 1:
        return data_source

    tracker = split_dataset_by_label(data_source)
    fewshot_dataset = []

    for items in tracker.values():
        if len(items) >= num_shots:
            sampled_items = random.sample(items, num_shots)
        else:
            if repeat:
                sampled_items = random.choices(items, k=num_shots)
            else:
                sampled_items = items
        # 相当于把所有类别（每类包含 num_shots 个样本）全部放到 fewshot_dataset 这个list里
        fewshot_dataset.extend(sampled_items)

    return fewshot_dataset

def split_dataset_by_label(data_source):
    """Split a dataset, i.e. a list of Datum objects,
    into class-specific groups stored in a dictionary.

    Args:
        data_source (list): a list of Datum objects.
    """
    output = defaultdict(list)

    for item in data_source:
        label = item['label']
        output[label].append(item)

    return output


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
PROJ_DIR = os.path.dirname(BASE_DIR)

@DATASETS.register_module()
class ModelNet(data.Dataset):
    def __init__(self, config):
        self.root = config.DATA_PATH
        self.npoints = config.npoints
        self.use_normals = config.USE_NORMALS
        self.num_category = config.NUM_CATEGORY
        self.tokenizer = config.tokenizer   # SimpleTokenizer
        self.uniform = True
        self.generate_from_raw_data = False
        self.subset = config.split
        assert (self.subset == 'train' or self.subset == 'test')
        self.template_init = config.template_init
        self.num_learnable_prompt_tokens = config.num_learnable_prompt_tokens
        # TODO: disable for backbones except for PointNEXT!!!
        self.use_height = config.use_height

        self.catfile = os.path.join(self.root, f'modelnet{self.num_category}_shape_names.txt')
        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        self.save_path = os.path.join(self.root, 'modelnet%d_%s_%dpts_fps.dat' % (self.num_category, self.subset, self.npoints))
        print_log('Load processed data from %s...' % self.save_path, logger='ModelNet')

        with open(self.save_path, 'rb') as f:
            self.list_of_points, self.list_of_labels = pickle.load(f)

    def __len__(self):
        return len(self.list_of_labels)

    def _get_item(self, index):
        points, label = self.list_of_points[index], self.list_of_labels[index]
        
        if  self.npoints < points.shape[0]:
            points = farthest_point_sample(points, self.npoints)

        if not self.use_normals:
            points = points[:, 0:3]
        points[:, 0:3] = pc_normalize(points[:, 0:3])

        label = int(label)  # before `int(.)`, `label` is a numpy array with one `numpy.int32` number
        return points, label

    def __getitem__(self, index):
        pointcloud, label = self._get_item(index)

        if self.subset == 'train':
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)

        if self.use_height:
            gravity_dim = 1 # (x,y,z), `y` is indexed by 1
            height_array = pointcloud[:, gravity_dim:gravity_dim + 1] - pointcloud[:, gravity_dim:gravity_dim + 1].min()
            pointcloud = np.concatenate((pointcloud, height_array), axis=1)

        # print('\n---------- after add height information for point cloud ----------\n')
        # print('pointcloud.shape:', pointcloud.shape)
        # print('\n---------- ----------\n')

        label_name = self.cat[label]
        # print('---------- label_name:', label_name)

        return pointcloud, label, label_name


@DATASETS.register_module()
class ModelNet_fs(ModelNet):
    '''This is a definition of fewshot version of ModelNet'''
    def __init__(self, config):
        self.root = config.DATA_PATH
        self.npoints = config.npoints
        self.use_normals = config.USE_NORMALS
        self.num_category = config.NUM_CATEGORY
        self.subset = config.split
        assert (self.subset == 'train' or self.subset == 'test')
        num_shots = config.nshots
        # TODO: disable for backbones except for PointNEXT!!!
        self.use_height = config.use_height

        self.catfile = os.path.join(self.root, f'modelnet{self.num_category}_shape_names.txt')
        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        self.save_path = os.path.join(self.root, 'modelnet%d_%s_%dpts_fps.dat' % (self.num_category, self.subset, self.npoints))
        print_log('Load processed data from %s...' % self.save_path, logger='ModelNet')

        with open(self.save_path, 'rb') as f:
            self.list_of_points, self.list_of_labels = pickle.load(f)

        if self.subset == 'train':
            train = read_mn_so_data(self.cat, self.list_of_points, self.list_of_labels)
            self.data_source = generate_fewshot_dataset(train, num_shots=num_shots)
        else:
            self.data_source = read_mn_so_data(self.cat, self.list_of_points, self.list_of_labels)

        print('\n============= Entering ModelNet_fs =============\n')

    def __len__(self):
        return len(self.data_source)

    def _get_item(self, index):
        item = self.data_source[index]
        points, label, label_name = item['pc'], item['label'], item['classname']
        
        if self.npoints < points.shape[0]:
            points = farthest_point_sample(points, self.npoints)

        if not self.use_normals:
            points = points[:, 0:3]
        points[:, 0:3] = pc_normalize(points[:, 0:3])

        return points, label, label_name
    
    def __getitem__(self, index):
        pointcloud, label, label_name = self._get_item(index)

        if self.subset == 'train':
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)

        if self.use_height:
            gravity_dim = 1 # (x,y,z), `y` is indexed by 1
            height_array = pointcloud[:, gravity_dim:gravity_dim + 1] - pointcloud[:, gravity_dim:gravity_dim + 1].min()
            pointcloud = np.concatenate((pointcloud, height_array), axis=1)

        return pointcloud, label, label_name
    

@DATASETS.register_module()
class ScanObjectNN(data.Dataset):
    # this is the `hardest` mode of ScanObjectNN
    def __init__(self, config):
        self.root = config.DATA_PATH
        self.sonn_type = config.sonn_type
        self.partition = config.split
        self.data, self.label = load_scanobjectnn_data(self.root, self.sonn_type, self.partition)
        self.num_points = config.npoints
        # TODO: disable for backbones except for PointNEXT!!!
        self.use_height = config.use_height

        self.shape_names_addr = os.path.join(self.root, 'shape_names.txt')
        with open(self.shape_names_addr) as file:
            lines = file.readlines()
            self.shape_names = [line.rstrip() for line in lines]

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        label_name = self.shape_names[int(label)]
        if self.partition == 'train':
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)
        
        if self.use_height:
            gravity_dim = 1 # (x,y,z), `y` is indexed by 1
            height_array = pointcloud[:, gravity_dim:gravity_dim + 1] - pointcloud[:, gravity_dim:gravity_dim + 1].min()
            pointcloud = np.concatenate((pointcloud, height_array), axis=1)

        return pointcloud, label, label_name

    def __len__(self):
        return self.data.shape[0]
    

@DATASETS.register_module()
class ScanObjectNN_fs(data.Dataset):
    def __init__(self, config):
        self.root = config.DATA_PATH
        self.sonn_type = config.sonn_type
        self.partition = config.split
        self.num_points = config.npoints
        self.use_height = config.use_height
        num_shots = config.nshots

        shape_names_addr = os.path.join(self.root, 'shape_names.txt')
        with open(shape_names_addr) as file:
            lines = file.readlines()
            shape_names = [line.rstrip() for line in lines]

        data, label = load_scanobjectnn_data(self.root, self.sonn_type, self.partition)
        data = read_mn_so_data(shape_names, data, label)

        if self.partition == 'train':
            self.data_source = generate_fewshot_dataset(data, num_shots=num_shots)
        else:
            self.data_source = data

        print('\n============= Entering ScanObject_fs =============\n')

    def __len__(self):
        return len(self.data_source)
    
    def __getitem__(self, idx):
        pointcloud = self.data_source[idx]['pc'][:self.num_points]
        label = self.data_source[idx]['label']
        label_name = self.data_source[idx]['classname']

        if self.partition == 'train':
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)

        if self.use_height:
            gravity_dim = 1 # (x,y,z), `y` is indexed by 1
            height_array = pointcloud[:, gravity_dim:gravity_dim + 1] - pointcloud[:, gravity_dim:gravity_dim + 1].min()
            pointcloud = np.concatenate((pointcloud, height_array), axis=1)

        return pointcloud, label, label_name


@DATASETS.register_module()
class ShapeNet(data.Dataset):
    def __init__(self, config):

        self.data_root = config.DATA_PATH   # data/shapenet-55
        self.pc_path = config.PC_PATH       # data/shapenet-55/shapenet_pc
        self.subset = config.split         # train
        self.npoints = config.npoints       # 8192
        self.tokenizer = config.tokenizer   # SimpleTokenizer
        self.train_transform = config.train_transform   # transforms.Compose([RandomResizedCrop(224, scale=(0.5, 1.0)), ToTensor(), Normalize()])
        self.id_map_addr = os.path.join(config.DATA_PATH, 'taxonomy.json')  # data/shapenet-55/taxonomy.json
        self.rendered_image_addr = config.IMAGE_PATH    # data/shapenet-55/shapenet_image
        self.picked_image_type = ['', '_depth0001']
        self.picked_rotation_degrees = list(range(0, 360, 12))  # [0, 30, 60, ..., 330]
        self.picked_rotation_degrees = [(3 - len(str(degree))) * '0' + str(degree) if len(str(degree)) < 3 else str(degree) for degree in self.picked_rotation_degrees]
        self.template_init = config.template_init
        self.num_learnable_prompt_tokens = config.num_learnable_prompt_tokens

        with open(self.id_map_addr, 'r') as f:
            self.id_map = json.load(f)

        # 原来 prompt 是从这加载出来的
        self.prompt_template_addr = os.path.join('./data/templates.json')
        with open(self.prompt_template_addr) as f:
            # config.pretrain_dataset_prompt -> shapenet_64
            # self.templates is a `list`, containing predefining prompts
            self.templates = json.load(f)[config.pretrain_dataset_prompt]

        self.synset_id_map = {}
        for id_dict in self.id_map:
            # e.g., id_dict
            #   {'synsetId': '03809312', 'name': "aircraft", 'children': [], 'numInstances': 14}
            # e.g., synset_id: '03809312'
            synset_id = id_dict["synsetId"]
            self.synset_id_map[synset_id] = id_dict

        # data/shapenet-55/train.txt
        self.data_list_file = os.path.join(self.data_root, f'{self.subset}.txt')
        # data/shapenet-55/test.txt
        #   打开作者提供的这个文件，我发现里面只有 11 条数据，是作者搞错了，还是有什么蹊跷的地方？
        test_data_list_file = os.path.join(self.data_root, 'test.txt')

        self.sample_points_num = self.npoints   # 8192
        self.whole = config.get('whole')    # self.whole -> True

        print_log(f'[DATASET] sample out {self.sample_points_num} points', logger='ShapeNet-55')
        print_log(f'[DATASET] Open file {self.data_list_file}', logger='ShapeNet-55')
        with open(self.data_list_file, 'r') as f:
            # 训练集中所有数据文件路径
            lines = f.readlines()

        if self.whole:
            with open(test_data_list_file, 'r') as f:
                # 测试集中所有数据文件路径
                test_lines = f.readlines()

            print_log(f'[DATASET] Open file {test_data_list_file}', logger='ShapeNet-55')
            # 训练 + 测试集文件路径
            lines = test_lines + lines

        self.file_list = []
        for line in lines:
            line = line.strip()
            # 类别代码，e.g., '03809312'
            taxonomy_id = line.split('-')[0]
            # model_id, e.g., '10cfc2090a2ade124c3a35cee92bb95b'
            model_id = line[len(taxonomy_id) + 1:].split('.')[0]
            # file_list 每个元素是字典
            self.file_list.append({
                'taxonomy_id': taxonomy_id,
                'model_id': model_id,
                'file_path': line
            })
        print_log(f'[DATASET] {len(self.file_list)} instances were loaded', logger='ShapeNet-55')

        self.permutation = np.arange(self.npoints)

        self.uniform = True
        self.augment = True
        # =================================================
        # TODO: disable for backbones except for PointNEXT!!!
        self.use_height = config.use_height
        # =================================================

        if self.augment:
            print("using augmented point clouds.")

        # self.template = "a point cloud model of {}."

    def pc_norm(self, pc):
        """ pc: NxC, return NxC """

        # centroid: 1xC
        centroid = np.mean(pc, axis=0)
        # pc: NxC
        pc = pc - centroid
        # m: a single number
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        return pc

    def random_sample(self, pc, num):
        np.random.shuffle(self.permutation)
        pc = pc[self.permutation[:num]]
        return pc

    def __getitem__(self, idx):
        sample = self.file_list[idx]

        data = IO.get(os.path.join(self.pc_path, sample['file_path'])).astype(np.float32)

        if self.uniform and self.sample_points_num < data.shape[0]:
            # 还真的进行了采样，每次读数据进行这样的采样，运行速度很慢吧
            data = farthest_point_sample(data, self.sample_points_num)
        else:
            data = self.random_sample(data, self.sample_points_num)
        # 对点云坐标归一化
        data = self.pc_norm(data)

        # 数据增强操作
        if self.augment:
            data = random_point_dropout(data[None, ...])
            data = random_scale_point_cloud(data)
            data = shift_point_cloud(data)
            data = rotate_perturbation_point_cloud(data)
            data = rotate_point_cloud(data)
            data = data.squeeze()

        # 仅有 PointNeXt 使用高度信息
        if self.use_height:
            self.gravity_dim = 1
            height_array = data[:, self.gravity_dim:self.gravity_dim + 1] - data[:,
                                                                       self.gravity_dim:self.gravity_dim + 1].min()
            data = np.concatenate((data, height_array), axis=1)
            data = torch.from_numpy(data).float()
        else:
            # numpy 转换成 tensor
            data = torch.from_numpy(data).float()

        # ======================added by jerry====================
        # 取到 model 描述，e.g., 'airliner' or 'airplane,aeroplane,plane'
        captions = self.synset_id_map[sample['taxonomy_id']]['name']
        # captions: ['airplane','aeroplane','plane']
        captions = [caption.strip() for caption in captions.split(',') if caption.strip()]
        # 从中随机选一个, e.g., aeroplane
        shape_name = random.choice(captions)
        # 记录 shape_name 编码长度，后续在 PromptLearner 生成 prompts 时候会用到
        shape_name_length = len(self.tokenizer.encode(shape_name))

        # 用提供的 template 模板初始化
        if self.template_init != '':
            prompt_prefix = self.template_init.replace('_', ' ')
        else:
            prompt_prefix = " ".join(['X'] * self.num_learnable_prompt_tokens)
        shape_caption = prompt_prefix + " " + shape_name + "."
        # here `tokenized_captions` is a 1-d tensor with `context_length` elements
        tokenized_captions = self.tokenizer(shape_caption)
        # =======================added by jerry===================

        # 图像目录：data/shapenet-55/shapenet_image/03809312-10cfc2090a2ade124c3a35cee92bb95b/
        picked_model_rendered_image_addr = self.rendered_image_addr + '/' +\
                                           sample['taxonomy_id'] + '-' + sample['model_id'] + '/'
        # 随机选一个角度，找到对应的图像名
        # e.g.,  03809312-10cfc2090a2ade124c3a35cee92bb95b_r_300.png
        picked_image_name = sample['taxonomy_id'] + '-' + sample['model_id'] + '_r_' +\
                            str(random.choice(self.picked_rotation_degrees)) +\
                            random.choice(self.picked_image_type) + '.png'
        picked_image_addr = picked_model_rendered_image_addr + picked_image_name

        try:
            image = pil_loader(picked_image_addr)
            image = self.train_transform(image)
        except:
            raise ValueError("image is corrupted: {}".format(picked_image_addr))

        # sample['taxonomy_id']: 03809312  
        # sample['model_id']: 10cfc2090a2ade124c3a35cee92bb95b 
        # tokenized_captions: [context_length] 每个word在词表中的序号
        # data: [num_points, 3]
        # image: [3, 224, 224]  经过 self.train_transform 变换是这样的
        return sample['taxonomy_id'], sample['model_id'], tokenized_captions, shape_name_length, data, image

    def __len__(self):
        return len(self.file_list)


@DATASETS.register_module()
class ShapeNetPart(data.Dataset):
    def __init__(self, config):
        # npoints=2500, split='train', class_choice=None, normal_channel=False

        self.root = config.DATA_PATH    # data/shapenetpart
        self.npoints = config.npoints   # 2048
        self.split = config.split      # test
        self.class_choice = config.class_choice # 包含全部类别的 list
        self.normal_channel = config.normal_channel # False
        self.catfile = os.path.join(PROJ_DIR, self.root, 'synsetoffset2category.txt') # data/shapenetpart/synsetoffset2category.txt
        self.cat = {}

        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        self.cat = {k: v for k, v in self.cat.items()}  # {'Airplane': 02691156, ...}
        self.classes_original = dict(zip(self.cat, range(len(self.cat))))   # {'Airplane': 0, ...}

        if not self.class_choice is  None:
            self.cat = {k:v for k,v in self.cat.items() if k in self.class_choice}

        self.meta = {}
        with open(os.path.join(PROJ_DIR, self.root, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
            train_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(PROJ_DIR, self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
            val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(PROJ_DIR, self.root, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
            test_ids = set([str(d.split('/')[2]) for d in json.load(f)])

        for item in self.cat:   # {'Airplane': '02691156', 'Bag': '02773838', ...}
            # print('category', item)
            self.meta[item] = []
            dir_point = os.path.join(PROJ_DIR, self.root, self.cat[item]) # data/shapenetpart/02691156
            fns = sorted(os.listdir(dir_point))
            # print(fns[0][0:-4])
            if self.split == 'trainval':
                fns = [fn for fn in fns if ((fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))]
            elif self.split == 'train':
                fns = [fn for fn in fns if fn[0:-4] in train_ids]
            elif self.split == 'val':
                fns = [fn for fn in fns if fn[0:-4] in val_ids]
            elif self.split == 'test':
                fns = [fn for fn in fns if fn[0:-4] in test_ids]
            else:
                print('Unknown split: %s. Exiting..' % (self.split))
                exit(-1)

            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0])
                self.meta[item].append(os.path.join(dir_point, token + '.txt'))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn))

        self.classes = {}
        for i in self.cat.keys():
            self.classes[i] = self.classes_original[i]

        # Mapping from category ('Chair') to a list of int [10,11,12,13] as segmentation labels
        self.category2part = {'Airplane': [0, 1, 2, 3], 'Bag': [4, 5], 'Cap': [6, 7], 'Car': [8, 9, 10, 11], 'Chair': [12, 13, 14, 15], 
                    'Earphone': [16, 17, 18], 'Guitar': [19, 20, 21], 'Knife': [22, 23], 'Lamp': [24, 25, 26, 27], 'Laptop': [28, 29], 
                    'Motorbike': [30, 31, 32, 33, 34, 35], 'Mug': [36, 37], 'Pistol': [38, 39, 40], 'Rocket': [41, 42, 43], 
                    'Skateboard': [44, 45, 46], 'Table': [47, 48, 49]}
        self.part2category = { 0:'Airplane', 1:'Airplane', 2:'Airplane', 3:'Airplane', 4:'Bag', 5:'Bag', 6:'Cap', 7:'Cap', 
                8:'Car', 9:'Car', 10:'Car', 11:'Car', 12:'Chair', 13:'Chair', 14:'Chair', 15:'Chair', 
                16:'Earphone', 17:'Earphone', 18:'Earphone', 19:'Guitar', 20:'Guitar', 21:'Guitar', 22:'Knife', 23:'Knife',
                24:'Lamp', 25:'Lamp', 26:'Lamp', 27:'Lamp', 28:'Laptop', 29:'Laptop', 30:'Motorbike', 31:'Motorbike',
                32:'Motorbike', 33:'Motorbike', 34:'Motorbike', 35:'Motorbike', 36:'Mug', 37:'Mug', 38:'Pistol', 39:'Pistol',
                40:'Pistol', 41:'Rocket', 42:'Rocket', 43:'Rocket', 44:'Skateboard', 45:'Skateboard', 46:'Skateboard',
                47:'Table', 48:'Table', 49:'Table'}

        self.cache = {}  # from index to (point_set, cls, seg) tuple
        self.cache_size = 20000

    def __getitem__(self, index):
        if index in self.cache:
            point_set, cls, seg = self.cache[index]
        else:
            fn = self.datapath[index]
            cat = self.datapath[index][0]
            cls = self.classes[cat]
            cls = np.array([cls]).astype(np.int32)
            data = np.loadtxt(fn[1]).astype(np.float32)
            if not self.normal_channel:
                point_set = data[:, 0:3]
            else:
                point_set = data[:, 0:6]
            seg = data[:, -1].astype(np.int32)
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls, seg)
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

        choice = np.random.choice(len(seg), self.npoints, replace=True)
        # resample
        point_set = point_set[choice, :]
        seg = seg[choice]

        return point_set, cls, seg

    def __len__(self):
        return len(self.datapath)


import collections.abc as container_abcs
int_classes = int
from torch._six import string_classes

import re
default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")
np_str_obj_array_pattern = re.compile(r'[SaUO]')

def customized_collate_fn(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    elem = batch[0]
    elem_type = type(elem)

    if isinstance(batch, list):
        batch = [example for example in batch if example[4] is not None]

    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return customized_collate_fn([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int_classes):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, container_abcs.Mapping):
        return {key: customized_collate_fn([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(customized_collate_fn(samples) for samples in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [customized_collate_fn(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))

def merge_new_config(config, new_config):
    for key, val in new_config.items():
        if not isinstance(val, dict):
            if key == '_base_':
                with open(new_config['_base_'], 'r') as f:
                    try:
                        val = yaml.load(f, Loader=yaml.FullLoader)
                    except:
                        val = yaml.load(f)
                config[key] = EasyDict()
                merge_new_config(config[key], val)
            else:
                config[key] = val
                continue
        if key not in config:
            config[key] = EasyDict()
        merge_new_config(config[key], val)
    return config

def cfg_from_yaml_file(cfg_file):
    # e.g., cfg_file -> ./data/ShapeNet-55.yaml
    config = EasyDict()
    with open(cfg_file, 'r') as f:
        new_config = yaml.load(f, Loader=yaml.FullLoader)
    merge_new_config(config=config, new_config=new_config)
    return config

class Dataset_3D():
    def __init__(self, args, tokenizer, dataset_type, train_transform=None):
        if dataset_type not in ['train', 'val', 'test']:
            raise ValueError("not supported dataset type.")
        
        self.dataset_name = args.dataset_name
        self.tokenizer = tokenizer
        self.train_transform = train_transform
        self.dataset_prompt = args.dataset_prompt
        self.dataset_type = dataset_type

        with open(os.path.join(PROJ_DIR, 'data/dataset_catalog.json'), 'r') as f:
            self.dataset_catalog = json.load(f)
            self.dataset_config_dir = self.dataset_catalog[self.dataset_name]['config']
        self.build_3d_dataset(args, self.dataset_config_dir)    # 第二个参数仅是一个yaml文件目录，不要被下面函数定义迷惑

    def build_3d_dataset(self, args, dataset_config_dir):
        config = cfg_from_yaml_file(dataset_config_dir)
        if 'scanobjectnn' in self.dataset_name:
            config.sonn_type = args.sonn_type
        config.tokenizer = self.tokenizer
        config.train_transform = self.train_transform
        config.dataset_prompt = self.dataset_prompt
        config.split = self.dataset_type
        config.args = args
        config.use_height = args.use_height
        config.npoints = args.npoints
        config.nshots = args.nshots
        config.template_init = args.template_init
        config.num_learnable_prompt_tokens = args.num_learnable_prompt_tokens
        config_others = EasyDict({'subset': config.split, 'whole': True})
        self.dataset = build_dataset_from_cfg(config, config_others)
