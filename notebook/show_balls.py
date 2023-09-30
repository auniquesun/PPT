""" Original Author: Haoqiang Fan """
import cv2
import sys
import os
import json
import numpy as np
import ctypes as ct

import torch
from torch.utils.data import DataLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
print('BASE_DIR:', BASE_DIR)

ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
print('ROOT_DIR:', ROOT_DIR)

from models.ULIP_models import ULIP_PointBERT_partseg
from utils.utils import get_dataset, to_categorical
from utils.tokenizer import SimpleTokenizer


part2category = { 0:'Airplane', 1:'Airplane', 2:'Airplane', 3:'Airplane', 4:'Bag', 5:'Bag', 6:'Cap', 7:'Cap', 
                8:'Car', 9:'Car', 10:'Car', 11:'Car', 12:'Chair', 13:'Chair', 14:'Chair', 15:'Chair', 
                16:'Earphone', 17:'Earphone', 18:'Earphone', 19:'Guitar', 20:'Guitar', 21:'Guitar', 22:'Knife', 23:'Knife',
                24:'Lamp', 25:'Lamp', 26:'Lamp', 27:'Lamp', 28:'Laptop', 29:'Laptop', 30:'Motorbike', 31:'Motorbike',
                32:'Motorbike', 33:'Motorbike', 34:'Motorbike', 35:'Motorbike', 36:'Mug', 37:'Mug', 38:'Pistol', 39:'Pistol',
                40:'Pistol', 41:'Rocket', 42:'Rocket', 43:'Rocket', 44:'Skateboard', 45:'Skateboard', 46:'Skateboard',
                47:'Table', 48:'Table', 49:'Table'}
category2part = {'Airplane': [0, 1, 2, 3], 'Bag': [4, 5], 'Cap': [6, 7], 'Car': [8, 9, 10, 11], 'Chair': [12, 13, 14, 15], 
                'Earphone': [16, 17, 18], 'Guitar': [19, 20, 21], 'Knife': [22, 23], 'Lamp': [24, 25, 26, 27], 'Laptop': [28, 29], 
                'Motorbike': [30, 31, 32, 33, 34, 35], 'Mug': [36, 37], 'Pistol': [38, 39, 40], 'Rocket': [41, 42, 43], 
                'Skateboard': [44, 45, 46], 'Table': [47, 48, 49]}

cmap = np.array([[1.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                    [3.12493437e-02, 1.00000000e+00, 1.31250131e-06],
                    [0.00000000e+00, 6.25019688e-02, 1.00000000e+00],
                    [1.00000000e+00, 0.00000000e+00, 9.37500000e-02],
                    [1.00000000e+00, 0.00000000e+00, 9.37500000e-02],
                    [1.00000000e+00, 0.00000000e+00, 9.37500000e-02],
                    [1.00000000e+00, 0.00000000e+00, 9.37500000e-02],
                    [1.00000000e+00, 0.00000000e+00, 9.37500000e-02],
                    [1.00000000e+00, 0.00000000e+00, 9.37500000e-02],
                    [1.00000000e+00, 0.00000000e+00, 9.37500000e-02]])

showsz = 800
mousex, mousey = 0.5, 0.5
zoom = 1.0
changed = True


def onmouse(*args):
    global mousex, mousey, changed
    y = args[1]
    x = args[2]
    mousex = x / float(showsz)
    mousey = y / float(showsz)
    changed = True


cv2.namedWindow('show3d')
# cv2.moveWindow('show3d', 0, 0)
cv2.setMouseCallback('show3d', onmouse)

dll = np.ctypeslib.load_library(os.path.join(BASE_DIR, 'render_balls'), '.')


def showpoints(xyz, c_gt=None, c_pred=None, waittime=0, showrot=False, magnifyBlue=0, freezerot=False,
               background=(0, 0, 0), normalizecolor=True, ballradius=10):
    global showsz, mousex, mousey, zoom, changed
    xyz = xyz - xyz.mean(axis=0)
    radius = ((xyz ** 2).sum(axis=-1) ** 0.5).max()
    xyz /= (radius * 2.2) / showsz
    if c_gt is None:
        c0 = np.zeros((len(xyz),), dtype='float32') + 255
        c1 = np.zeros((len(xyz),), dtype='float32') + 255
        c2 = np.zeros((len(xyz),), dtype='float32') + 255
    else:
        c0 = c_gt[:, 0]
        c1 = c_gt[:, 1]
        c2 = c_gt[:, 2]

    if normalizecolor:
        c0 /= (c0.max() + 1e-14) / 255.0
        c1 /= (c1.max() + 1e-14) / 255.0
        c2 /= (c2.max() + 1e-14) / 255.0

    c0 = np.require(c0, 'float32', 'C')
    c1 = np.require(c1, 'float32', 'C')
    c2 = np.require(c2, 'float32', 'C')

    show = np.zeros((showsz, showsz, 3), dtype='uint8')

    def render():
        rotmat = np.eye(3)
        if not freezerot:
            xangle = (mousey - 0.5) * np.pi * 1.2
        else:
            xangle = 0
        rotmat = rotmat.dot(np.array([
            [1.0, 0.0, 0.0],
            [0.0, np.cos(xangle), -np.sin(xangle)],
            [0.0, np.sin(xangle), np.cos(xangle)],
        ]))
        if not freezerot:
            yangle = (mousex - 0.5) * np.pi * 1.2
        else:
            yangle = 0
        rotmat = rotmat.dot(np.array([
            [np.cos(yangle), 0.0, -np.sin(yangle)],
            [0.0, 1.0, 0.0],
            [np.sin(yangle), 0.0, np.cos(yangle)],
        ]))
        rotmat *= zoom
        nxyz = xyz.dot(rotmat) + [showsz / 2, showsz / 2, 0]

        ixyz = nxyz.astype('int32')
        show[:] = background
        dll.render_ball(
            ct.c_int(show.shape[0]),
            ct.c_int(show.shape[1]),
            show.ctypes.data_as(ct.c_void_p),
            ct.c_int(ixyz.shape[0]),
            ixyz.ctypes.data_as(ct.c_void_p),
            c0.ctypes.data_as(ct.c_void_p),
            c1.ctypes.data_as(ct.c_void_p),
            c2.ctypes.data_as(ct.c_void_p),
            ct.c_int(ballradius)
        )

        if magnifyBlue > 0:
            show[:, :, 0] = np.maximum(show[:, :, 0], np.roll(show[:, :, 0], 1, axis=0))
            if magnifyBlue >= 2:
                show[:, :, 0] = np.maximum(show[:, :, 0], np.roll(show[:, :, 0], -1, axis=0))
            show[:, :, 0] = np.maximum(show[:, :, 0], np.roll(show[:, :, 0], 1, axis=1))
            if magnifyBlue >= 2:
                show[:, :, 0] = np.maximum(show[:, :, 0], np.roll(show[:, :, 0], -1, axis=1))
        if showrot:
            cv2.putText(show, 'xangle %d' % (int(xangle / np.pi * 180)), (30, showsz - 30), 0, 0.5,
                        cv2.cv.CV_RGB(255, 0, 0))
            cv2.putText(show, 'yangle %d' % (int(yangle / np.pi * 180)), (30, showsz - 50), 0, 0.5,
                        cv2.cv.CV_RGB(255, 0, 0))
            cv2.putText(show, 'zoom %d%%' % (int(zoom * 100)), (30, showsz - 70), 0, 0.5, cv2.cv.CV_RGB(255, 0, 0))

    changed = True
    while True:
        if changed:
            render()
            changed = False
        cv2.imshow('show3d', show)

        if waittime == 0:
            cmd = cv2.waitKey(10) % 256
        else:
            cmd = cv2.waitKey(waittime) % 256
        if cmd == ord('q'):
            break
        elif cmd == ord('Q'):
            sys.exit(0)

        if cmd == ord('t') or cmd == ord('p'):
            if cmd == ord('t'):
                if c_gt is None:
                    c0 = np.zeros((len(xyz),), dtype='float32') + 255
                    c1 = np.zeros((len(xyz),), dtype='float32') + 255
                    c2 = np.zeros((len(xyz),), dtype='float32') + 255
                else:
                    c0 = c_gt[:, 0]
                    c1 = c_gt[:, 1]
                    c2 = c_gt[:, 2]
            else:
                if c_pred is None:
                    c0 = np.zeros((len(xyz),), dtype='float32') + 255
                    c1 = np.zeros((len(xyz),), dtype='float32') + 255
                    c2 = np.zeros((len(xyz),), dtype='float32') + 255
                else:
                    c0 = c_pred[:, 0]
                    c1 = c_pred[:, 1]
                    c2 = c_pred[:, 2]
            if normalizecolor:
                c0 /= (c0.max() + 1e-14) / 255.0
                c1 /= (c1.max() + 1e-14) / 255.0
                c2 /= (c2.max() + 1e-14) / 255.0
            c0 = np.require(c0, 'float32', 'C')
            c1 = np.require(c1, 'float32', 'C')
            c2 = np.require(c2, 'float32', 'C')
            changed = True

        if cmd == ord('n'):
            zoom *= 1.1
            changed = True
        elif cmd == ord('m'):
            zoom /= 1.1
            changed = True
        elif cmd == ord('r'):
            zoom = 1.0
            changed = True
        elif cmd == ord('s'):
            cv2.imwrite(f'images/pred_{args.class_choice}_2.png', show)
        if waittime != 0:
            break
    return cmd


def main(args):
    '''
    代码逻辑：
        模型加载数据，前向传播，接收的都是tensor，这些tensor是放在batch里的，
            但显示图像要一张张来，所以生成预测后再来个循环，遍历循环中每个实例
    '''

    with open('../data/labels.json') as fin:
        labels = json.load(fin)
        args.classnames = labels[args.dataset_name]

    # 1. define model
    model = ULIP_PointBERT_partseg(args).to(args.gpu)

    # 2. load model weights
    ckpt_path = os.path.join('../outputs', args.proj_name, args.exp_name, 'checkpoint_best.pt')
    saved_data = torch.load(ckpt_path, map_location=torch.device(args.gpu))
    # --- 2.1 well-trained weights of soft prompts
    model.prompt_learner.load_state_dict(saved_data['state_dict_prompt'])
    # --- 2.2 well-trained weights of the partseg model
    model.point_encoder.load_state_dict(saved_data['state_dict_partseg'])

    # 3. load data
    # 加载数据有个问题，就是 config 怎么往进传，模仿一下 main_cls.py 的做法
    tokenizer = SimpleTokenizer()
    test_dataset = get_dataset(None, tokenizer, args, 'test')
    print('------ len(test_dataset)', len(test_dataset))

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=True, # set `shuffle=True` for random shapes
        num_workers=args.workers, pin_memory=True, drop_last=False)

    # 4. inference on test data
    flag = False
    for inputs in test_loader:
        points = inputs[0].to(args.gpu)
        # cls_label: [batch,]
        obj_label = inputs[1].to(args.gpu)
        # part_label: [batch, npoints]
        partseg_label = inputs[2].long().to(args.gpu)

        # obj_label: [batch],    partseg_label: [batch, num_points]
        batch_size, num_points, _ = points.size()

        for i in range(batch_size):
            part = partseg_label[i][0].item()
            # find target class == `i-th` instance in this batch
            if part2category[part] == args.class_choice: 
                flag = True
                break

        if flag:
            # partseg_pred: [batch_size, num_points, num_part_classes]
            partseg_pred = model(points, to_categorical(obj_label, 16, args))

            refined_pred = torch.zeros(batch_size, num_points, dtype=torch.int32, device=partseg_pred.device)
            for j in range(batch_size):
                # partseg_label[i, 0] is a tensor, it should be converted to an integer to serve as an index
                idx = partseg_label[j, 0].item()
                cat = part2category[idx]
                logits = partseg_pred[j, :, :]
                refined_pred[j, :] = torch.argmax(logits[:, category2part[cat]], dim=1) + category2part[cat][0]
            
            points = points[i].cpu().numpy()
            gt_label = partseg_label[i] - partseg_label[i].min()
            gt_label = gt_label.cpu().numpy()
            pred_label = refined_pred[i] - refined_pred[i].min()
            pred_label = pred_label.cpu().numpy()

            break

    if flag:
        # points: [2048, 3], gt_label: [2048], pred_label: [2048]
        gt = cmap[gt_label, :]
        pred = cmap[pred_label, :]
        showpoints(points, gt, c_pred=pred, waittime=0, showrot=False, magnifyBlue=0, freezerot=False,
                background=(255, 255, 255), normalizecolor=True, ballradius=args.ballradius)
    else:
        print(f'There is no {args.class_choice} in test_loader!')


if __name__ == '__main__':
    from parser import args
    print('\n', args, '\n', )
    main(args)