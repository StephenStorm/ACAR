import multiprocessing as mp
mp.set_start_method('spawn', force=True)
import os
import argparse
import json
import pprint
import socket
import time
from easydict import EasyDict
import yaml
from tensorboardX import SummaryWriter

import torch
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from calc_mAP import run_evaluation
from datasets import ava, spatial_transforms, temporal_transforms
from distributed_utils import init_distributed
import losses
from models import AVA_model
from scheduler import get_scheduler
from utils import *
from PIL import Image
from datasets.ava import get_aug_info, batch_pad



from my_test.mmdet_test import get_person

def local_spatial_transform(spatial_transform, clip):
    if spatial_transform is not None:
        assert isinstance(spatial_transform, list)
                    
        init_size = clip[0].size[:2]
        clips, aug_info = [], []
        for st in spatial_transform:
            params = st.randomize_parameters()
            aug_info.append(get_aug_info(init_size, params))
        
            clips.append(torch.stack([st(img) for img in clip], 0).permute(1, 0, 2, 3))
    else:
        aug_info = [None]
        clips = [torch.stack(clip, 0).permute(1, 0, 2, 3)]
    return clips, aug_info


def read_img(root, video_name):
    image_dir = os.path.join(root, video_name)
    image_paths = os.listdir(image_dir)
    clip = []
    for image_path in image_paths:
        try:
            image_path_t = os.path.join(image_dir, image_path)
            with Image.open(image_path_t) as img:
                img = img.convert('RGB')
        except BaseException as e:
            raise RuntimeError('Caught "{}" when loading {}'.format(str(e), image_path))   
        clip.append(img)
    
    return clip



def main(config):
    video_root = '/opt/tiger/minist/datasets/test/'
    # video_names = os.listdir(video_root)
    video_name = 'v0d00fg10000c3rq96bc77ufsrjbdn4g'
    format_str = video_name + '_{}.jpg'
    out_file = open('result.txt', 'w')

    with open(config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    opt = EasyDict(config)

    # writer = SummaryWriter(os.path.join(opt.result_path, 'tb'))

    # create model
    model = AVA_model(opt.model)
    model.eval()
    model.cuda()
    val_aug = opt.val.augmentation

    transform_choices, total_choices = [], 1
    for aug in val_aug.spatial:
        kwargs_list = aug.get('kwargs', {})
        if not isinstance(kwargs_list, list):
            kwargs_list = [kwargs_list]
        cur_choices = [getattr(spatial_transforms, aug.type)(**kwargs) for kwargs in kwargs_list]
        transform_choices.append(cur_choices)
        total_choices *= len(cur_choices)

    spatial_transform = []
    for choice_idx in range(total_choices):
        idx, transform = choice_idx, []
        for cur_choices in transform_choices:
            n_choices = len(cur_choices)
            cur_idx = idx % n_choices
            transform.append(cur_choices[cur_idx])
            idx = idx // n_choices
        spatial_transform.append(spatial_transforms.Compose(transform))

    temporal_transform = getattr(temporal_transforms, val_aug.temporal.type)(**val_aug.temporal.get('kwargs', {}))

    clip = read_img(video_root, video_name)

    mid_time = len(clip) // 2
    mid_time = 1

    clip, aug_info = local_spatial_transform(spatial_transform, clip)
    # print(len(clip), clip[0].shape)  # 1 torch.Size([3, 5, 455, 256])

    # img_dir = os.path.join(video_root, video_name)
    center_img = os.path.join(video_root, video_name, format_str.format(mid_time))
    # print(center_img)
    # return
    bboxes = get_person(center_img, 0.5) # [ [] ] 双层列表， 每个元素是以一个长度为5的列表，分别是[x0, y0, x1, y1, score]
    print(bboxes)

    
    labels = []
    for i in range(len(bboxes)):
        label = {
            'bounding_box': bboxes[i],
            'label' :[-10],
            'person_id':[-1]
        }
        labels.append(label)
  
    input_data = {'clip': clip, 'aug_info': aug_info, 'label': labels, 'video_name': video_name, 'mid_time': mid_time}
    batch = [input_data]

    clips, aug_info = [], []
    for i in range(len(batch[0]['clip'])):
        clip, pad_ratios = batch_pad([_['clip'][i] for _ in batch])
        clips.append(clip)
        cur_aug_info = []
        for datum, pad_ratio in zip(batch, pad_ratios):
            datum['aug_info'][i]['pad_ratio'] = pad_ratio
            cur_aug_info.append(datum['aug_info'][i])
        aug_info.append(cur_aug_info)
    filenames = [_['video_name'] for _ in batch]
    labels = [_['label'] for _ in batch]
    mid_times = [_['mid_time'] for _ in batch]

    input = {
        'clips': clips,
        'aug_info': aug_info,
        'filenames': filenames,
        'labels': labels,
        'mid_times': mid_times
    }
    # print(input)
    # return
    # input.cuda()
    # model.eval()
    
    with torch.no_grad():
        ret = model(input, evaluate=True)
        num_rois = ret['num_rois']
        outputs = ret['outputs']
        targets = ret['targets']
    fnames, mid_times, bboxes = ret['filenames'], ret['mid_times'], ret['bboxes']
    # print(len(num_rois))
    print(outputs)
    for k in range(num_rois):
        prefix = "%s,%s,%.3f,%.3f,%.3f,%.3f"%(fnames[k], mid_times[k],
                                                bboxes[k][0], bboxes[k][1],
                                                bboxes[k][2], bboxes[k][3])
        for cls in range(outputs.shape[1]):
            score_str = '%.3f'%outputs[k][cls]
            out_file.write(prefix + ",%d,%s\n" % (idx_to_class[cls]['id'], score_str))
    out_file.close()

    return 

    id2classes = val_data.idx_to_class
    print(type(data), type(id2classes))
    print(id2classes)

    print(data[30])

    '''
    val_sampler = DistributedSampler(val_data, round_down=False)

    val_loader = ava.AVAmulticropDataLoader(
        val_data,
        batch_size=opt.val.batch_size,
        shuffle=False,
        num_workers=opt.val.get('workers', 1),
        pin_memory=True,
        sampler=val_sampler
    )

    val_logger = None
    if rank == 0:
        logger.info('# val data: {}'.format(len(val_data)))
        logger.info('val spatial aug: {}'.format(spatial_transform))
        logger.info('val temporal aug: {}'.format(temporal_transform))

        val_log_items = ['epoch']
        if opt.val.with_label: # False
            val_log_items.append('loss')
        if opt.val.get('eval_mAP', None) is not None:
            val_log_items.append('mAP')
        if len(val_log_items) > 1:
            val_logger = Logger(
                os.path.join(opt.result_path, 'val.log'),
                val_log_items)

    if opt.get('pretrain', None) is not None:
        load_pretrain(opt.pretrain, net)

    begin_epoch = 1
    if opt.get('resume_path', None) is not None:
        if not os.path.isfile(opt.resume_path):
            opt.resume_path = os.path.join(opt.result_path, opt.resume_path)
        checkpoint = torch.load(opt.resume_path, map_location=lambda storage, loc: storage.cuda())

        begin_epoch = checkpoint['epoch'] + 1
        net.load_state_dict(checkpoint['state_dict'])
        if rank == 0:
            logger.info('Resumed from checkpoint {}'.format(opt.resume_path))


    criterion, act_func = getattr(losses, opt.loss.type)(**opt.loss.get('kwargs', {}))

    if opt.get('evaluate', False):  # evaluation mode
        val_epoch(begin_epoch - 1, val_loader, net, criterion, act_func,
                  opt, logger, val_logger, rank, world_size, writer)

    if rank == 0:
        writer.close()
    '''


'''
def val_epoch(epoch, data_loader, model, criterion, act_func,
              opt, logger, epoch_logger, rank, world_size, writer):
    if rank == 0:
        logger.info('Evaluation at epoch {}'.format(epoch))

    model.eval()

    out_file = open(os.path.join(opt.result_path, 'tmp', 'predict_rank%d.csv'%rank), 'w')

    for i, data in enumerate(data_loader):  
        with torch.no_grad():
            ret = model(data, evaluate=True)
            num_rois = ret['num_rois']
            outputs = ret['outputs']
            targets = ret['targets']
        if num_rois == 0:
            end_time = time.time()
            continue


        fnames, mid_times, bboxes = ret['filenames'], ret['mid_times'], ret['bboxes']
        outputs = act_func(outputs).cpu().data
        idx_to_class = data_loader.dataset.idx_to_class
        for k in range(num_rois):
            prefix = "%s,%s,%.3f,%.3f,%.3f,%.3f"%(fnames[k], mid_times[k],
                                                    bboxes[k][0], bboxes[k][1],
                                                    bboxes[k][2], bboxes[k][3])
            for cls in range(outputs.shape[1]):
                score_str = '%.3f'%outputs[k][cls]
                out_file.write(prefix + ",%d,%s\n" % (idx_to_class[cls]['id'], score_str))

    out_file.close()
    dist.barrier()
    dist.barrier()
'''

if __name__ == '__main__':
   config_path = '/opt/tiger/minist/ACAR-Net/configs/AVA-Kinetics/evalAVA_SLOWFAST_R101_ACAR_HR2O_DEPTH1.yaml'
   main(config_path)
