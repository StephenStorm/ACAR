import multiprocessing as mp

from torch.serialization import check_module_version_greater_or_equal
mp.set_start_method('spawn', force=True)
import os

from easydict import EasyDict
import yaml
import torch
from datasets import ava, spatial_transforms, temporal_transforms
from distributed_utils import init_distributed
import losses
from models import AVA_model

from utils import *
from PIL import Image
from datasets.ava import get_aug_info, batch_pad

from collections import OrderedDict



from my_test.mmdet_test import get_person
from my_test.test_utils import id2class, class2id, id_class_dict
from my_test.visua_result import visual_result

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


# def read_img(root, video_name, index, annotation_path = 'annotations/ava_val_v2.2_fair_0.85.pkl'):
def read_img(root, video_name, temporal_transform, spatial_transform = None):
    image_dir = os.path.join(root, video_name)
    
    image_paths = os.listdir(image_dir)
    
    start_frame = 0
    n_frames = len(image_paths)
    frame_indices = list(range(start_frame, start_frame + n_frames))

    frame_indices = temporal_transform(frame_indices)

    frame_format = video_name + '_{}.jpg'
    clip = []
    for i in range(len(frame_indices) ):
        image_path = os.path.join(image_dir, frame_format.format(frame_indices[i]))
        try:
            image_path_t = os.path.join(image_dir, image_path)
            with Image.open(image_path_t) as img:
                img = img.convert('RGB')
        except BaseException as e:
            raise RuntimeError('Caught "{}" when loading {}'.format(str(e), image_path))   
        clip.append(img)

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




def main(config):
    video_root = '/opt/tiger/minist/datasets/'
    video_name = 'v0d00fg10000c3rq96bc77ufsrjbdn4g'

    format_str = video_name + '_{}.jpg'
    out_file = open('result.txt', 'w')

    with open(config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    opt = EasyDict(config)

    # writer = SummaryWriter(os.path.join(opt.result_path, 'tb'))
    
    # create model
    model = AVA_model(opt.model)
    
    model_param_tmp = open('model_param.txt', 'w')
    for param, weight in model.state_dict().items():
        model_param_tmp.write('{} {}'.format(param, weight.shape))
    
    model_param_tmp.write('-'*100 + '\n')
    # model_param_tmp.wri
        
    
    model.cuda()
    model.eval()
    criterion, act_func = getattr(losses, opt.loss.type)(**opt.loss.get('kwargs', {}))
    
    if opt.get('resume_path', None) is not None:
        print('loading state_dict form checkpoint...')
        if not os.path.isfile(opt.resume_path):
            opt.resume_path = os.path.join(opt.result_path, opt.resume_path)
        print('resume path : {}'.format(opt.resume_path))
        checkpoint = torch.load(opt.resume_path, map_location=lambda storage, loc: storage.cuda())
        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            new_state_dict[k[7:]] = v

        model.load_state_dict(new_state_dict)

        '''
        for param in new_state_dict:
            # print(param, checkpoint[param].shape)
            model_param_tmp.write(param + '\n')
            # model_param_tmp.write(str(checkpoint[param].shape) + '\n')
        model_param_tmp.close()
        '''
        # begin_epoch = checkpoint['epoch'] + 1
        # if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        #     model.load_state_dict(checkpoint['state_dict'])
        # else:
        #     model.load_state_dict(checkpoint)
        # model.load_state_dict(checkpoint)
        print('load checkpoint done !!!')
    
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

    clip, aug_info = read_img(video_root, video_name, temporal_transform, spatial_transform)
    print('aug info : {}'.format(aug_info))
    print('clip shape : {}'.format(clip[0].shape))
    mid_time = len(os.listdir(os.path.join(video_root, video_name))) // 2

    center_img = os.path.join(video_root, video_name, format_str.format(mid_time))
    print('center img path {} '.format(center_img))

    bboxes = get_person(center_img, 0.5) # [ [] ] 双层列表， 每个元素是以一个长度为4的列表，分别是[x0, y0, x1, y1]
    if len(bboxes) == 0:
        return
    print(bboxes)

    
    labels = []
    for i in range(len(bboxes)):
        label = {
            'bounding_box': bboxes[i],
            'label' :[0],
            'person_id':[-1]
        }
        labels.append(label)
  
    input_data = {'clip': clip, 'aug_info': aug_info, 'label': labels, 'video_name': video_name, 'mid_time': str(mid_time)}
    batch = [input_data]

    clips, aug_info = [], []
    for i in range(len(batch[0]['clip'])): # for i in range(1)
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
    print('batch [0] shape {}'.format(clips[0].shape))
    # print('filenames : {}'.format(filenames))
    # print('mid_times: {}'.format(mid_times))

    
    with torch.no_grad():
        ret = model(input, evaluate=True)
        num_rois = ret['num_rois']
        outputs = ret['outputs']
        targets = ret['targets']
    fnames, mid_times, bboxes = ret['filenames'], ret['mid_times'], ret['bboxes']
    # print('bboxes: {}'.format(bboxes))
    outputs = act_func(outputs).cpu().data
    # print('output shape : {}'.format(outputs.shape))
    # print(outputs)
    visual_result(center_img, bboxes, outputs)

    for k in range(num_rois):
        prefix = "%s,%s,%.3f,%.3f,%.3f,%.3f"%(fnames[k], mid_times[k],
                                                bboxes[k][0], bboxes[k][1],
                                                bboxes[k][2], bboxes[k][3])
        for cls in range(outputs.shape[1]):
            score_str = '%.3f'%outputs[k][cls]
            out_file.write(prefix + ",%d,%s\n" % (id_class_dict[cls]['id'], score_str))
    out_file.close()

    return 

if __name__ == '__main__':
   config_path = '/opt/tiger/minist/ACAR-Net/configs/AVA-Kinetics/evalAVA_SLOWFAST_R101_ACAR_HR2O_DEPTH1.yaml'
   main(config_path)
