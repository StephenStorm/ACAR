from mmdet.apis import init_detector, inference_detector
import mmcv
import numpy as np
import torch
import cv2






# Specify the path to model config and checkpoint file
config_file = '/opt/tiger/minist/mmdetection/configs/yolo/yolov3_d53_mstrain-608_273e_coco.py'
checkpoint_file = '/opt/tiger/minist/mmdetection/model_zoo/yolov3_d53_mstrain-608_273e_coco_20210518_115020-a2c3acb8.pth'



model = init_detector(config_file, checkpoint_file, device='cuda:0')




def get_person(img, score_thr = 0.9):
    '''
    img: img path
    result:result = inference_detector(model, img)
    '''
      # or img = mmcv.imread(img), which will only load it once
    img_cv = cv2.imread(img)
    h, w, c = img_cv.shape
    # print(h, w)
    result = inference_detector(model, img)
    # model.show_result(img, result, out_file='result.jpg')


    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(result)
    ]
    labels = np.concatenate(labels)

    # print(len(result))
    bboxes = np.vstack(result)
    assert bboxes.shape[1] == 5

    scores = bboxes[:, -1]
    inds = scores > score_thr
    bboxes = bboxes[inds, :]
    labels = labels[inds]

    inds = labels == 0
    bboxes = bboxes[inds, :]
    # print(bboxes)
    bboxes[:, 0:4:2] = bboxes[:, 0:4:2] / w
    bboxes[:, 1:4:2] = bboxes[:, 1:4:2] / h
    # print(bboxes)
    return bboxes[:,0:4]

if __name__ == '__main__' :
    img = '/opt/tiger/minist/datasets/test/v0d00fg10000c3rq96bc77ufsrjbdn4g/v0d00fg10000c3rq96bc77ufsrjbdn4g_0.jpg'
    bbox = get_person(img)
    print(bbox)





'''

img = mmcv.imread(img)
img = img.copy()
if isinstance(result, tuple):
    bbox_result, segm_result = result
    if isinstance(segm_result, tuple):
        segm_result = segm_result[0]  # ms rcnn
else:
    bbox_result, segm_result = result, None
bboxes = np.vstack(bbox_result)
labels = [
    np.full(bbox.shape[0], i, dtype=np.int32)
    for i, bbox in enumerate(bbox_result)
]
labels = np.concatenate(labels)
# draw segmentation masks
segms = None
if segm_result is not None and len(labels) > 0:  # non empty
    segms = mmcv.concat_list(segm_result)
    if isinstance(segms[0], torch.Tensor):
        segms = torch.stack(segms, dim=0).detach().cpu().numpy()
    else:
        segms = np.stack(segms, axis=0)
# if out_file specified, do not show image in window
if out_file is not None:
    show = False

'''










# visualize the results in a new window
# print(result)


# model.show_result(img, result)

# or save the visualization results to image files
# model.show_result(img, result, out_file='result.jpg')

# test a video and show the results
# video = mmcv.VideoReader('video.mp4')
# for frame in video:
#     result = inference_detector(model, frame)
#     model.show_result(frame, result, wait_time=1)