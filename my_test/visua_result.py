import os 
import cv2
import numpy as np


from my_test.test_utils import id_class_dict

def visual_result(image_path, bboxes = None, outputs = None, save = True, prob_thresh = 0.35):
    '''
    
    bbox: list of list
    output: 2d tensor

    '''
    assert os.path.exists(image_path), 'iamge_path {} not exitst !!!'
    img = cv2.imread(image_path)
    h, w, c = img.shape


    save_root = 'visual_results/'
    image_name = image_path.split('/')[-1]
    save_path = os.path.join(save_root, image_name)
    assert len(bboxes) == outputs.shape[0]
    for b in range(len(bboxes)) :
        prob = np.array(outputs[b])
        bbox = np.array(bboxes[b])
        sort_idx = np.argsort(prob)
        top3_idx = sort_idx[-3:][::-1]
        top3_prob = prob[top3_idx]
        top3_class = [ id_class_dict[idx]['name'] for idx in top3_idx]
        # print(top3_idx, top3_prob, top3_class)
        # print(h, w)
        bbox = [int(bbox[0] * w), int(bbox[1] * h), int(bbox[2] * w), int(bbox[3] * h)]
        # print(bbox)
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255))

        for idx in range(len(top3_idx)):
            print('{} : {:.4f}'.format(top3_class[idx], top3_prob[idx]) )
            if top3_prob[idx] > prob_thresh :
                cv2.putText(img, '{} : {:.2f}'.format(top3_class[idx], top3_prob[idx]), (bbox[0],bbox[1] - 20 + idx * 12 ), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
        # print(sorted_prob)
    if save:
        cv2.imwrite(save_path, img)


if __name__ == "__main__":
    img_path = '/opt/tiger/minist/datasets/test/v0d00fg10000c3rq96bc77ufsrjbdn4g/v0d00fg10000c3rq96bc77ufsrjbdn4g_2.jpg'
    visual_result(img_path)