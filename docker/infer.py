from mmdet.apis import init_detector, inference_detector
from mmdet.utils import register_all_modules
import mmcv
# from pycocotools import mask
import json
import numpy as np
import os
import glob
from skimage import measure
from shapely.geometry import Polygon


def inference():

    config_file = 'comb/mask-rcnn_convnext-v2-b_fpn_lsj-3x-fcmae_coco.py'
    checkpoint_file = 'comb/best_coco_segm_mAP_50_epoch_15.pth'

    register_all_modules()
    # build the model from a config file and a checkpoint file
    model = init_detector(config_file, checkpoint_file, device='cuda:0')  # or device='cpu'

    with open("empt_ann.json") as file:
        gt = json.load(file)

    empty_submit = dict()
    empty_submit["info"] = {
        "contributor": "",
        "date_created": "2023-09-04",
        "description": "",
        "url": "",
        "version": "",
        "year": 0
        }
    empty_submit["licenses"] = [
        {
            "name": "",
            "id": 0,
            "url": ""
        }
        ]
    empty_submit["images"] = []
    empty_submit["categories"] = gt["categories"]
    empty_submit["annotations"] = []

    count_anns = 1
    images = glob.glob("/opt/app/saved_images/" + "*.png")
    for image in images:
        img = mmcv.imread( image, channel_order='rgb')
        image_id = int(os.path.splitext(os.path.basename(image))[0])
        empty_submit["images"].append({'id': image_id, 'width': 512, 'height': 512, 'license':0,'date_captured':str(0), 'file_name': str(image_id)+'.png'})
        result = inference_detector(model, img)
        print(f" Inferred {image_id} ")
        masks = result.pred_instances.masks.cpu().numpy()
        scores = result.pred_instances.scores.cpu().numpy()
        labels = result.pred_instances.labels.cpu().numpy()
        boxes = result.pred_instances.bboxes.cpu().numpy() 
        indexes = np.flip(np.argsort(scores))
        count = 0   
        while count<3 and indexes.size != 0: #or count < len(indexes)
            
            index = indexes[count]
            score = scores[index]
            count += 1
            if score>0.8:
                mask = masks[index]
                label= labels[index]
                box = boxes[index]
                contours = measure.find_contours(mask)
             
                for contour in contours:             
                        for i in range(len(contour)):
                            row, col = contour[i]
                            contour[i] = (col - 1, row - 1)

                        # Simplify polygon
                        poly = Polygon(contour)
                        poly = poly.simplify(1.0, preserve_topology=False)
                
                        if(poly.is_empty):
                            continue
                        segmentation = np.array(poly.exterior.coords).ravel().tolist()
                        new_ann = dict()
                        new_ann["id"] = count_anns
                        new_ann["image_id"] = image_id
                        new_ann["category_id"] = (label+26).tolist()
                        new_ann["segmentation"] = [segmentation]
                        area = poly.area
                        new_ann["area"] = area
                        x, y = contour.min(axis=0)
                        w, h = contour.max(axis=0) - contour.min(axis=0)
                        new_ann["bbox"]  = [int(x), int(y), int(w), int(h)]
                        new_ann["iscrowd"] = 0
                        new_ann["attributes"] = {
                            "occluded": False
                        }
                        
                        # if int(area)>90:
                        count_anns += 1
                        empty_submit["annotations"].append(new_ann.copy())


    return empty_submit