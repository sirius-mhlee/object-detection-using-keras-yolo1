import numpy as np

import Configuration as cfg

class BBox:
    def __init__(self, left, top, right, bottom, prob):
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom
        self.prob = prob

    def get_class(self):
        return np.argmax(self.prob)

def nms_bbox(bbox_list, nms_threshold, obj_threshold):
    for class_idx in range(cfg.object_class_num):
        sorted_idx = list(reversed(np.argsort([bbox.prob[class_idx] for bbox in bbox_list])))

        for i in range(len(sorted_idx)):
            bbox_i = sorted_idx[i]
            
            if bbox_list[bbox_i].prob[class_idx] == 0: 
                continue
            else:
                for j in range(i + 1, len(sorted_idx)):
                    bbox_j = sorted_idx[j]
                    
                    inter_left = max(bbox_list[bbox_i].left, bbox_list[bbox_j].left)
                    inter_top = max(bbox_list[bbox_i].top, bbox_list[bbox_j].top)
                    inter_right = min(bbox_list[bbox_i].right, bbox_list[bbox_j].right)
                    inter_bottom = min(bbox_list[bbox_i].bottom, bbox_list[bbox_j].bottom)
                    inter_width = max(0.0, inter_right - inter_left + 1)
                    inter_height = max(0.0, inter_bottom - inter_top + 1)
                    inter_area = inter_width * inter_height

                    area_bbox_i = (bbox_list[bbox_i].right - bbox_list[bbox_i].left) * (bbox_list[bbox_i].bottom - bbox_list[bbox_i].top)
                    area_bbox_j = (bbox_list[bbox_j].right - bbox_list[bbox_j].left) * (bbox_list[bbox_j].bottom - bbox_list[bbox_j].top)

                    inter_ratio = inter_area / (area_bbox_i + area_bbox_j - inter_area)

                    if inter_ratio >= nms_threshold:
                        bbox_list[bbox_j].prob[class_idx] = 0
                        
    nms_bbox_list = [bbox for bbox in bbox_list if bbox.prob[bbox.get_class()] >= obj_threshold]
    return nms_bbox_list