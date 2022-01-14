import json
import os

import pdb


class GTLoader:
  def __init__(self, label_path, cls_filter=None):
    self._cls_filter = cls_filter
    self._label_dict = {}
    with open(label_path, 'r') as f:
      label_list = json.load(f)
      for temp_label in label_list:
        temp_id = os.path.splitext(temp_label['name'])[0]
        self._label_dict[temp_id] = temp_label
      assert len(self._label_dict.keys()) == len(label_list)

  def load_gt(self, img_name):
    gt_list = self._get_by_name(img_name)
    gt_dict = self._fetch_info(gt_list)
    return gt_dict
  
  def _get_by_name(self, img_name):
    if os.path.splitext(img_name) == 2 and os.path.splitext(img_name)[1] == '.jpg':
      img_name = os.path.splitext(img_name)[0]
    return self._label_dict[img_name]['labels']

  def _fetch_info(self, gt_list):
    '''convert pytorch detection model output to list of dictionary of list
        {
          'scores': [0.97943294], 
          'classes': [14], 
          'boxes': [[x1, y1, x2, y2]]
        }
    '''
    ret_dict = {
      'classes' : [],
      'boxes' : [],
      'scores' : []
    }
    for temp_dict in gt_list:
      if self._cls_filter != None and \
          temp_dict['category'] not in self._cls_filter:
        continue
      ret_dict['scores'].append(1.)
      ret_dict['classes'].append(temp_dict['category'])
      ret_dict['boxes'].append([
          temp_dict['box2d']['y1'],
          temp_dict['box2d']['x1'],
          temp_dict['box2d']['y2'],
          temp_dict['box2d']['x2']])
    return ret_dict
    

if __name__ == "__main__":
  label_path = "/data_volumn_2/bdd100k/labels/bdd100k_labels_images_val.json"
  gt_loader = GTLoader(label_path)