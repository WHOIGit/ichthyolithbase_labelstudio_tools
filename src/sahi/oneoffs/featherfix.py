import json
import os
from pprint import pprint
from sahi.utils.coco import Coco, CocoAnnotation

# from core import LabelStudioPlus
#
# lsp = LabelStudioPlus.from_config('../configs/ls_config.ichthyolith_sahi.json')
# lsp.set_project('Fullslide Seg Training v1')
#
# upload_me = '../datasets/intermediary/73_U1553C_7R_2W_21-27cm__N2of2_Z200__0_0_3000_3000.jpg'
# s3key = 'temp/train2/73_U1553C_7R_2W_21-27cm__N2of2_Z200__0_0_3000_3000.jpg'
# lsp.upload_s3key(upload_me, s3key, clobber=False)
#
# surrogate_task = lsp.get_tasks(ids=[1992])[0]
# task_data = surrogate_task['data']
# task_data['slice_offset'] = dict(x=0,y=0)
# task_data['slide_id'] = '73_U1553C_7R_2W_21-27cm__N2of2_Z200__0_0_3000_3000'
# task_data['shape'] = dict(w=3000,h=3000)
# task_data['image'] = lsp.s3key_to_url(s3key)
# pprint(task_data)
#
# ...
# new_task = lsp.client.tasks.create(project=lsp.project.id, data=task_data)


coco = Coco.from_coco_dict_or_path('../datasets/annotations/fuecoco_FIXED.json')
img = [img for img in coco.images if '73_U1553C_7R_2W_21-27cm__N2of2_Z200' in img.file_name][0]
other_annots = [ann for ann in img.annotations if ann.category_name=='other']
print(len(other_annots))
other_annots.sort(key= lambda ann: ann.area)
for ann in other_annots[:-1]: # keep last one, which is the biggest which is the full object
    print(ann.area, ann.bbox)
    idx = img.annotations.index(ann)
    img.annotations.pop(idx)

# with open('../datasets/annotations/fuecoco_FIXED3.json','w') as f:
#     json.dump(coco.json, f)
