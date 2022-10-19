import fiftyone as fo
import clip
import torch
import json
import numpy as np
from glob import glob
import os

from src.retriever import *


IMG_DIR = 'data/demo/KeyFramesC00_V00'
OBJECT_DIR = 'data/demo/ObjectsC00_V00'
FEATURE_DIR = 'data/demo/CLIPFeatures_C00_V00'


retriever = Retriever(IMG_DIR)

retriever.add_meta_data_images()

retriever.add_object_detection(OBJECT_DIR)

retriever.extract_vector_features_per_frame(FEATURE_DIR)

retriever.add_clip_embedding()




# # LOAD DATASET
# # argument: path dataset, top k images
# dataset = fo.Dataset.from_images_dir(
#     'data_demo/KeyFramesC00_V00', name=None, tags=None, recursive=True)


# # LAUNCH SESSION
# session = fo.launch_app(dataset, auto=False)


# # ADD META(video, framid)
# # can omptimize without loop?
# for sample in dataset:
#     _, sample['video'], sample['frameid'] = sample['filepath'][:-
#                                                                4].rsplit('/', 2)
#     sample.save()


# # ADD META(OBJECT)
# for sample in dataset:
#     object_path = 'data/ObjectsC00_V00/{}.json'.format(
#         sample['filepath'][-20:-4])
#     with open(object_path) as jsonfile:
#         det_data = json.load(jsonfile)
#     detections = []
#     for cls, box, score in zip(det_data['detection_class_entities'], det_data['detection_boxes'], det_data['detection_scores']):
#         # Convert to [top-left-x, top-left-y, width, height]
#         boxf = [float(box[1]), float(box[0]), float(box[3]) -
#                 float(box[1]), float(box[2]) - float(box[0])]
#         scoref = float(score)

#         # Only add objects with confidence > 0.4
#         if scoref > 0.4:
#             detections.append(
#                 fo.Detection(
#                     label=cls,
#                     bounding_box=boxf,
#                     confidence=float(score)
#                 )
#             )
#     sample["object_faster_rcnn"] = fo.Detections(detections=detections)
#     sample.save()


# # ADD CLIP EMBEDDING
# all_keyframe = glob('data/KeyFramesC00_V00/*/*.jpg')
# video_keyframe_dict = {}
# all_video = glob('data/KeyFramesC00_V00/*')
# all_video = [v.rsplit('/', 1)[-1] for v in all_video]

# for kf in all_keyframe:
#     _, vid, kf = kf[:-4].rsplit('/', 2)
#     if vid not in video_keyframe_dict.keys():
#         video_keyframe_dict[vid] = [kf]
#     else:
#         video_keyframe_dict[vid].append(kf)


# for k, v in video_keyframe_dict.items():
#     video_keyframe_dict[k] = sorted(v)

# for v in all_video:
#     print(v)
#     clip_path = 'data/CLIPFeatures_C00_V00/{}.npy'.format(v)
#     a = np.load(clip_path)
#     os.makedirs('data/CLIPFeatures_C00_V00/{}'.format(v))
#     for i, k in enumerate(video_keyframe_dict[v]):
#         np.save('data/CLIPFeatures_C00_V00/{}/{}.npy'.format(v, k), a[i])



# for sample in dataset:
#     object_path = 'data/CLIPFeatures_C00_V00/{}.npy'.format(
#         sample['filepath'][-20:-4])
#     clip_embedding = np.load(object_path)
#     sample['clip_embedding'] = clip_embedding

#     sample.save()





