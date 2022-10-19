import fiftyone as fo
import clip
import torch
import json
import numpy as np
from glob import glob
import os



# Fix hard code in rsplit

class Retriever:
    def __init__(self, img_dir='data/demo/KeyFramesC00_V00'):

        self.img_dir = img_dir
        self.dataset = fo.Dataset.from_images_dir(
            img_dir, name=None, tags=None, recursive=True)

        print(img_dir)

    def add_meta_data_images(self): # Add video, frameid
        for sample in self.dataset:
            _, sample['video'], sample['frameid'] = sample['filepath'][:-
                                                               4].rsplit('/', 2)
            sample.save()

    def add_object_detection(self, object_dir='data/demo/ObjectsC00_V00'):

        self.object_dir = object_dir

        for sample in self.dataset:
            object_path = os.path.join(object_dir, sample['filepath'][-20:-4] + '.json')
            with open(object_path) as jsonfile:
                det_data = json.load(jsonfile)
            detections = []
            for cls, box, score in zip(det_data['detection_class_entities'], det_data['detection_boxes'], det_data['detection_scores']):
                # Convert to [top-left-x, top-left-y, width, height]
                boxf = [float(box[1]), float(box[0]), float(box[3]) -
                        float(box[1]), float(box[2]) - float(box[0])]
                scoref = float(score)

                # Only add objects with confidence > 0.4
                if scoref > 0.4:
                    detections.append(
                        fo.Detection(
                            label=cls,
                            bounding_box=boxf,
                            confidence=float(score)
                        )
                    )
            sample["object_faster_rcnn"] = fo.Detections(detections=detections)
            sample.save()


    def get_all_list_keyframe(self):

        # path_all_video = os.path.join(self.img_dir, '*')
        path_all_keyframe = os.path.join(self.img_dir,'*','*.jpg')

        all_keyframe = glob(path_all_keyframe)
        video_keyframe_dict = {}

        # all_video = glob(path_all_video)
        # all_video = [v.rsplit('/', 1)[-1] for v in all_video]

        for kf in all_keyframe:
            _, vid, kf = kf[:-4].rsplit('/', 2)
            if vid not in video_keyframe_dict.keys():
                video_keyframe_dict[vid] = [kf]
            else:
                video_keyframe_dict[vid].append(kf)

        for k, v in video_keyframe_dict.items():
            video_keyframe_dict[k] = sorted(v)

        return video_keyframe_dict


    def get_all_list_video(self):

        path_all_video = os.path.join(self.img_dir, '*')
        
        all_video = glob(path_all_video)
        all_video = [v.rsplit('/', 1)[-1] for v in all_video]

        return all_video

    def extract_vector_features_per_frame(self, features_dir='/data/demo/CLIPFeatures_C00_V00'):

        self.features_dir = features_dir

        all_video = self.get_all_list_video()
        all_keyframes = self.get_all_list_keyframe()

        for v in all_video:
            # clip_path = '/mnt/g/CLIPFeatures_C00_V00/{}.npy'.format(v)
            clip_path = os.path.join(features_dir,  v + '.npy')

            features = np.load(clip_path)

            feature_video_dir = os.path.join(features_dir, v)

            if not os.path.isdir(feature_video_dir):
                os.makedirs(feature_video_dir)

            for i, frameid in enumerate(all_keyframes[v]):

                feature_frame_dir = os.path.join(feature_video_dir, frameid + '.npy')
                np.save(feature_frame_dir, features[i])

    def add_clip_embedding(self):
        for sample in self.dataset:
            object_path = os.path.join(
                self.features_dir, sample['filepath'][-20:-4] + '.npy')
            
            clip_embedding = np.load(object_path)
            sample['clip_embedding'] = clip_embedding

            sample.save()


    def add_text_query_similarity(self, text_features):
        for sample in self.dataset:
            a = sample['clip_embedding']
            query_similarity = (text_features @ a.reshape(1,
                                512).T).cpu().numpy().item()
            sample['text_query_similarity'] = query_similarity
            
            # print(sample)


            sample.save()


        self.dataset = self.dataset.sort_by("text_query_similarity", reverse=True)


    def export(self, top_k, export_dir):

        result = self.dataset[:top_k]

        result.export(export_dir=export_dir,
                       dataset_type=fo.types.FiftyOneDataset)
        

    


