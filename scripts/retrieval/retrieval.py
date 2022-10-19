import pprint
import argparse
import yaml

import fiftyone as fo
import clip
import torch
import json
import numpy as np
from glob import glob
import os

from src.retriever import *
from src.models.ViT import VIT


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--img_dir", default="data/demo/KeyFramesC00_V00", help="image directory")

    parser.add_argument(
        "--object_dir", default="data/demo/ObjectsC00_V00", help="object directory")

    parser.add_argument(
        "--feature_dir", default="data/demo/CLIPFeatures_C00_V00", help="vector features directory")

    parser.add_argument(
        "--text_query", default="A doctor is speaking", help="text query")

    parser.add_argument(
        "--top_k", default=100, help="top k highest score images")

    parser.add_argument(
        "--export", default="data/export", help="directory for exporting top k images")
    
    return parser.parse_args()


def main(args):

    # load data

    retriever = Retriever(args.img_dir)

    retriever.add_meta_data_images()

    retriever.add_object_detection(args.object_dir)

    retriever.extract_vector_features_per_frame(args.feature_dir)

    retriever.add_clip_embedding()

    # model

    encoder = VIT()

    # query

    text_search = args.text_query
    text_search_features = encoder.encode_text(text_search)

    retriever.add_text_query_similarity(text_search_features)

    retriever.export(args.top_k)


    # data = fo.Dataset.from_dir(
    #     dataset_dir = 'data/export',
    #     dataset_type=fo.types.FiftyOneDataset
    # )


if __name__ == "__main__":
    args = get_parser()
    main(args)



