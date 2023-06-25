import pandas as pd
import numpy as np
import cv2
import os
import dbimutils
import json

from huggingface_hub import hf_hub_download
from PIL import Image
from onnxruntime import InferenceSession
from pathlib import Path
from typing import Union, Tuple, Dict


class Interrogator():
    def __init__(self, model_repo_and_name: list[dict[str, str]], label_file_path: str = "selected_tags.csv") -> None:
        self.model_repos_and_names = model_repo_and_name
        if not os.path.exists(label_file_path):
            self.label_file_path = "selected_tags.csv"
        self.label_file_path = label_file_path
        self.tag_groups = json.load(open("tag_groups.json"))
    
    # TODO: update this to support CPU and AMD
    def load_model(self, repo: str) -> InferenceSession:
        if repo not in self.model_repos_and_names:
            raise ValueError("Provided repo is not listed")
        return InferenceSession(
            hf_hub_download(
                repo, self.model_repos_and_names[repo]
                )
            )
    
    def load_labels(self) -> tuple(list[str]):
        if not os.path.exists(self.label_file_path):
            raise FileNotFoundError("Failed to find the label file.")
        df = pd.read_csv(self.label_file_path)
        tag_names = df['name'].tolist()
        rating_indexes = list(np.where(df['category'] == 9)[0])
        character_indexes = list(np.where(df["category"] == 4)[0])
        general_indexes = list(np.where(df["category"] == 0)[0])
        return tag_names, rating_indexes, character_indexes, general_indexes
    
    def find_groups(self, labels: dict) -> dict:
        '''
        Finds the groups that the tags belong to, returns a dict with k=group, v=list of tuple of (tag, probability)

        labels: dict of labels (k=tag, v=probability)
        '''
        
        associated_tag_groups = {}

        #from TAG_GROUPS, create a dict of tag groups and their associated tags, each group is a key, within each key is a tuple of (tag, probability)
        for group in self.tag_groups.keys():
            for tag in self.tag_groups[group]:
                if tag in labels.keys():
                    #append as a list of tuples
                    associated_tag_groups.setdefault(group, []).append((tag, labels[tag]))
        return associated_tag_groups
    
    def interrogate_folder(self, image_folder: str, repo: str, general_threshold: float, character_threshold: float):
        if not os.path.exists(image_folder):
            raise FileNotFoundError("Unable to find an image folder")
        model = self.load_model(repo=repo)
        tags = self.load_labels()
        file_dict = {}
        for file in image_folder:
            if os.path.splitext()[1].lower() not in ['png', 'jpg', 'jpeg', 'webp']:
                continue
            image = Image.open(file)
            image_tags = self.interrogate(image, model, general_threshold, character_threshold, tags)
            file_dict[file] = image_tags
            
    def interrogate(self, image: Image.Image, model: InferenceSession, general_threshold: float,
                    character_threshold: float, tags: tuple[list[str]]):
        tag_names, _, character_indexes, general_indexes = tags
        _, height, _, _ = model.get_inputs()[0].shape
        
        image = image.convert("RGBA")
        new_image = Image.new("RGBA", image.size, "WHITE")
        new_image.paste(image, mask=image)
        image = new_image.convert("RGB")
        image = np.asarray(image)
        
        # PIL RGB to OpenCV BGR
        image = image[:, :, ::-1]
        
        image = dbimutils.make_square(image, height)
        image = dbimutils.smart_resize(image, height)
        image = image.astype(np.float32)
        image = np.expand_dims(image, 0)
        
        input_name = model.get_inputs()[0].name
        label_name = model.get_outputs()[0].name
        
        probabilities = model.run([label_name], {input_name: image})[0]
        
        labels = list(zip(tag_names, probabilities[0].astype(float))) # a list of tuples of the tag name and the confidence
        general_names = [labels[i] for i in general_indexes]
        general_res = [x for x in general_names if x[1] > general_threshold]
        
        character_names = [labels[i] for i in character_indexes]
        character_res = [x for x in character_names if x[1] > character_threshold]
        full_tags = dict(general_res + character_res)
        full_tags = self.find_groups(full_tags)
        return full_tags


def tag(probs: list, tag_names: list[str]):
    '''
    Creates a dict of tags and their probabilities.

    probs: list of probabilities
    tag_names: list of tag names
    '''
    labels = dict(zip(tag_names, probs[0].astype(float)))
    return labels

#^remove? already handled line78

# def find_groups(labels: dict) -> dict:
#     '''
#     Finds the groups that the tags belong to, returns a dict with k=group, v=list of tuple of (tag, probability)
    
#     labels: dict of labels (k=tag, v=probability)
#     '''
    
#     associated_tag_groups = {}

#     #from TAG_GROUPS, create a dict of tag groups and their associated tags, each group is a key, within each key is a tuple of (tag, probability)
#     for group in TAG_GROUPS.keys():
#         for tag in TAG_GROUPS[group]:
#             if tag in labels.keys():
#                 #append as a list of tuples
#                 associated_tag_groups.setdefault(group, []).append((tag, labels[tag]))

#     return associated_tag_groups

