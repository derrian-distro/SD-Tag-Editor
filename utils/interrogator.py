from typing import Any

import torch
import pandas as pd
import numpy as np
import os
from utils import dbimutils
from repos import DEFAULT_REPOS
import json
from pathlib import Path

from huggingface_hub import hf_hub_download
from PIL import Image
import onnxruntime as ort


class Interrogator:
    def __init__(
        self,
        model_repo_and_name: dict[str, dict[str, str]] = DEFAULT_REPOS,
        label_file_path: str = "selected_tags.csv",
    ) -> None:
        self.model_repos_and_names = model_repo_and_name
        if not os.path.exists(label_file_path):
            self.label_file_path = "selected_tags.csv"
        self.label_file_path = label_file_path
        self.tag_groups = json.load(open("tag_groups.json"))

    # TODO: update this to support AMD
    def load_model(self, model_name: str) -> ort.InferenceSession:
        """Loads the model within the repo if the repo is in the repo list

        Args:
            model_name (str): the name of the model to load

        Raises:
            ValueError: Raises if no repo is provided

        Returns:
            InferenceSession: The onnxruntime model that is used for inference
        """
        if model_name not in self.model_repos_and_names:
            raise ValueError("Provided repo is not listed")
        print(f"loading {model_name}")
        path = Path(
            hf_hub_download(
                self.model_repos_and_names[model_name]["repo"],
                self.model_repos_and_names[model_name]["model"],
            )
        )

        return ort.InferenceSession(
            str(path), providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )

    def load_labels(self) -> tuple[Any, list[Any], list[Any], list[Any]]:
        if not os.path.exists(self.label_file_path):
            raise FileNotFoundError("Failed to find the label file.")
        df = pd.read_csv(self.label_file_path)
        tag_names = df["name"].tolist()
        rating_indexes = list(np.where(df["category"] == 9)[0])
        character_indexes = list(np.where(df["category"] == 4)[0])
        general_indexes = list(np.where(df["category"] == 0)[0])
        print("loading tags")
        return tag_names, rating_indexes, character_indexes, general_indexes

    def find_groups(self, labels: dict) -> dict:
        """
        Finds the groups that the tags belong to, returns a dict with k=group, v=list of tuple of (tag, probability)

        labels: dict of labels (k=tag, v=probability)
        """

        associated_tag_groups = {}

        # from TAG_GROUPS, create a dict of tag groups and their associated tags, each group is a key, within each key is a tuple of (tag, probability)
        for group in self.tag_groups.keys():
            for tag in self.tag_groups[group]:
                if tag in labels:
                    # append as a list of tuples
                    associated_tag_groups.setdefault(group, []).append(
                        (tag, labels[tag])
                    )
        return associated_tag_groups

    def interrogate_folder(
        self,
        image_folder: str,
        repo: str,
        general_threshold: float,
        character_threshold: float,
        subfolders: bool = False,
    ):
        print(image_folder)
        if not os.path.exists(image_folder):
            raise FileNotFoundError("Unable to find an image folder")
        model = self.load_model(model_name=repo)
        tags = self.load_labels()
        file_dict = {}
        if subfolders:
            for root, _, files in os.walk(image_folder):
                for file in files:
                    if os.path.splitext(file)[1].lower() not in [
                        ".png",
                        ".jpg",
                        ".jpeg",
                        ".webp",
                        ".gif",
                    ]:
                        continue
                    print(f"Tagging image: {file}")
                    try:
                        image = Image.open(os.path.join(root, file))
                        image_tags = self.interrogate(
                            image, model, general_threshold, character_threshold, tags
                        )
                        file_dict[file] = (
                            os.path.join(root, file),
                            image_tags[0],
                            image_tags[1],
                        )
                    except Exception:
                        continue
        else:
            for file in os.listdir(image_folder):
                if os.path.splitext(file)[1].lower() not in [
                    ".png",
                    ".jpg",
                    ".jpeg",
                    ".webp",
                    ".gif",
                ]:
                    continue
                print(f"Tagging image: {file}")
                try:
                    image = Image.open(os.path.join(image_folder, file))
                    image_tags = self.interrogate(
                        image, model, general_threshold, character_threshold, tags
                    )
                    file_dict[file] = image_tags
                except Exception:
                    continue
        return file_dict

    def interrogate(
        self,
        image: Image.Image,
        model: ort.InferenceSession,
        general_threshold: float,
        character_threshold: float,
        tags: tuple[Any, list[Any], list[Any], list[Any]],
    ):
        tag_names, rating_indexes, character_indexes, general_indexes = tags
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

        labels = list(
            zip(tag_names, probabilities[0].astype(float))
        )  # a list of tuples of the tag name and the confidence
        rating_names = [labels[i] for i in rating_indexes]
        rating_res = list(rating_names)

        general_names = [labels[i] for i in general_indexes]
        general_res = [x for x in general_names if x[1] > general_threshold]

        character_names = [labels[i] for i in character_indexes]
        character_res = [x for x in character_names if x[1] > character_threshold]
        full_tags = dict(general_res + character_res)
        full_tags = self.find_groups(full_tags)
        return rating_res, full_tags


# TODO: decide if used?
def tag(probs: list, tag_names: list[str]):
    """
    Creates a dict of tags and their probabilities.

    probs: list of probabilities
    tag_names: list of tag names
    """
    return dict(zip(tag_names, probs[0].astype(float)))
