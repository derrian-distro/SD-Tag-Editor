import torch  # ?
import pandas as pd
import numpy as np
import os
import onnxruntime as ort
import json


from huggingface_hub.file_download import hf_hub_download
from PIL import Image
from typing import Any, Union
from utils import dbimutils
from repos import DEFAULT_REPOS
from pathlib import Path


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

    def find_groups(self, tags: dict) -> dict:
        """
        Finds the groups that the tags belong to, returns a dict with k=group, v=list of tuple of (tag, probability)

        tags: dict of tags (k=tag, v=probability)
        """
        associated_tag_groups = {}

        def depth_search(tag: str, tag_group: Union[dict, list]):
            if not isinstance(tag_group, dict):
                return tag in tag_group

            for group in tag_group.keys():
                if tag_found := depth_search(tag, tag_group[group]):
                    return (
                        {group: tag}
                        if isinstance(tag_found, bool)
                        else {group: tag_found}
                    )
            return False

        for tag, value in tags.items():
            tag_group = depth_search(tag, self.tag_groups)
            # print(tag_group)
            if isinstance(tag_group, bool):
                if "other" not in associated_tag_groups:
                    associated_tag_groups["other"] = [tag]
                continue

            group_type: Union[dict, str] = tag_group

            current_group = associated_tag_groups

            while isinstance(group_type, dict):
                # group_type.keys() will only ever have one key
                group = list(group_type.keys())[0]
                if group not in current_group:
                    current_group[group] = (
                        {} if isinstance(group_type[group], dict) else []
                    )
                current_group = current_group[group]
                group_type = group_type[group]
            # append as dict, truncate to 6 decimals on tags[tag]
            current_group.append({group_type: round(value, 6)})

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
        image = np.asarray(image)  # type: ignore

        # PIL RGB to OpenCV BGR
        image = image[:, :, ::-1]  # type: ignore

        image = dbimutils.make_square(image, height)
        image = dbimutils.smart_resize(image, height)
        image = image.astype(np.float32)  # type: ignore
        image = np.expand_dims(image, 0)  # type: ignore

        input_name = model.get_inputs()[0].name
        label_name = model.get_outputs()[0].name

        probabilities = model.run([label_name], {input_name: image})[0]

        labels = list(zip(tag_names, probabilities[0].astype(float)))
        for i in range(len(labels)):
            labels[i] = (labels[i][0].replace("_", " "), labels[i][1])

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
