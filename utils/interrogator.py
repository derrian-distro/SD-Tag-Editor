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
        model: str = "swin",
    ) -> None:
        self.model_repos_and_names = model_repo_and_name
        if not Path(label_file_path).exists():
            label_file_path = "selected_tags.csv"
        self.label_file_path = Path(label_file_path)
        self.tag_groups = json.load(open("tag_groups.json"))
        self.model = self._load_model(model)
        self.labels = self._load_labels()

    # TODO: update this to support AMD
    def _load_model(self, model_name: str) -> ort.InferenceSession:
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

    def _load_labels(self) -> tuple[Any, list[Any], list[Any], list[Any]]:
        if not self.label_file_path.exists():
            raise FileNotFoundError("Failed to find the label file.")
        df = pd.read_csv(str(self.label_file_path))
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
                return tag.lower() in tag_group

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
            if isinstance(tag_group, bool):
                if "other" not in associated_tag_groups:
                    associated_tag_groups["other"] = [{tag: 0}]
                else:
                    associated_tag_groups["other"].append({tag: 0})
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
        general_threshold: float = 0.35,
        character_threshold: float = 0.75,
        subfolders: bool = False,
    ):
        image_folder: Path = Path(image_folder)
        if not image_folder.exists():
            raise FileNotFoundError("Unable to find an image folder")
        file_dict = {}
        if subfolders:
            files = list(self.walk(image_folder))
            for file in files:
                if tags := self.tag_image(file, general_threshold, character_threshold):
                    file_dict[file] = tags
        else:
            for file in image_folder.iterdir():
                if file.is_dir():
                    continue
                if tags := self.tag_image(
                    file,
                    general_threshold,
                    character_threshold,
                ):
                    file_dict[os.path.join(image_folder, file)] = tags
        return file_dict

    def tag_image(
        self,
        file_path: Path,
        general_threshold: float = 0.35,
        character_threshold: float = 0.75,
    ) -> Union[dict[str, Any], None]:
        if file_path.suffix.lower() not in [
            ".png",
            ".jpg",
            ".jpeg",
            ".webp",
            ".gif",
        ]:
            return None
        print(f"tagging image: {file_path.name}")
        try:
            image = Image.open(file_path)
            rating_res, general_res, character_res = self.interrogate(
                image, self.model, general_threshold, character_threshold, self.labels
            )
            return {
                "file_path": str(file_path),
                "rating": rating_res,
                "general": dict(general_res),
                "character": dict(character_res),
            }
        except Exception:
            return None

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
        # full_tags = dict(general_res + character_res)
        # full_tags = self.find_groups(full_tags)

        return rating_res, general_res, character_res

    def walk(self, path: Path):
        for p in path.iterdir():
            if p.is_dir():
                yield from self.walk(p)
                continue
            yield p.resolve()


# if os.path.splitext(str(file_path))[1].lower() not in [
#     ".png",
#     ".jpg",
#     ".jpeg",
#     ".webp",
#     ".gif",
# ]:
