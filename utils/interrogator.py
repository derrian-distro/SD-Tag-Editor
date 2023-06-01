import pandas as pd
import numpy as np
import cv2
import os
import os

from huggingface_hub import hf_hub_download
from PIL import Image
from onnxruntime import InferenceSession
from pathlib import Path
from typing import Union, Tuple, Dict

class Interrogator:
    def __init__(self, current_model_name: Union[str, None] = os.path.join("models", "model.onnx"), 
                 tags_path: Union[str, None] = os.path.join('selected_tags.csv')) -> None:
        self.current_model_name = current_model_name
        self.tags_path = tags_path
        self.model = None
        self.tags = None
        pass

    # SmilingWolf/wd-v1-4-moat-tagger-v2
    def download(self, repo_id: str, model_name: str) -> tuple[os.PathLike, os.PathLike]:
        #TODO: model names selection
        print(f"Downloading {model_name} from repo {repo_id}...")

        model_path = Path(hf_hub_download(repo_id=repo_id, filename=model_name)), 
        tags_path = Path(hf_hub_download(repo_id=repo_id, filename='selected_tags.csv'))
        return model_path, tags_path

    def load(self) -> None:
        model_path, tags_path = self.download()

        providers = ['CUDAExecutionProvider', 'ROCmExecutionProvider', 'CPUExecutionProvider']
        if use_cpu:
            providers = ['CPUExecutionProvider']
        #TODO: define use_cpu later

        self.model = InferenceSession(str(model_path), providers=providers)
        self.tags = pd.read_csv(tags_path)

    def interrogate(self, input_image) -> tuple[dict[str, float], dict[str, float]]:
        pass
        #TODO: define interrogate later
        #return tags, scores