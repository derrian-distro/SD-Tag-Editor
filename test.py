from utils.autoprune import prune_tags
from utils.interrogator import Interrogator
import json
import os
import shutil


def main():
    inter = Interrogator()
    folder = input("folder containing images: ")
    image_tags = inter.interrogate_folder(folder, "moat", 0.35, 0.75, True)
    for value in image_tags.values():
        value = (value[0], value[1], prune_tags(value[2]))

    with open("test.json", "w") as f:
        json.dump(image_tags, f, indent=4)


if __name__ == "__main__":
    main()
