from utils.autoprune import prune_tags
from utils.interrogator import Interrogator
import json
import os
import shutil
from PIL import Image


def main():
    inter = Interrogator()
    model = inter.load_model("moat")
    tags = inter.load_labels()

    file = Image.open(input("path to image: "))
    _, full_tags = inter.interrogate(file, model, 0.35, 0.75, tags)

    # folder = input("folder containing images: ")
    # image_tags = inter.interrogate_folder(folder, "moat", 0.35, 0.75, True)

    with open("test_unpruned.json", "w") as f:
        json.dump(full_tags, f, indent=2)

    full_tags = prune_tags(full_tags)
    with open("test.json", "w") as f:
        json.dump(full_tags, f, indent=2)

    # # with open("test_unpruned.json", "r") as f:
    # #     image_tags = json.load(f)

    # for value in full_tags.values():
    #     value = (value[0], value[1], prune_tags(value[2]))

    # with open("test.json", "w") as f:
    #     json.dump(image_tags, f, indent=2)


if __name__ == "__main__":
    main()
