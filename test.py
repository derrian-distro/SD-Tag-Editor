from utils.autoprune import prune_tags
from utils.interrogator import Interrogator
import json
import os
import shutil
from PIL import Image
from utils.jsonmanip import convert_to_tag_list


def main():
    inter = Interrogator()
    # model = inter.load_model("moat")
    # tags = inter.load_labels()

    # file_path = input("path to image: ")
    # file = Image.open(file_path)
    # rating_res, general_tags, character_tags = inter.interrogate(
    #     file, model, 0.35, 0.75, tags
    # )
    # full_tags = inter.find_groups(dict(general_tags + character_tags))
    # full_tags = prune_tags(full_tags)
    # with open("old-.json", "w") as f:
    #     json.dump((file_path, rating_res, full_tags), f, indent=2)

    folder = input("folder containing images: ")
    image_tags = inter.interrogate_folder(folder, "moat", 0.35, 0.75, True)
    for key, value in image_tags.items():
        image_tags[key] = {
            "file_path": value["file_path"],
            "rating": value["rating"],
            "tags": prune_tags(
                inter.find_groups(dict(value["general"] + value["character"]))
            ),
        }
    with open("test.json", "w") as f:
        json.dump(image_tags, f, indent=2)

    with open("test.txt", "w") as f:
        for value in image_tags.values():
            f.write(", ".join(convert_to_tag_list(value["tags"])) + "\n")

    # with open("test_unpruned.json", "w") as f:
    #     json.dump(full_tags, f, indent=2)

    # full_tags = prune_tags(full_tags)
    # full_tags = convert_to_tag_list(full_tags)
    # with open("output.txt", "w") as f:
    #     f.write(", ".join(full_tags))
    # with open("test.json", "w") as f:
    #     json.dump(full_tags, f, indent=2)

    # with open("test_unpruned.json", "r") as f:
    #     image_tags = json.load(f)

    # for value in full_tags.values():

    # value = (value[0], value[1], prune_tags(value[2]))


if __name__ == "__main__":
    main()
