from utils.autoprune import prune_tags
from utils.interrogator import Interrogator
from pathlib import Path
import json

from utils.jsonmanip import convert_to_tag_list


def main():
    inter_swin = Interrogator(model="swin")
    inter_conv = Interrogator(model="conv")
    tag_folder = Path(input("folder to tag: "))
    if not tag_folder.exists():
        exit()
    tagged_images_swin = inter_swin.interrogate_folder(
        image_folder=tag_folder.resolve(),
        general_threshold=0.35,
        character_threshold=0.75,
        subfolders=False,
    )
    tagged_images_conv = inter_conv.interrogate_folder(
        image_folder=tag_folder.resolve(),
        general_threshold=0.35,
        character_threshold=0.75,
        subfolders=False,
    )
    for key, val in tagged_images_swin.items():
        comp = tagged_images_conv[key]
        val["rating"] = compare_ratings(val["rating"], comp["rating"])
        val["general"] = compare_tags(val["general"], comp["general"])
        val["character"] = compare_tags(val["character"], comp["character"])
    for image in tagged_images_swin.keys():
        new_path = Path(tagged_images_swin[image]["file_path"]).with_suffix(".txt")
        print(f"writing tags to: {new_path}")
        tagged_images_swin[image]["general"].update(
            tagged_images_swin[image]["character"]
        )
        tags = convert_to_tag_list(
            prune_tags(inter_swin.find_groups(tagged_images_swin[image]["general"]))
        )

        with new_path.open("w", encoding="utf-8") as f:
            f.write(", ".join(tags))


def compare_ratings(
    r1: list[tuple[str, float]], r2: list[tuple[str, float]]
) -> list[tuple[str, float]]:
    return [(r1[i][0], (r1[i][1] + r2[i][1]) / 2) for i in range(len(r1))]


def compare_tags(t1: dict, t2: dict) -> dict:
    for key, val in t1.items():
        if key in t2:
            t1[key] = (val + t2[key]) / 2
            t2.pop(key)
    for key, val in t2.items():
        t1[key] = val
    return t1


if __name__ == "__main__":
    main()
