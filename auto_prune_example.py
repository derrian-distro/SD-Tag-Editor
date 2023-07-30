from utils.interrogator import Interrogator
from utils.autoprune import prune_tags
from utils.jsonmanip import convert_to_tag_list
import pathlib


def main():
    inter = Interrogator()

    tag_folder = pathlib.Path(input("folder to tag: "))
    if not tag_folder.exists():
        exit()
    tagged_images = inter.interrogate_folder(
        image_folder=tag_folder.resolve(),
        general_threshold=0.35,
        character_threshold=0.75,
        subfolders=False,
    )
    for image in tagged_images.keys():
        new_path = pathlib.Path(tagged_images[image]["file_path"]).with_suffix(".txt")
        print(f"writing tags to: {new_path}")
        tags = convert_to_tag_list(
            prune_tags(inter.find_groups(tagged_images[image]["general"]))
        )
        with new_path.open("w") as f:
            f.write(", ".join(tags))


if __name__ == "__main__":
    main()
