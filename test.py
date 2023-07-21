from utils.interrogator import Interrogator
import json
import os
import shutil


def main():
    inter = Interrogator()
    folder = input("folder containing images: ")
    image_tags = inter.interrogate_folder(folder, "moat", 0.35, 0.75, True)
    for file, scores, _ in image_tags.values():
        highest_score = tuple(scores[0])
        for score in scores:
            if score[1] > highest_score[1]:
                highest_score = tuple(score)

        shutil.move(file, os.path.join("filtered", highest_score[0]))

    with open("test.json", "w") as f:
        json.dump(image_tags, f, indent=4)


if __name__ == "__main__":
    main()
