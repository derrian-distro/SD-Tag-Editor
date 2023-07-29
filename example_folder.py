from utils.autoprune import prune_tags
from utils.interrogator import Interrogator
import json
from utils.jsonmanip import convert_to_tag_list


def main():
    # load the interrogator class
    inter = Interrogator()
    # get the folder containing all images
    folder = input("folder containing images: ")
    # tag all images, including all sub folders
    image_tags = inter.interrogate_folder(folder, "moat", 0.35, 0.75, subfolders=True)
    """
        interrogate folder outputs like this:
        {
            "image_name.ext": {
                "file_path": full file path,
                "rating": the list of the 4 ratings, general, sensitive, questionable, and explicit
                "general": all of the general tags in a [{"tag_name": confidence}], confidence being a float
                "character": all of the character tags in a [{"tag_name": confidence}], confidence being a float
            },
            {
                ...
            },
            ...
        }
    """
    # for each image, convert the tags to the tag tree format, then prune them
    for key, value in image_tags.items():
        # convert to a dict because it is assumed to be {"tag": confidence}
        # and the default outputs are tuples
        tag_tree = inter.find_groups(value["general"] + value["character"])
        image_tags[key] = {
            "file_path": value["file_path"],
            "rating": value["rating"],
            "tags": prune_tags(tag_tree),
        }
    # output the images in tag tree format so that the user can see what it looks like
    with open("test.json", "w") as f:
        json.dump(image_tags, f, indent=2)

    with open("test.txt", "w") as f:
        for value in image_tags.values():
            # output the tags into a txt file after converting them to a
            # list of tags ["tag1", "tag2", "tag3", ...]
            tag_list = convert_to_tag_list(value["tags"])
            # convert the tag list from a list to a string separated by ", "
            # and write it to a new line in the txt file
            f.write(", ".join(tag_list) + "\n")


if __name__ == "__main__":
    main()
