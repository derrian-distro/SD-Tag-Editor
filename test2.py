from utils.autoprune import prune_existing_tags
from utils.interrogator import Interrogator
from utils.jsonmanip import convert_to_tag_list


def main():
    inter = Interrogator()
    with open(input("txt file: "), "r") as f:
        tags = inter.find_groups({x: 0 for x in f.read().split(", ")})
        tags = prune_existing_tags(tags)
        tags = convert_to_tag_list(tags)
        print(tags)


if __name__ == "__main__":
    main()
