import json
import os
import sys
from collections import UserDict
from enum import Enum
from pathlib import Path
from typing import Iterable, Optional, TypeAlias

from pydantic import BaseModel, RootModel, computed_field

groups_filename = "tag_groups.json"

work_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()

# fmt: off
COLOR_MODIFIERS = {
    "multicolored", "two-tone",
    "striped", "vertical_striped", "pinstripe",
    "plaid", "polka_dot", "checkered",
    "colored_inner_hair", "colored_tips", "streaked",
    "heterochromia"
}

GROUP_MODIFIERS = {
    "2boys", "2girls", "2others",
    "3boys", "3girls", "3others",
    "4boys", "4girls", "4others",
    "5boys", "5girls", "5others",
    "6+boys", "6+girls", "6+others",
    "multiple_boys", "multiple_girls", "multiple_others",
    "furry_with_non-furry", "furry_with_furry",
}
# fmt: on
ONLY_PRUNE_BASE = {"attire", "expression", "gesture", "position", "trait"}


class TagCategory(str, Enum):
    base = "base"
    modifier = "modifier"
    length = "length"
    subtype = "subtype"
    other = "other"


class ScoredTag(RootModel):
    root: tuple[str, float]

    def __init__(self, name: str, score: float):
        super().__init__(root=(name, score))

    @computed_field
    @property
    def name(self) -> str:
        return self.root[0]

    @computed_field
    @property
    def score(self) -> float:
        return self.root[1]

    def __iter__(self):
        return iter(self.root)

    def __getitem__(self, item: int):
        return self.root[item]


class Group(BaseModel):
    base: list[str] = []
    modifier: list[str] = []
    length: list[str] = []
    subtype: list[str] = []

    def get_all(self) -> list[str]:
        return self.base + self.modifier + self.length + self.subtype

    def find_tag(self, tag: str | ScoredTag) -> str | None:
        if isinstance(tag, ScoredTag):
            tag = tag.name

        if tag in self.base:
            return TagCategory.base
        if tag in self.modifier:
            return TagCategory.modifier
        if tag in self.length:
            return TagCategory.length
        if tag in self.subtype:
            return TagCategory.subtype
        return None


TagDict: TypeAlias = dict[str, Group | list[str]]


class ScoredGroup(Group):
    base: list[ScoredTag] = []
    modifier: list[ScoredTag] = []
    length: list[ScoredTag] = []
    subtype: list[ScoredTag] = []

    def find_tag(self, tag: str | ScoredTag) -> str | None:
        raise NotImplementedError("ScoredGroup does not support find_tag")

    def update(self, tag: ScoredTag, category: str):
        getattr(self, category).append(tag)


ScoredTagDict = dict[str, ScoredGroup | list[ScoredTag]]


class GroupTree(RootModel, UserDict):
    root: dict[str, TagDict | list[str]]

    @property
    def data(self):
        return self.root

    # these 3 methods are wrapped purely for typechecking
    def items(self) -> Iterable[tuple[str, ScoredTagDict]]:
        return self.root.items()

    def keys(self) -> Iterable[str]:
        return self.root.keys()

    def values(self) -> Iterable[ScoredTagDict]:
        return self.root.values()


class ScoredGroupTree(RootModel, UserDict):
    root: dict[str, ScoredTagDict | list[ScoredTag]]

    @property
    def data(self):
        return self.root

    # these 3 methods are wrapped purely for typechecking
    def items(self) -> Iterable[tuple[str, ScoredTagDict]]:
        return self.root.items()

    def keys(self) -> Iterable[str]:
        return self.root.keys()

    def values(self) -> Iterable[ScoredTagDict]:
        return self.root.values()


def get_resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


def load_groups(fpath: Optional[Path] = "tag_groups.json") -> GroupTree:
    # Try multiple locations for the file
    possible_paths = [
        Path(fpath).resolve(),  # Original path
        Path(get_resource_path(fpath)).resolve(),  # PyInstaller path
        work_dir / fpath,  # Relative to script directory
        Path.cwd() / fpath,  # Relative to current working directory
    ]

    for path in possible_paths:
        if path.is_file():
            return GroupTree.model_validate_json(path.read_text(encoding="utf-8"))

    raise FileNotFoundError(
        f"Could not find tag groups file. Searched in: {[str(p) for p in possible_paths]}"
    )


def traverse_tag_tree(groups: GroupTree, tag: ScoredTag) -> tuple[str, str, str | None]:
    """
    tag: the input tag
    tuple[str] -> outputs a list of the traversed groups to find the tag
    """
    for root_name, root_group in groups.items():
        for group_name, group in root_group.items():
            if group_name == "other":
                if tag.name in group:
                    return [root_name, group_name, None]
                continue
            subgroup = group.find_tag(tag)
            if subgroup is not None:
                return [root_name, group_name, subgroup]
    return ["other", "other", None]


def build_tag_tree(groups: GroupTree, tags: dict[str, float]) -> ScoredGroupTree:
    output_tree: ScoredGroupTree = ScoredGroupTree({})

    tag: ScoredTag
    for tag in tags:
        root_key, group_key, category = traverse_tag_tree(groups, tag)

        if root_key not in output_tree:
            output_tree[root_key] = ScoredTagDict()

        if group_key not in output_tree[root_key]:
            output_tree[root_key][group_key] = [] if category is None else ScoredGroup()

        if category is None:
            output_tree[root_key][group_key].append(tag)
            continue
        else:
            output_tree[root_key][group_key].update(tag, category)
            continue

    return output_tree


def determine_prune_type(img_comp_tags: dict[str, ScoredGroup | list[ScoredTag]]) -> tuple[bool, int]:
    SOLO_TAGS = ["1girl", "1boy", "1other"]
    other: list[ScoredTag] = img_comp_tags.get("other", [])
    has_multi_view = any(tag.name == "multiple_views" for tag in other)

    if img_comp_tags.get("group") is None or (not img_comp_tags["group"].base):
        return has_multi_view, 1

    group_tags_sorted = sorted(img_comp_tags["group"].base, key=lambda x: x.score, reverse=True)
    if group_tags_sorted[0].name in GROUP_MODIFIERS:
        return True, 1

    if len(group_tags_sorted) == 1:
        return has_multi_view, 1

    if all(tag.name in SOLO_TAGS for tag in group_tags_sorted[:2]):
        return True, 2

    return has_multi_view, 1


def prune_tags(tag_tree: ScoredGroupTree) -> ScoredGroupTree:
    multi_char, keep_count = determine_prune_type(tag_tree.get("image_composition", {}))
    for root_name, root_group in tag_tree.items():
        for group_name, group in root_group.items():
            if group_name == "other":
                continue

            mismatched = group.modifier and bool([x for x in group.modifier if "mismatched" in x.name])
            group.base = prune_base(
                group_name,
                group.base,
                bool(group.modifier),
                mismatched,
                keep_count if group_name == "group" else 1,
                multi_char if root_name in ONLY_PRUNE_BASE else False,
            )
            if root_name in ONLY_PRUNE_BASE and multi_char is True:
                continue
            if group.length:
                group.length = [sorted(group.length, key=lambda x: x.score, reverse=True)[0]]
            if group.subtype:
                group.subtype = [sorted(group.subtype, key=lambda x: x.score, reverse=True)[0]]
            tag_tree[root_name][group_name] = group
    return tag_tree


def process_color_tags(
    tag_list: list[ScoredTag], has_mismatched: bool
) -> tuple[list[ScoredTag], list[ScoredTag], int]:
    mod_tags: list[ScoredTag] = []

    for color_tag in COLOR_MODIFIERS:
        temp_tags = [x for x in tag_list if color_tag in x.name]
        mod_tags.extend(temp_tags)
    for mod_tag in mod_tags:
        tag_list.remove(mod_tag)
    mod_tags = sorted(mod_tags, key=lambda x: x.score, reverse=True)
    if len(mod_tags) > 1:
        for tag in mod_tags:
            if "multicolored" in tag.name:
                mod_tags.remove(tag)
                break

    mod_tags = mod_tags[0 : 2 if has_mismatched else 1]
    if len(mod_tags) > 1:
        color_keep = 4
    elif mod_tags and has_mismatched:
        color_keep = 3
    elif mod_tags:
        color_keep = 2
    else:
        color_keep = 1

    return tag_list, mod_tags, color_keep


def prune_base(
    group_name: str,
    tag_list: list[ScoredTag],
    has_mod: bool = False,
    has_mismatched: bool = False,
    num_to_keep: int = 1,
    prune_name_only: bool = False,
) -> list[ScoredTag]:
    match len(tag_list):
        case 0:
            return []
        case 1:
            if tag_list[0].name == group_name and has_mod:
                return []
            else:
                return tag_list
        case _:
            pass

    if prune_name_only:
        return [x for x in tag_list if x.name != group_name]

    tag_list, mod_tags, color_tags = process_color_tags(tag_list, has_mismatched)

    tag_list = [x for x in tag_list if x.name != group_name]
    tag_list = sorted(tag_list, key=lambda x: x.score, reverse=True)
    tag_list = tag_list[0 : max(num_to_keep, color_tags)]
    if mod_tags:
        tag_list.extend(mod_tags)
    return tag_list


def flatten_tags(
    tag_tree: ScoredGroupTree,
    with_probs: bool = False,
) -> list[str] | list[ScoredTag]:
    tags: list[ScoredTag] = []
    for root_group in tag_tree.values():
        for name, group in root_group.items():
            tags.extend(group if name == "other" else group.get_all())

    if with_probs:
        return tags
    else:
        return [x.name for x in tags]


def prune(
    groups: GroupTree,
    tags: dict[str, float] | list[ScoredTag],
) -> ScoredGroupTree:
    if isinstance(tags, dict):
        tags = [ScoredTag(name, score) for name, score in tags.items()]
    tag_tree = build_tag_tree(groups, tags)
    pruned = prune_tags(tag_tree)
    return pruned


def main(debug: bool = True):
    input_dir = work_dir.joinpath("inputs")
    output_dir = work_dir.joinpath("pruned")

    group_tree: GroupTree = load_groups(get_resource_path(groups_filename))

    input_files = [x for x in input_dir.iterdir() if x.suffix.lower() == ".json"]
    for tfile in input_files:
        out_file = output_dir.joinpath(tfile.name)
        # load tags
        tags = json.loads(tfile.read_text(encoding="utf-8"))

        # build and prune tag tree
        pruned = prune(group_tree, tags)
        # write pruned tree to file
        out_file.write_text(pruned.model_dump_json(exclude_none=True, indent=2))
        print(f"{tfile.name} pruned to: {out_file.name}")
        if debug:
            # print results
            print("Flattened tag tree:")
            print(flatten_tags(pruned, True))


if __name__ == "__main__":
    main()
