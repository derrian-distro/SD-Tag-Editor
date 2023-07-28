from typing import Any, Dict


SKIP_KEYS = ["other", "modifier", "character", "nudity"]
MODIFIER_TAGS = {
    "multicolored": 1,
    "heterochromia": 2,
    "two-tone": 2,
    "gradient": 2,
}
GROUP_PRUNE_LIST = ["solo", "1girl", "1boy", "1other"]


def prune_tags(tag_group: Dict[str, Any]) -> Dict[str, Any]:
    if "image composition" in tag_group and "group" in tag_group["image composition"]:
        tag_group["image composition"]["group"] = sorted(
            tag_group["image composition"]["group"], key=sorted_func, reverse=True
        )
        if len(tag_group["image composition"]["group"]) > 1:
            tags = tag_group["image composition"]["group"]
            dont_prune = next(iter(tags[0].keys())) in ["1girl", "1boy"] and next(
                iter(tags[1].keys())
            ) in ["1girl", "1boy"]
            tag_group["image composition"]["group"] = tags[:2]
        else:
            dont_prune = (
                next(iter(tag_group["image composition"]["group"][0].keys()))
                not in GROUP_PRUNE_LIST
            )
    else:
        dont_prune = False
    return prune_tags_helper(tag_group, dont_prune=dont_prune)


def prune_tags_helper(
    tag_group: Dict[str, Any],
    base_tag: str = None,
    dont_prune: bool = None,
    top_level: str = None,
) -> Dict[str, Any]:
    for key, group in tag_group.items():
        if isinstance(group, dict):
            prune_tags_helper(
                group,
                base_tag if key == "base" else key,
                dont_prune,
                top_level or key,
            )
        elif len(group) > 1 and key not in SKIP_KEYS:
            keep_array = []
            sorted_array = sorted(group, key=sorted_func, reverse=True)
            if top_level in {"attire", "body trait"} and dont_prune:
                handle_base_tag(sorted_array, base_tag, key)
                tag_group[key] = sorted_array
                continue
            if key == "group":
                continue
            modifier = handle_priority(sorted_array, keep_array)
            for tag in sorted_array:
                if base_tag in tag or key in tag:
                    continue
                keep_array.append(tag)
                modifier -= 1
                if modifier < 1:
                    break
            tag_group[key] = keep_array
    return tag_group


def prune_existing_tags(tag_group: dict, base_tag: str = None) -> dict:
    for key in tag_group:
        if isinstance(tag_group[key], dict):
            prune_existing_tags(
                tag_group[key], base_tag=base_tag if key == "base" else key
            )
        elif len(tag_group[key]) > 1:
            tag_group[key] = handle_base_tag(tag_group[key], base_tag, key)
    return tag_group


def handle_base_tag(sorted_array: list, base_tag: str, current_key: str) -> list:
    i = 0
    while i < len(sorted_array):
        if base_tag in sorted_array[i] or current_key in sorted_array[i]:
            sorted_array.pop(i)
            break
        i += 1
    return sorted_array


def handle_priority(sorted_array: list, keep_array: list) -> int:
    priority = 0
    modifier = 0
    while "priority" in sorted_array[0]:
        if sorted_array[0]["priority"] <= priority:
            sorted_array.pop(0)
            continue
        modifier = 2
        priority = sorted_array[0]["priority"]
        if not keep_array:
            keep_array.append(sorted_array[0])
        keep_array[0] = sorted_array[0]
        sorted_array.pop(0)
    return modifier


def sorted_func(x: dict) -> float:
    tag = list(x.keys())[0].split(" ")[0]
    if tag in MODIFIER_TAGS:
        x["priority"] = MODIFIER_TAGS[tag]
    return x.get("priority", list(x.values())[0])
