def prune_tags(tag_group: dict, base_tag: str = None) -> dict:
    for key in tag_group:
        if isinstance(tag_group[key], dict):
            prune_tags(tag_group[key], base_tag if key == "base" else key)
        elif len(tag_group[key]) > 1:
            if key in ["other", "modifier", "character", "nudity"]:
                continue
            sorted_array = sorted(
                tag_group[key], key=lambda x: list(x.values())[0], reverse=True
            )
            if base_tag in sorted_array[0] or key in sorted_array[0]:
                sorted_array.pop(0)
            tag_group[key] = [sorted_array[0]]
    return tag_group
