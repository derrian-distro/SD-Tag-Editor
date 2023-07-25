def convert_to_tag_list(tag_groups: dict):
    tags = []
    for value in tag_groups.values():
        if isinstance(value, dict):
            tags.extend(iter(convert_to_tag_list(value)))
        elif value:
            tags.extend(next(iter(tag.keys())) for tag in value)
        else:
            return tags
    return tags
