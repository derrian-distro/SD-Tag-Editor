from dataclasses import dataclass, field
from run import get_tags, run_model, setup
from tag_tree_functions import flatten_tags, prune, GroupTree
from pathlib import Path
from tqdm import tqdm
import orjson


@dataclass
class Item:
    character: list[str] = field(default_factory=lambda: [])
    general: list[str] = field(default_factory=lambda: [])
    artist: list[str] = field(default_factory=lambda: [])
    rating: dict[str, float] = field(default_factory=lambda: {})


def process_tags(
    group_tree: GroupTree,
    char_labels: dict[str, float],
    gen_labels: dict[str, float],
    ratings: dict[str, float],
    artists: list[str] = None,
) -> Item:
    if artists is None:
        artists = []
    char_labels = list(char_labels.keys())
    gen_labels = flatten_tags(prune(group_tree, dict(gen_labels)), True)
    gen_labels = [x[0] for x in sorted(gen_labels, key=lambda x: x[1], reverse=True)]
    return Item(
        character=char_labels,
        general=gen_labels,
        artist=artists,
        rating={str(x): float(y) for x, y in ratings.items()},
    )


def main():
    batches, model, labels, transform, group_tree = setup(
        model="vit-large", image_or_images=input("in: ").strip('" '), subfolder=True, batch_size=500
    )
    for batch in tqdm(batches):
        batch: list[Path]
        outputs = run_model(model, transform, batch)
        for i, img in enumerate(outputs):
            js = batch[i].with_suffix(".json")
            char, gen, rating = get_tags(probs=img, labels=labels, gen_threshold=0.35, char_threshold=0.75)
            artist = []
            if js.is_file():
                data = orjson.loads(js.read_bytes())
                for x in data["character"]:
                    char[x] = (char.get(x, 0.75) + 0.75) / 2
                for x in data["general"]:
                    gen[x] = (gen.get(x, 0.75) + 0.75) / 2
                artist = data["artist"]
            item = process_tags(
                group_tree=group_tree, char_labels=char, gen_labels=gen, artists=artist, ratings=rating
            )
            batch[i].with_suffix(".json").write_bytes(orjson.dumps(item, option=orjson.OPT_INDENT_2))


if __name__ == "__main__":
    main()
