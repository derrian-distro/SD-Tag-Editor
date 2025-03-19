from dataclasses import dataclass, field
from pathlib import Path

import orjson
from simple_parsing import parse_known_args
from tqdm import tqdm

from run import MODEL_REPO_MAP, ScriptOptions, get_tags, run_model, setup
from tag_tree_functions import GroupTree, flatten_tags, prune


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


def main(opts: ScriptOptions):
    batches, model, labels, transform, group_tree = setup(
        opts.model, opts.image_or_images, opts.subfolder, opts.batch_size
    )
    for batch in tqdm(batches):
        batch: list[Path]
        outputs = run_model(model, transform, batch)
        for i, img in enumerate(outputs):
            js = batch[i].with_suffix(".json")
            char, gen, rating = get_tags(
                probs=img, labels=labels, gen_threshold=opts.gen_threshold, char_threshold=opts.char_threshold
            )
            artist = []
            if js.is_file():
                data = orjson.loads(js.read_bytes())
                for x in data["character"]:
                    char[x] = (char.get(x, opts.char_threshold) + opts.char_threshold) / 2
                for x in data["general"]:
                    gen[x] = (gen.get(x, opts.gen_threshold) + opts.gen_threshold) / 2
                artist = data["artist"]
            item = process_tags(
                group_tree=group_tree, char_labels=char, gen_labels=gen, artists=artist, ratings=rating
            )
            batch[i].with_suffix(".json").write_bytes(orjson.dumps(item, option=orjson.OPT_INDENT_2))


if __name__ == "__main__":
    opts, _ = parse_known_args(ScriptOptions)
    if opts.model not in MODEL_REPO_MAP:
        print(f"Available models: {list(MODEL_REPO_MAP.keys())}")
        raise ValueError(f"Unknown model name '{opts.model}'")
    main(opts)
