# Tagging Project

## WD Tagger has confidence - can make use of it, intelligently combine tags

Make use of confidence and tag groups to merge tags

Without tag groups - it wouldn't know to merge "bikini" and "white bikini"

### Tag everything, including confidence, and that is stored into a file and read out when processing when ready for editing

Confidences are naturally "there" - dict?
If a tag is "manually added" then assign it to a confidence of 1
    If a tag is manually added but already exists in the dataset, we take the average of the captions confidences.
        Take 20 images, manually add one - 19 automatics, 1 manual 100%, and average.
    Use averages to figure out which tag to delete
        Per image, it'll look at combinations it can do,
    If user selects two tags, it's one or the other
        "pantyhose" or "black-band pantyhose" -> find all images that have both of these tags, dynamically select whichever one has the highest confidence in each image, and the lowest one goes away.
    No combining tags - breaks booru tag style structure
    All tags the tagger outputs is what we can handle - tag groups need to be all encompassing to pretty much all of these
        - though, we are working with primarily working with descriptors
Make tag groups editable - user can create their own.

### Tag Groups

Tag groups are used for sorting tags
    all tags that are hair styles, all tags that are tops, bottoms, et cetera. Helps the user to find these.
    Assign categories - "clothing" -> top, bottom, et cetera.
        -> user can mass remove clothing or group tags.
    Think of the Character Traits option in my extension
        -> remove categories or groups of tags, more granular control
If user wants to "change all pantyhose type tags" into "black pantyhose"
    -> if it cannot find black pantyhose, or pantyhose at all, then it won't do anything
The goal when leaving or changing tags is to not overchange tags
    -> We don't want the tag editor to automatically change every instance of pantyhose to black pantyhose if it contains a white pantyhose, for example
    -> Start with images that have both pantyhose and black pantyhose, and then go from there
    -> If it doesn't find "black pantyhose" but contains something else like "white pantyhose" then it won't do anything
        -> Manual overrides? Force change.

### GUI

API system?
    -> frontend and backend separated
We set up al lthe code so there's a specific library of functions that do all of the functions - listing, tagging, etc
    Abstracted away
Someone could make a Gradio interface if they would like
Developers can use our tagger without needing to understand it

>Set up environment for PySide6

Start with the GUI - will inform us what endpoint functions we need in the API
    GUI and API are separate repos

## Steps

> Set up PySide6
    > Learn how to use
    > Set up modules, submodules, et cetera.
    > Do we want to use Qt Design Studio? or implement by hand? **QT Creator**
    
> Figure out our pre-run steps
    > DLing tagger
        >.onnx -> requires onnx dependencies & model DL
            >moatv2, convnextv2-v2, swinv2, vitv2
            >model DL will use `from huggingface_hub import hf_hub_download`
            >Build onnxruntime-gpu with rocm for AMD

## Requirements

> pip freeze
> list of things to install:
    > PySide6==6.5.0
    > qt-material
    > onnxruntime, onnxruntime-gpu (onnxruntime is for arm cpu/macos)
        >if AMD, check if ROCm version is 5.4 (tentatively - confirm whether SD uses 5.4 normally)
    > huggingface-hub
    > pillow (image conversion for tagging), possibly for displaying images as well, I don't remember if PySide6 handles images well
        >cv2
    >pandas
    >numpy
