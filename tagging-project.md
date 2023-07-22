# Tagging Project

## WD Tagger has confidence - can make use of it, intelligently combine tags

* Make use of confidence and tag groups to merge tags

* Without tag groups - it wouldn't know to merge "bikini" and "white bikini"

### Tag everything, including confidence, and that is stored into a file and read out when processing when ready for editing

* Confidences are naturally "there" - dict?
* If a tag is "manually added" then assign it to a confidence of 1
    * If a tag is manually added but already exists in the dataset, we take the average of the captions confidences.
        * Take 20 images, manually add one - 19 automatics, 1 manual 100%, and average.
    * Use averages to figure out which tag to delete
        * Per image, it'll look at combinations it can do,
    * If user selects two tags, it's one or the other
        * "pantyhose" or "black-band pantyhose" * find all images that have both of these tags, dynamically select whichever one has the highest confidence in each image, and the lowest one goes away.
    * No combining tags - breaks booru tag style structure
    * All tags the tagger outputs is what we can handle - tag groups need to be all encompassing to pretty much all of these
        * though, we are working with primarily working with descriptors
* Make tag groups editable - user can create their own.

### Tag Groups

* Tag groups are used for sorting tags
    * all tags that are hair styles, all tags that are tops, bottoms, et cetera. Helps the user to find these.
    * Assign categories - "clothing" * top, bottom, et cetera.
        * user can mass remove clothing or group tags.
    * Think of the Character Traits option in my extension
        * remove categories or groups of tags, more granular control
* If user wants to "change all pantyhose type tags" into "black pantyhose"
    * if it cannot find black pantyhose, or pantyhose at all, then it won't do anything
* The goal when leaving or changing tags is to not overchange tags
    * We don't want the tag editor to automatically change every instance of pantyhose to black pantyhose if it contains a white pantyhose, for example
    * Start with images that have both pantyhose and black pantyhose, and then go from there
    * If it doesn't find "black pantyhose" but contains something else like "white pantyhose" then it won't do anything
        * Manual overrides? Force change.


### Pruning Rules
* If character, don't prune!
* Prune depending on 1girl, 2girl, et cetera
Every section has subsections broken out as necessary
    * for each section that has [], we remove all but the most confident item
        * e.g., 
        ```
        "character": [
            "fat", (0.85) <- keep this
            "curvy", (0.76) <- remove (incl. below)
            "skinny", (0.02)
            "petite", (0.01)
        ]
        ```

    * for each section that has {}, we use it as a tree until we hit a [] section (keep descending into the tree)
        * e.g.,
        ```
        "body trait": { (descending downward through the groups until we hit square brackets)
            "injury": {
                "type": [
                    <removed for clarity>
                ],
                "scar": [
                    <removed for clarity>
                ]
                <removed for clarity>
            },
            <removed for clarity>
        },
        ```

    * specific sections (attire) require special exceptions
        * rule: if tag with the highest confidence that is not in "modifier" is the same as the group tag, then remove that tag and go with the second highest, if one exists.
            * e.g.,
            ```
            "attire": {
                "full body": {
                    "dress": {
                        "base": [
                            "dress", (removed (same name as "dress" tag group))
                            "print dress", (0.95) (kept, second highest confidence)
                            "multicolored dress", (0.02)
                            "aqua dress", (etc...)
                            "white dress",
                            "purple dress",
                            "yellow dress",
                            "green dress",
                            "grey dress",
                            "ribbed dress",
                            "gradient dress",
                            "two-tone dress",
                            "polka dot dress",
                            "red dress",
                            "plaid dress",
                            "orange dress",
                            "black dress",
                            "brown dress",
                            "silver dress",
                            "blue dress",
                            "pink dress",
                            "white robe",
                            "black robe"
                        ],
                        "type": [ (keep top confidence)
                            "open robe",
                            "robe",
                            "striped dress",
                            "vertical-striped dress",
                            "sundress",
                            "tube dress",
                            "microdress"
                        ],
                        "other": [ (keep top confidence)
                            "taut dress",
                            "tight dress",
                            "fur-trimmed dress",
                            "lace-trimmed dress",
                            "ribbon-trimmed dress"
                        ],
                        "modifier": [ (don't prune - can have multiple modifiers)
                            "backless dress",
                            "checkered dress",
                            "china dress",
                        ]
                    }
                }
            }
            ```

        * any section that has the "base" tag group also follow this rule
            


    * if we have a tag that is equal to the tag group name, then we remove that tag, and go with the second highest confident tag, if one exists.
        * e.g.,
        ```
        "attire": {
            "full body": {
                "dress": {
                    "base": [
                        "dress", (0.95) <- removed (same name as "dress" tag group)
                        "print dress", (0.89) <- next highest confidence, kept 
                    ]
                }
            }
        }
        ```
        
### GUI

* API system?
    * frontend and backend separated

* We set up all the code so there's a specific library of functions that do all of the functions - listing, tagging, etc
    * Abstracted away

* Someone could make a Gradio interface if they would like
    * Developers can use our tagger without needing to understand it

* Set up environment for PySide6

* Start with the GUI - will inform us what endpoint functions we need in the API
    * GUI and API are separate repos

## Steps

* Set up PySide6
    * Learn how to use
    * Set up modules, submodules, et cetera.
    * Do we want to use Qt Design Studio? or implement by hand? **QT Creator**
    
* Figure out our pre-run steps
    * DLing tagger
        * .onnx * requires onnx dependencies & model DL
            * moatv2, convnextv2-v2, swinv2, vitv2
            * model DL will use `from huggingface_hub import hf_hub_download`
            * Build onnxruntime-gpu with rocm for AMD
