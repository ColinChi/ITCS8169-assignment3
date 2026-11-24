"""Class names and prompt templates. Fixed before any test result was computed."""

# "raw": the folder name split at capitals, nothing else.
RAW_NAMES = {
    "Bedroom": ["bedroom"],
    "Coast": ["coast"],
    "Flower": ["flower"],
    "Forest": ["forest"],
    "Highway": ["highway"],
    "Industrial": ["industrial"],
    "InsideCity": ["inside city"],
    "Kitchen": ["kitchen"],
    "LivingRoom": ["living room"],
    "Mountain": ["mountain"],
    "Office": ["office"],
    "OpenCountry": ["open country"],
    "Store": ["store"],
    "Street": ["street"],
    "Suburb": ["suburb"],
    "TallBuilding": ["tall building"],
}

# "syn": the raw name plus common synonyms (scene-category names as used in Places / SUN).
SYN_NAMES = {
    "Bedroom": ["bedroom"],
    "Coast": ["coast", "beach", "seashore"],
    "Flower": ["flower", "close-up of a flower"],
    "Forest": ["forest", "woods"],
    "Highway": ["highway", "freeway"],
    "Industrial": ["industrial area", "factory", "industrial site"],
    "InsideCity": ["inside city", "city center", "downtown street with buildings"],
    "Kitchen": ["kitchen"],
    "LivingRoom": ["living room"],
    "Mountain": ["mountain", "mountain range"],
    "Office": ["office"],
    "OpenCountry": ["open country", "countryside", "open field"],
    "Store": ["store", "shop interior", "supermarket aisle"],
    "Street": ["street", "city street"],
    "Suburb": ["suburb", "suburban house"],
    "TallBuilding": ["tall building", "skyscraper"],
}

# Every image except the Flower class is grayscale, hence the black-and-white templates.
TEMPLATES = [
    "a photo of a {}.",
    "a black and white photo of a {}.",
    "a grayscale photo of a {}.",
    "a photo of the {}.",
    "a picture of a {}.",
    "an image of a {}.",
    "a scene of a {}.",
    "a photo of a {} scene.",
    "a low resolution photo of a {}.",
    "a cropped photo of a {}.",
]

NAME_SETS = {"raw": RAW_NAMES, "syn": SYN_NAMES}


def prompts_for(names: list[str], templates: list[str] = TEMPLATES) -> list[str]:
    return [t.format(n) for n in names for t in templates]
