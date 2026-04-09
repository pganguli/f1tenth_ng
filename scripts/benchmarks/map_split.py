"""Canonical train/non-train map split for the F1TENTH benchmark suite."""

TRAIN_MAPS: list[str] = [
    "Austin",
    "BrandsHatch",
    "Budapest",
    "Catalunya",
    "Hockenheim",
    "IMS",
    "Melbourne",
    "Montreal",
    "MoscowRaceway",
    "Oschersleben",
    "Sakhir",
    "SaoPaulo",
    "Sepang",
    "Spielberg",
    "YasMarina",
    "Zandvoort",
]

NON_TRAIN_EVAL_MAPS: list[str] = [
    "MexicoCity",
    "Monza",
    "Nuerburgring",
    "Shanghai",
    "Silverstone",
    "Sochi",
    "Spa",
]

# Backward-compatible alias for older scripts that refer to eval maps.
EVAL_MAPS: list[str] = list(NON_TRAIN_EVAL_MAPS)

SANITY_TRAIN_MAPS: list[str] = [
    "Austin",
    "Budapest",
    "Melbourne",
    "Sakhir",
    "Zandvoort",
]

SANITY_EVAL_MAPS: list[str] = [
    "Nuerburgring",
    "Sochi",
    "Spa",
]

ALL_MAPS: list[str] = sorted(TRAIN_MAPS + NON_TRAIN_EVAL_MAPS)
