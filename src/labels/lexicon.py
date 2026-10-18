"""Multilingual lexicon for knee-MRI report label extraction (v0).

Languages confirmed present in train.csv: Spanish, Dutch, English (others likely; extensible).
Terms are accent-folded + lowercased before matching, so write them folded (no accents).
Each of the 12 targets is either:
  - pathology-only (the pathology term IS the target), or
  - anatomy x pathology (anatomy term AND pathology term must co-occur in one clause).
"""

# shared pathology term groups (accent-folded, lowercase)
PATH = {
    "tear": ["tear", "torn", "rupture", "ruptured", "rotura", "ruptura", "desgarro",
             "roto", "rota", "scheur", "ruptuur", "gescheurd", "lesion",
             "rhxh", "ρηξη", "ρηξης", "ρηξεις"],  # Greek rupture/tear (multiple GR reports)
    "oa": ["osteoarthritis", "arthrosis", "degenerative", "degeneration", "chondral loss",
           "cartilage loss", "artrosis", "degenerativ", "condropat", "condral", "artrose",
           "degeneratief", "kraakbeen", "gonartrosis",
           # cartilage/chondral expressions dominate OA phrasing (linguistic audit)
           "chondropath", "chondromalacia", "condromalacia", "chondral ulcer",
           "ulcera condral", "ulceras condral", "cartilage lesion", "cartilage defect",
           "kraakbeenlijden", "osteophyte", "osteofito", "chondrale", "chondraal",
           # missed OA phrasings (linguistic audit of gold-positive misses)
           "joint space narrowing", "jsn", "joint space loss", "pinzamiento",
           "subchondral", "subcondral", "osteoarthritic",
           "εκφυλιστικ", "εκφυλισ", "αρθρωσ", "αρθριτιδ", "οστεοαρθρ"],  # Greek degenerative/arthrosis
    "effusion": ["effusion", "joint fluid", "derrame", "derrame articular", "vocht",
                 "gewrichtsvocht", "effusie", "hydrops"],
    "synovitis": ["synovitis", "synovial", "sinovitis", "sinovial", "synovitis",
                  "synoviale"],
    "baker": ["baker", "popliteal cyst", "quiste popliteo", "quiste de baker",
              "bakercyste", "popliteale cyste", "quiste poplíteo"],
    "contusion": ["contusion", "bone bruise", "bone marrow edema", "bone edema",
                  "contusion osea", "edema oseo", "edema medular", "contusie",
                  "botoedeem", "beenmergoedeem", "bone marrow oedema"],
    "fracture": ["fracture", "fractured", "fractura", "fisura", "fractuur", "breuk",
                 "avulsion", "avulsion fracture"],
}

ANATOMY = {
    "acl": ["anterior cruciate", "acl", "ligamento cruzado anterior", "lca",
            "cruzado anterior", "voorste kruisband", "vkb", "lig cruzado anterior"],
    "mcl": ["medial collateral", "mcl", "ligamento colateral medial",
            "ligamento colateral interno", "lcm", "colateral interno",
            "mediale collaterale", "mcl band"],
    "med_meniscus": ["medial meniscus", "menisco interno", "menisco medial",
                     "mediale meniscus", "binnenmeniscus"],
    "lat_meniscus": ["lateral meniscus", "menisco externo", "menisco lateral",
                     "laterale meniscus", "buitenmeniscus"],
    "medial_comp": ["medial femorotibial", "femorotibial medial", "femorotibial interno",
                    "compartimento medial", "medial compartment", "mediale compartiment",
                    "femorotibiaal mediaal", "mediaal femorotibiaal", "condilo femoral medial",
                    "medial femoral condyle", "meseta tibial medial", "platillo tibial medial",
                    "medial tibial plateau", "femoral condyle medial", "εσω διαμερισμα", "εσω",
                    # tricompartmental OA fires every compartment
                    "tricompartmental", "tricompartimental", "three compartment",
                    "all three compartment", "pancompartmental"],
    "lateral_comp": ["lateral femorotibial", "femorotibial lateral", "femorotibial externo",
                     "compartimento lateral", "lateral compartment", "laterale compartiment",
                     "femorotibiaal lateraal", "lateraal femorotibiaal", "condilo femoral lateral",
                     "lateral femoral condyle", "meseta tibial lateral", "platillo tibial lateral",
                     "lateral tibial plateau", "femoral condyle lateral", "εξω διαμερισμα", "εξω",
                     "tricompartmental", "tricompartimental", "three compartment",
                     "all three compartment", "pancompartmental"],
    "patellofemoral": ["patellofemoral", "femoropatelar", "patelofemoral", "retropatelar",
                       "patellofemoraal", "femoropatellar", "rotuliano femoral", "trochlea",
                       "troclea", "patellar facet", "faceta de la rotula",
                       "faceta lateral de la rotula", "faceta medial de la rotula",
                       "rotulian", "trochlear", "patelar",
                       "tricompartmental", "tricompartimental", "three compartment",
                       "all three compartment", "pancompartmental"],
}

# negation cues (folded). Scope: same clause, cue anywhere in clause.
NEGATION = ["no", "sin", "ausencia", "ausente", "no se observ", "no hay", "negativ",
            "descarta", "without", "absence", "no evidence", "negative for", "unremarkable",
            "normal", "intact", "integro", "integra", "conservad", "preservad",
            "geen", "zonder", "afwezig", "intact", "normaal", "niet",
            # Greek negation/normal cues (added with Greek pathology terms to avoid FPs on negated findings)
            "δεν", "χωρις", "ουδεν", "ουδενος", "μη ", "φυσιολογικ", "ακεραι", "ελευθερ"]

UNCERTAINTY = ["possible", "probable", "suggestive", "cannot exclude", "equivocal",
               "suspicious", "posible", "probable", "sugestiv", "dudos", "sospech",
               "no se puede excluir", "mogelijk", "waarschijnlijk", "verdenking", "mogelijke"]

# target -> (kind, anatomy_key_or_None, pathology_key, default_assessed)
# kind: "path" (pathology-only) or "anat_path" (anatomy x pathology)
TARGET_RULES = {
    "ACL": ("anat_path", "acl", "tear", True),
    "MCL": ("anat_path", "mcl", "tear", False),
    "Medial Meniscus": ("anat_path", "med_meniscus", "tear", True),
    "Lateral Meniscus": ("anat_path", "lat_meniscus", "tear", True),
    "Medial OA": ("anat_path", "medial_comp", "oa", True),
    "Lateral OA": ("anat_path", "lateral_comp", "oa", True),
    "PF OA": ("anat_path", "patellofemoral", "oa", True),
    "Effusion": ("path", None, "effusion", True),
    "Synovitis": ("path", None, "synovitis", False),
    "Baker's": ("path", None, "baker", False),
    "Contusion": ("path", None, "contusion", False),
    "Fracture": ("path", None, "fracture", False),
}
