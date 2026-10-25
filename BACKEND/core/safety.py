from core.textutil import norm

EMERGENCY = [
    "douleur thoracique", "douleur dans la poitrine", "douleur a la poitrine", "oppression thoracique",
    "difficulte a respirer", "du mal a respirer", "ne peux plus respirer", "perte de connaissance",
    "perdu connaissance", "evanoui", "convulsion", "paralysie", "bouche de travers",
    "difficulte a parler", "hemorragie", "saignement abondant", "crache du sang", "vomit du sang",
    "reaction allergique grave", "gonflement du visage", "gonflement de la gorge", "overdose", "empoisonn",
]
DISTRESS = [
    "me tuer", "envie de mourir", "veux mourir", "mettre fin a mes jours", "me suicider",
    "me faire du mal", "plus envie de vivre",
]

EMERGENCY_MESSAGE = (
    "Ce que vous décrivez peut nécessiter une prise en charge urgente. "
    "Contactez sans attendre les services d'urgence de votre pays ou rendez-vous aux urgences les plus proches."
)
SUPPORT_MESSAGE = (
    "Je suis désolé que vous traversiez un moment aussi difficile. Vous n'êtes pas seul(e) et vous méritez du soutien. "
    "Si vous pensez à vous faire du mal, parlez-en dès maintenant à une personne de confiance, ou contactez "
    "les services d'urgence ou une ligne d'écoute de votre pays. Si vous le souhaitez, vous pouvez aussi me parler "
    "de ce que vous ressentez, ici."
)


def detect_alert(query: str):
    q = norm(query)
    if any(p in q for p in DISTRESS):
        return {"level": "support", "message": SUPPORT_MESSAGE}
    if any(p in q for p in EMERGENCY):
        return {"level": "emergency", "message": EMERGENCY_MESSAGE}
    return None
