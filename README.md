# 🏥 Assistant médical SMA — Multimodal RAG

Assistant médical francophone qui répond aux questions de santé à partir de **vos propres documents** (PDF, Word), complétés par une recherche web quand ils ne suffisent pas. Il cite ses sources (fichier et page), vérifie ses réponses, s'abstient quand il ne sait pas et signale les situations d'urgence.

> ⚠️ **Information à but éducatif uniquement.** Cet assistant ne remplace pas l'avis d'un professionnel de santé et ne pose aucun diagnostic.

---

## Sommaire

1. [Fonctionnalités](#fonctionnalités)
2. [Architecture](#architecture)
3. [Structure du projet](#structure-du-projet)
4. [Installation](#installation)
5. [Configuration (.env)](#configuration-env)
6. [Indexer les documents](#indexer-les-documents)
7. [Lancer l'application](#lancer-lapplication)
8. [API](#api)
9. [Limites connues](#limites-connues)
10. [Dépannage](#dépannage)

---

## Fonctionnalités

**Réponses fiables**
- **RAG sur documents locaux** : recherche sémantique FAISS avec embeddings Gemini (768 dimensions).
- **Recherche web de secours** quand les documents ne contiennent pas la réponse.
- **Citations précises** : chaque extrait est donné au modèle avec son fichier et sa page ; la réponse les cite entre parenthèses.
- **Vérification automatique** : un juge LLM, des règles de sécurité et un score ROUGE-L produisent un **score de confiance**. Une réponse invalide est régénérée une fois.
- **Abstention** : si aucune source n'est trouvée ou si la vérification rejette clairement la réponse, l'assistant refuse de répondre plutôt que d'inventer.

**Sécurité**
- **Détection d'urgence** (règles écrites à la main, sans LLM) : bandeau d'alerte et orientation vers les urgences.
- **Message de soutien** pour les propos de détresse.
- **Filtre de domaine** : les questions hors médecine reçoivent un refus poli, sans recherche ni génération.

**Multimodal**
- **Analyse de fichiers joints** : images, PDF, Word, Excel/CSV, texte (modèle de vision Gemini, avec plan B Groq pour les contenus textuels).
- **Graphe médical** (Neo4j, optionnel) affiché en schéma.
- **Tableaux** affichés en tableau.

**Conversation**
- **Questions de suivi** : « et pour les enfants ? » est réécrit en question autonome avant la recherche.
- **Traduire / résumer / simplifier** la réponse précédente (« traduis ça en arabe »), sans nouvelle recherche. Le texte arabe s'affiche de droite à gauche.

**Interface (style ChatGPT / Claude)**
- Barre latérale avec l'historique groupé par date, recherche, **renommer**, **supprimer** (avec confirmation).
- **Modifier** un message envoyé, **régénérer** une réponse, **copier**, bouton **Arrêter** pendant la génération.
- **Mode clair / sombre**, tiroir sur mobile, réponses affichées en continu (streaming).

---

## Architecture

```
Question (+ fichier éventuel)
        │
        ▼
 Sécurité ── urgence / détresse ──► alerte ou message de soutien
        │
        ▼
 Filtre de domaine ── hors sujet ──► refus poli
        │              ── « traduis / résume » ──► reformulation de la réponse précédente
        ▼
 Routeur ──► text  : TextAgent  (FAISS + web)
         ──► graph : GraphAgent (Neo4j)
         ──► image : ImageAgent (fichier joint)
        │
        ▼
 Génération en streaming (Groq, secours Gemini)
        │
        ▼
 Validation (juge LLM + règles + ROUGE-L) ──► score de confiance
        │        └─ invalide : une régénération, puis abstention si rejet net
        ▼
 Réponse + sources + confiance  (flux SSE vers l'interface)
```

---

## Structure du projet

```
SMA Multimodal RAG/
├── BACKEND/                    API FastAPI + agents
│   ├── api/main.py             Routes HTTP et flux SSE
│   ├── agents/
│   │   ├── RAG_agent.py        TextAgent : FAISS, pertinence, web, citations
│   │   ├── multimodal_agent.py ImageAgent : analyse des fichiers joints
│   │   ├── graph_agent.py      GraphAgent : graphe médical
│   │   └── validation_agent.py Juge, règles de sécurité, confiance
│   ├── core/
│   │   ├── pipeline.py         Orchestration complète d'une question
│   │   ├── router.py           Choix text / graph / image
│   │   ├── safety.py           Détection d'urgence
│   │   ├── sessions.py         Historique en mémoire (TTL)
│   │   └── output_parser.py    Texte / tableau / graphe
│   ├── database/
│   │   ├── vector_store.py     Index FAISS (par lots, avec reprise)
│   │   ├── faiss_index/        Index (généré)
│   │   └── graph_db.py         Connexion Neo4j
│   ├── models/                 Chargement des LLM et des embeddings
│   ├── retrievers/             Récupération texte et graphe
│   ├── scripts/
│   │   ├── ingest_data.py      Indexation des PDF / DOCX
│   │   └── ingest_graph.py     Remplissage du graphe
│   ├── utils/                  Logger, prompts, recherche web
│   ├── evaluation/             Score de confiance, évaluation RAGAS
│   ├── data/documents/         ← vos PDF et DOCX
│   └── config/settings.py      Configuration (lit le .env)
├── FRONTEND/                   Interface Next.js 15 / React 19
│   ├── app/                    page.js, layout.js, globals.css
│   ├── components/             Sidebar.js, Message.js
│   └── lib/                    sse.js (flux), store.js (conversations)
├── requirements.txt
└── README.md
```

`agents/orchestrator.py`, `agents/generator_agent.py` et `memory/` sont présents dans le dépôt mais ne sont pas utilisés par `core/pipeline.py`.

---

## Installation

**Prérequis** : Python 3.10 ou plus, Node.js 18.18 ou plus, et des clés API gratuites ([Groq](https://console.groq.com), [Google AI Studio](https://aistudio.google.com)).

**Backend** (PowerShell, depuis la racine du projet) :

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Frontend** :

```powershell
cd FRONTEND
npm install
```

---

## Configuration (.env)

Créez `BACKEND/.env` :

```env
# Clés API
GROQ_API_KEY=...
GEMINI_API_KEY=...
TAVILY_API_KEY=...            # optionnelle : recherche web

# Embeddings (changer l'un des deux impose de réindexer avec --rebuild)
EMBEDDING_MODEL=models/gemini-embedding-001
EMBEDDING_DIM=768

# Offre gratuite Gemini : rester sous ~100 textes par minute
EMBED_BATCH_SIZE=50
EMBED_PAUSE_S=35
```

Réglages facultatifs (valeurs par défaut entre parenthèses) :

| Variable | Rôle |
|---|---|
| `LLM_PROVIDERS` (`groq,gemini`) | Ordre des modèles de génération ; les suivants servent de secours |
| `JUDGE_PROVIDERS` (`gemini,groq`) | Modèles du juge de validation (de préférence ≠ générateur) |
| `GROQ_MODEL` (`openai/gpt-oss-120b`), `GEMINI_MODEL` (`gemini-flash-latest`) | Modèles utilisés |
| `VALIDATION_MIN_SCORE` (`6`) | Note minimale (sur 10) pour qu'une réponse soit valide |
| `ABSTAIN_BELOW_SCORE` (`4`) | Sous cette note, l'assistant s'abstient |
| `MAX_REGENERATIONS` (`1`) | Régénérations autorisées si la réponse est invalide |
| `RETRIEVAL_MAX_DISTANCE` (`1.0`) | Distance FAISS maximale ; plus bas = plus strict |
| `VISION_MODELS`, `VISION_TIMEOUT_S` (`40`), `VISION_BUDGET_S` (`90`) | Modèles et délais pour l'analyse de fichiers |
| `GENERATION_TEMPERATURE` (`0.1`), `LLM_TIMEOUT` (`30`) | Génération |
| `MAX_UPLOAD_MB` (`15`), `MAX_MESSAGE_CHARS` (`4000`), `SESSION_TTL_S` (`7200`) | Limites de l'API |
| `CORS_ORIGINS` (`http://localhost:3000,http://127.0.0.1:3000`) | Origines autorisées |
| `FAISS_PATH` (`database/faiss_index`) | Dossier de l'index |

Le graphe médical (Neo4j) est optionnel : sans lui, les demandes de graphe répondent qu'aucune donnée n'est disponible. Les variables de connexion sont lues dans `database/graph_db.py`.

Créez `FRONTEND/.env.local` si le backend n'est pas sur `http://localhost:8000` :

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

---

## Indexer les documents

Déposez vos PDF et DOCX dans `BACKEND/data/documents/`, puis, depuis `BACKEND` :

```powershell
python scripts/ingest_data.py                    # indexe ce qui ne l'est pas encore, du plus petit au plus gros
python scripts/ingest_data.py --only liste-nat   # un seul fichier (nom contenant ce texte)
python scripts/ingest_data.py --rebuild          # efface tout et recommence
```

- Les fichiers sont découpés en extraits de 1000 caractères, indexés par groupes de 50, avec une pause entre les envois.
- Un fichier journal (`database/faiss_index_manifest.json`) mémorise la progression : **si l'indexation s'arrête, relancez la même commande, elle reprend où elle s'est arrêtée.**
- N'utilisez `--rebuild` que si vous changez de modèle d'embeddings ou modifiez un fichier déjà indexé : il efface l'index.
- **Redémarrez le backend** après une indexation pour qu'il recharge l'index.

**Quotas de l'offre gratuite Gemini.** Chaque extrait compte pour une requête d'embedding : environ 100 par minute et un plafond journalier de l'ordre du millier (ces limites peuvent changer). Un gros document peut demander plusieurs jours. Le script s'arrête avec « Quota JOURNALIER atteint » : relancez-le le lendemain. Pour supprimer cette contrainte, activez la facturation sur Google AI Studio, ou passez à des embeddings locaux (réindexation complète nécessaire).

---

## Lancer l'application

Dans deux terminaux.

**Backend** (depuis `BACKEND`, environnement virtuel activé) :

```powershell
uvicorn api.main:app --reload --port 8000
```

Vérification : <http://localhost:8000/api/health>

**Frontend** (depuis `FRONTEND`) :

```powershell
npm run dev
```

Ouvrez <http://localhost:3000>.

Les conversations sont enregistrées **dans le navigateur** (`localStorage`) : elles ne se retrouvent ni sur un autre appareil ni dans un autre navigateur. Le serveur garde le contexte d'une session pendant 2 heures ; au-delà, l'interface renvoie les derniers échanges pour le restaurer.

---

## API

| Méthode | Route | Rôle |
|---|---|---|
| `GET` | `/api/health` | État du serveur, des modèles et de l'index |
| `POST` | `/api/chat` | Pose une question, réponse en flux SSE |
| `DELETE` | `/api/sessions/{id}` | Efface le contexte d'une session |

`POST /api/chat` (`multipart/form-data`) :

| Champ | Description |
|---|---|
| `message` | Question (4000 caractères maximum) |
| `session_id` | Identifiant de conversation (8 à 64 caractères : lettres, chiffres, `_`, `-`) |
| `route` | `auto` (défaut), `text`, `graph` ou `image` |
| `history` | Facultatif : JSON `[{"q": "...", "a": "..."}]`, utilisé si la session serveur est vide |
| `file` | Facultatif : png, jpg, jpeg, webp, gif, bmp, pdf, docx, xlsx, csv, txt, md (15 Mo maximum) |

Événements SSE : `meta`, `status`, `alert`, `warning`, `sources`, `token`, `reset`, `result`, `error`, `done`.

---

## Limites connues

- **Tableaux dans les PDF** : l'extraction de texte (`PyPDF`) peut mélanger les colonnes d'un tableau complexe (posologies, niveaux de soins). Vérifiez la réponse avec le document cité.
- **PDF scannés** : sans texte extractible, ils sont ignorés à l'indexation (pas d'OCR).
- **Sources web** : elles n'ont pas de numéro de page et sont moins fiables que vos documents.
- **Images médicales** : l'assistant décrit uniquement ce qu'il voit et ne conclut jamais qu'un examen est normal ou anormal. La confiance est plafonnée à « Moyenne ».
- **Traductions** : une traduction automatique peut déformer un chiffre ; vérifiez les doses avec la réponse d'origine.
- **Conversations** : stockées dans le navigateur, limitées à 100.

---

## Dépannage

| Symptôme | Cause probable | Solution |
|---|---|---|
| `429 RESOURCE_EXHAUSTED` à l'indexation | Quota Gemini | Voir « Quotas » ci-dessus ; relancer plus tard |
| `Quota JOURNALIER atteint` | Plafond du jour atteint | Relancer le lendemain, ou activer la facturation |
| `503 UNAVAILABLE` sur un fichier joint | Modèle Gemini surchargé | Réessayer ; les contenus textuels basculent sur Groq |
| « Index FAISS absent ou incompatible » | Index non construit, ou modèle d'embeddings changé | `python scripts/ingest_data.py --rebuild` |
| Réponses qui ne citent pas un document récent | Backend non redémarré | Redémarrer `uvicorn` |
| `unrecognized arguments: --only` | Ancienne version de `ingest_data.py` | Remplacer par la version à jour |
| L'interface n'atteint pas le serveur | URL ou CORS | Vérifier `NEXT_PUBLIC_API_URL` et `CORS_ORIGINS` |

---

## Licence

À définir.