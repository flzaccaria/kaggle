"""Genera il notebook Kaggle MuseumSCAT (EDA + baseline) usando nbformat."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(src):
    cells.append(nbf.v4.new_markdown_cell(src))

def code(src):
    cells.append(nbf.v4.new_code_cell(src))

# ----------------------------------------------------------------------------
md(r"""# 🪲 MuseumSCAT — Notebook 01: EDA + Baseline

> *"Prima si guarda il nemico, poi si carica il fucile."* — proverbio da Grandmaster
> inventato cinque minuti fa.

Benvenuto nella **collezione di scarabei stercorari danesi** del Natural History
Museum of Denmark. Il nostro compito **non** è riconoscere gli scarabei: è
**trascrivere due campi** scritti sui cartellini di ogni esemplare, esattamente
come li ha vergati un entomologo tra il tardo '800 e oggi.

I due campi da predire per ogni immagine:

| Campo | Cosa contiene | Esempi |
|---|---|---|
| `verbatimDate` | la data di raccolta, *verbatim* | `27.IV.2022`, `18-7 1965`, `IV.91`, `MISSING` |
| `verbatimLocality` | il luogo di raccolta, *verbatim* | `Dyrehaven`, `Kb \| Dyrehaven`, `MISSING` |

**I numeri che contano (e spaventano):**
- **200** esempi etichettati (`train.csv`) contro **~3.300** immagini di test. Pochissimo
  training → niente OCR addestrato da zero: si va di modelli **pre-addestrati** + few-shot.
- Metrica **doppia**: **NED** (Normalized Edit Distance) **+ AURC**. Non basta trascrivere
  bene: devi anche dare un **confidence score** onesto per ogni predizione. Il modello
  deve "sapere quello che non sa".

**Piano di questo notebook:**
1. Trovare e caricare i dati (regola 1: non fidarsi mai dei path).
2. EDA sulle etichette: formati delle date, delle località, valori mancanti, cartellini multipli.
3. Guardare le immagini vere.
4. Reimplementare la metrica NED in locale → così possiamo validarci **senza** sprecare submission.
5. Una **baseline naive** che produce un `submission.csv` valido: numero sulla leaderboard, subito.
6. Cosa ci ha insegnato l'EDA → e quale modello scegliere nel Notebook 02.

Andiamo. 🚀""")

# ----------------------------------------------------------------------------
md(r"""## 1. Setup e caricamento dati

Regola numero 1 del mestiere: **non scrivere mai a mano il path dei dati.** Kaggle
monta la competizione sotto `/kaggle/input/<slug>/`, ma lo slug può cambiare. Cerchiamolo.""")

code(r"""import os, glob, re, itertools, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from collections import Counter

pd.set_option('display.max_colwidth', 120)
random.seed(42); np.random.seed(42)

# Cerca la cartella della competizione sotto /kaggle/input senza hardcodare lo slug.
CANDIDATES = glob.glob('/kaggle/input/*museumscat*') + glob.glob('/kaggle/input/*')
DATA_DIR = None
for c in CANDIDATES:
    if os.path.exists(os.path.join(c, 'train.csv')):
        DATA_DIR = c
        break
if DATA_DIR is None:
    # fallback: prima cartella disponibile (utile se giri in locale)
    DATA_DIR = CANDIDATES[0] if CANDIDATES else '.'

print('DATA_DIR =', DATA_DIR)
print('Contenuto:')
for f in sorted(os.listdir(DATA_DIR)):
    print('  ', f)""")

code(r"""# Individua la cartella delle immagini (di solito 'images')
IMG_DIR = None
for cand in ['images', 'Images', 'train_images', 'imgs']:
    p = os.path.join(DATA_DIR, cand)
    if os.path.isdir(p):
        IMG_DIR = p
        break
if IMG_DIR is None:
    # a volte le immagini stanno nella root del dataset
    IMG_DIR = DATA_DIR
print('IMG_DIR =', IMG_DIR)
print('Numero file immagine (jpeg):', len(glob.glob(os.path.join(IMG_DIR, '**', '*.jpeg'), recursive=True)))""")

code(r"""train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
test  = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))

print('train:', train.shape)
print('test :', test.shape)
print('\nColonne train:', list(train.columns))
print('Colonne test :', list(test.columns))
train.head()""")

md(r"""### 1.1 Un piccolo mistero: `NULL` o `MISSING`?

La documentazione della gara si contraddice: in un punto dice di segnare i valori
mancanti come `"NULL"`, in un altro come `"MISSING"`. **Non indovinare: guarda i dati.**
Questo è un riflesso che devi allenare — la ground truth è l'unica fonte di verità.""")

code(r"""for col in ['verbatimDate', 'verbatimLocality']:
    print(f'--- {col} ---')
    vc = train[col].astype(str).value_counts()
    # mostra i token "sospetti" di mancanza
    for tok in ['MISSING', 'NULL', 'nan', 'None', '']:
        if tok in vc.index:
            print(f'  token "{tok}": {vc[tok]} occorrenze')
    print('  valori nulli pandas (NaN):', train[col].isna().sum())
    print()

# Definiamo il token di "mancante" REALE trovato nei dati (non quello della doc)
MISSING_TOKEN = 'MISSING'
for tok in ['MISSING', 'NULL']:
    if (train[['verbatimDate','verbatimLocality']].astype(str) == tok).any().any():
        MISSING_TOKEN = tok
        break
print('>>> Token di mancanza usato nella ground truth:', repr(MISSING_TOKEN))""")

# ----------------------------------------------------------------------------
md(r"""## 2. EDA sulle etichette

Prima ancora delle immagini, i **testi** ci raccontano moltissimo: quanto sono lunghi,
quanto spesso mancano, quali formati assumono. Da qui capiremo *quanto è difficile* il
problema e *dove* un modello sbaglierà.""")

code(r"""# Frequenza dei valori mancanti in ciascun campo
for col in ['verbatimDate', 'verbatimLocality']:
    s = train[col].astype(str)
    n_missing = (s == MISSING_TOKEN).sum()
    print(f'{col:20s}  mancanti: {n_missing:3d} / {len(s)}  ({100*n_missing/len(s):.1f}%)')""")

code(r"""# Distribuzione della lunghezza (in caratteri) dei campi NON mancanti
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, col in zip(axes, ['verbatimDate', 'verbatimLocality']):
    s = train[col].astype(str)
    lengths = s[s != MISSING_TOKEN].str.len()
    ax.hist(lengths, bins=20, color='#008ABC', edgecolor='white')
    ax.set_title(f'Lunghezza di {col}')
    ax.set_xlabel('caratteri'); ax.set_ylabel('conteggio')
plt.tight_layout(); plt.show()""")

md(r"""### 2.1 Che aspetto hanno le date?

La doc ci avvisa di un delirio di formati: numeri romani per il mese (`IV` = aprile),
date parziali (`IV.91`), separatori misti. Classifichiamo automaticamente ogni data in
una "famiglia" di formato con qualche regex. Non serve la perfezione: serve capire la
distribuzione.""")

code(r"""ROMAN = r'(?:I{1,3}|IV|V|VI{0,3}|IX|XI{0,2}|X)'

def date_family(s):
    s = str(s).strip()
    if s == MISSING_TOKEN:
        return 'missing'
    if re.search(ROMAN, s):                       # mese in numeri romani
        return 'roman_numeral'
    if re.search(r'\d{1,2}[./\- ]\d{1,2}[./\- ]\d{2,4}', s):  # gg sep mm sep aaaa
        return 'numeric_full'
    if re.search(r'\d{1,2}[./ ]\d{2,4}$', s):     # parziale tipo mese/anno
        return 'partial'
    if re.search(r'\d', s):
        return 'other_numeric'
    return 'no_digits'

fam = train['verbatimDate'].map(date_family)
print(fam.value_counts())
fam.value_counts().plot.bar(color='#008ABC', figsize=(8,3), title='Famiglie di formato data')
plt.tight_layout(); plt.show()""")

md(r"""### 2.2 Cartellini multipli: il trucco della pipe `|`

Alcuni esemplari hanno le informazioni spalmate su **più cartellini** fisici sullo
stesso spillo. Nella ground truth vengono uniti con `|`. La metrica prova **tutti gli
ordinamenti** dei cartellini e poi rimuove le pipe: quindi puoi usarle generosamente
quando l'ordine di lettura è ambiguo. Vediamo quanto sono frequenti.""")

code(r"""for col in ['verbatimDate', 'verbatimLocality']:
    s = train[col].astype(str)
    has_pipe = s.str.contains(r'\|')
    print(f'{col:20s}  con pipe (multi-card): {has_pipe.sum()} ({100*has_pipe.mean():.1f}%)')

print('\nEsempi multi-card:')
mask = train['verbatimLocality'].astype(str).str.contains(r'\|')
display(train.loc[mask, ['image_file','verbatimDate','verbatimLocality']].head(8))""")

md(r"""### 2.3 Località: coordinate, gerarchie e le trappole (`Dania`, sterco di mucca)

Le località vanno dal semplice `Dyrehaven` a stringhe con coordinate GPS. Ricorda le
due trappole della doc: **`Dania`** (nome della collezione, non un luogo) e **`i kogøding`**
(= "nello sterco di mucca", non un luogo) **non** compaiono nella ground truth. Controlliamo.""")

code(r"""loc = train['verbatimLocality'].astype(str)

has_coord = loc.str.contains(r'\d+\.\d+\s*[NnSs].*\d+\.\d+\s*[EeWw]', regex=True)
print('Località con coordinate GPS:', has_coord.sum())

# Verifica trappole: NON dovrebbero comparire come località
for trap in ['Dania', 'kogøding', 'kogjød', 'Coll.', 'det.', 'Tilg.']:
    n = loc.str.contains(re.escape(trap), case=False).sum()
    print(f'  contiene "{trap}": {n}')

print('\nLocalità più frequenti (non mancanti):')
print(loc[loc != MISSING_TOKEN].value_counts().head(12))""")

# ----------------------------------------------------------------------------
md(r"""## 3. Guardiamo le immagini vere

I numeri sono belli ma il problema è **visivo**. Mostriamo alcuni esemplari con la loro
ground truth accanto: è qui che capisci quanto è dura la scrittura a mano storica, e
perché un semplice OCR soffrirà.""")

code(r"""def load_image(image_file):
    for base in [IMG_DIR, DATA_DIR]:
        p = os.path.join(base, image_file)
        if os.path.exists(p):
            return Image.open(p).convert('RGB')
    hits = glob.glob(os.path.join(IMG_DIR, '**', image_file), recursive=True)
    return Image.open(hits[0]).convert('RGB') if hits else None

sample = train.sample(min(6, len(train)), random_state=1).reset_index(drop=True)
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for ax, (_, row) in zip(axes.ravel(), sample.iterrows()):
    img = load_image(row['image_file'])
    if img is not None:
        ax.imshow(img)
    ax.set_title(f"date: {row['verbatimDate']}\nloc: {row['verbatimLocality']}", fontsize=9)
    ax.axis('off')
plt.tight_layout(); plt.show()""")

# ----------------------------------------------------------------------------
md(r"""## 4. La metrica in casa nostra: NED (e un assaggio di AURC)

**Questa è la parte più importante di tutto il notebook.** Su Kaggle hai poche submission
al giorno: se ti validi solo sulla leaderboard, sei cieco. Reimplementiamo la metrica in
locale, così possiamo misurare *qualsiasi* idea sui 200 esempi di train prima di sprecare
un tentativo.

La **NED** (Normalized Edit Distance) tra predizione e verità:
- `0.0` = match perfetto, `1.0` = completamente sbagliato;
- **case-insensitive**;
- per le **date**, la punteggiatura è normalizzata (`.  ,  -  ·  spazio` sono equivalenti);
- gestisce le **pipe**: prova tutti gli ordinamenti dei cartellini e tiene il migliore.

> ⚠️ È la *nostra* approssimazione della metrica ufficiale: utile per il CV locale, non
> identica al decimale. Ma per confrontare due idee è più che sufficiente.""")

code(r"""def levenshtein(a, b):
    "Distanza di edit classica (pura Python: le stringhe sono corte, va benissimo)."
    if a == b: return 0
    if not a: return len(b)
    if not b: return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j-1] + 1, prev[j-1] + (ca != cb)))
        prev = cur
    return prev[-1]

def normalize(s, is_date):
    s = str(s).strip().lower()
    if is_date:
        s = re.sub(r'[\.,\-·/\s]+', ' ', s).strip()  # unifica separatori di data
    return s

def ned_single(pred, truth, is_date):
    "NED con gestione pipe: prova ogni ordinamento dei cartellini della verità."
    p_cards = [normalize(x, is_date) for x in str(pred).split('|')]
    t_cards = [normalize(x, is_date) for x in str(truth).split('|')]
    best = 1.0
    # prova tutte le permutazioni dei cartellini di verità contro quelli predetti
    for perm in itertools.permutations(t_cards):
        p_join = ' '.join(p_cards)
        t_join = ' '.join(perm)
        d = levenshtein(p_join, t_join)
        m = max(len(p_join), len(t_join), 1)
        best = min(best, d / m)
    return best

# sanity check
assert ned_single('27.5.2022', '27-5-2022', is_date=True) == 0.0
assert ned_single('Dyrehaven', 'dyrehaven', is_date=False) == 0.0
print('NED sanity check: OK ✅')""")

code(r"""def score_df(pred_df, truth_df):
    "NED medio sui due campi, come proxy dello score (piu basso = meglio)."
    m = truth_df.merge(pred_df, on='image_file', suffixes=('_true', '_pred'))
    date_ned = m.apply(lambda r: ned_single(r['verbatimDate_pred'],  r['verbatimDate_true'],  True),  axis=1)
    loc_ned  = m.apply(lambda r: ned_single(r['verbatimLocality_pred'], r['verbatimLocality_true'], False), axis=1)
    return date_ned.mean(), loc_ned.mean(), (date_ned.mean() + loc_ned.mean()) / 2""")

md(r"""### 4.1 Un assaggio di AURC (perché la confidence conta)

L'**AURC** ordina le predizioni per confidence decrescente e misura la NED media man mano
che "copri" sempre più predizioni. L'idea: un buon modello è **sicuro quando ha ragione**
e **incerto quando sbaglia**. Alta confidence su risposte sbagliate = punizione severa.

Non replichiamo la formula ufficiale al decimale, ma implementiamo la logica per capirla.""")

code(r"""def aurc(neds, confidences):
    "Area under risk-coverage curve (approssimazione). neds e confidences: array 1D."
    order = np.argsort(-np.asarray(confidences))      # confidence alta prima
    neds_sorted = np.asarray(neds)[order]
    coverages = np.arange(1, len(neds_sorted) + 1)
    risk = np.cumsum(neds_sorted) / coverages          # NED media cumulata
    return risk.mean()                                  # area (media) sotto la curva

# Intuizione: con la STESSA qualita di predizioni, ordinare bene la confidence abbassa l'AURC.
demo_neds = np.array([0.0, 0.0, 1.0, 1.0])
print('Confidence perfetta (giuste prima):', round(aurc(demo_neds, [1,1,0,0]), 3))
print('Confidence pessima  (sbagliate prima):', round(aurc(demo_neds, [0,0,1,1]), 3))""")

# ----------------------------------------------------------------------------
md(r"""## 5. Baseline naive → prima submission

Obiettivo di una baseline: **non vincere**, ma (a) validare la pipeline end-to-end e
(b) mettere un numero sulla leaderboard come riferimento. Confrontiamo due strategie
banali sui 200 esempi di train con la *nostra* NED:

1. **Tutto `MISSING`** — sorprendentemente competitiva se molti campi sono davvero vuoti.
2. **Valore più frequente** — la "moda" di date e località.

Poi scegliamo la migliore e generiamo il `submission.csv`.""")

code(r"""# Split interno per una valutazione onesta (train piccolo: teniamo l'80/20)
tr = train.sample(frac=0.8, random_state=0)
va = train.drop(tr.index)
truth_va = va[['image_file','verbatimDate','verbatimLocality']].copy()

most_common_date = tr['verbatimDate'].astype(str).mode()[0]
most_common_loc  = tr['verbatimLocality'].astype(str).mode()[0]
print('Data piu frequente     :', repr(most_common_date))
print('Localita piu frequente :', repr(most_common_loc))

def make_pred(df, date_val, loc_val):
    return pd.DataFrame({
        'image_file': df['image_file'].values,
        'verbatimDate': date_val,
        'verbatimLocality': loc_val,
    })

strategies = {
    'tutto MISSING'    : make_pred(va, MISSING_TOKEN, MISSING_TOKEN),
    'valore piu freq.' : make_pred(va, most_common_date, most_common_loc),
}
print('\n%-18s  %8s  %8s  %8s' % ('strategia','NED date','NED loc','NED avg'))
scores = {}
for name, pred in strategies.items():
    d, l, a = score_df(pred, truth_va)
    scores[name] = a
    print('%-18s  %8.3f  %8.3f  %8.3f' % (name, d, l, a))

best_strategy = min(scores, key=scores.get)
print('\n>>> Baseline scelta:', best_strategy)""")

md(r"""### 5.1 Costruzione del `submission.csv`

Il formato di submission atteso (5 colonne):

`image_file, verbatimDate, verbatimDate_confidence, verbatimLocality, verbatimLocality_confidence`

Due note pratiche:
- Kaggle **non accetta celle vuote**: i mancanti vanno scritti col token (`MISSING`).
- La **confidence** serve all'AURC. Per una baseline uniforme la mettiamo costante (es. `0.5`):
  non aiuta il ranking, ma è un valore valido. Il vero lavoro sulla confidence arriva col modello.""")

code(r"""# Applica la strategia migliore a TUTTO il test set
if best_strategy == 'tutto MISSING':
    date_val, loc_val = MISSING_TOKEN, MISSING_TOKEN
else:
    date_val = train['verbatimDate'].astype(str).mode()[0]
    loc_val  = train['verbatimLocality'].astype(str).mode()[0]

submission = pd.DataFrame({
    'image_file': test['image_file'].values,
    'verbatimDate': date_val,
    'verbatimDate_confidence': 0.5,
    'verbatimLocality': loc_val,
    'verbatimLocality_confidence': 0.5,
})

# Controlli di validità: niente NaN, niente celle vuote, tutte le immagini di test presenti
assert submission['image_file'].nunique() == test['image_file'].nunique()
assert not submission.isna().any().any(), 'ci sono NaN!'
assert (submission[['verbatimDate','verbatimLocality']] != '').all().all(), 'celle vuote!'

submission.to_csv('submission.csv', index=False)
print('submission.csv scritto:', submission.shape)
submission.head()""")

# ----------------------------------------------------------------------------
md(r"""## 6. Cosa ci ha insegnato l'EDA (e come scegliamo il modello)

Riepilogo da portare al Notebook 02:

- **Pochissimo training (200):** il fine-tuning pesante è fuori discussione. La strada è un
  modello **pre-addestrato** usato in few-shot / zero-shot.
- **Scrittura a mano storica + formati caotici:** l'OCR classico (Tesseract) inciamperà sul
  corsivo e sui numeri romani. Un **VLM** (vision-language model) che "ragiona" sull'immagine
  ed estrae direttamente i due campi è molto più promettente — e può gestire le regole
  (escludi `Coll.`/`det.`, ignora `Dania` e lo sterco di mucca) via **prompt**.
- **AURC pesa:** dobbiamo produrre una confidence sensata. Idee: log-prob del modello,
  auto-consistenza (più campionamenti → accordo = confidence), o incrociare date/località.
- **Pipe multi-card:** possiamo usarle generosamente quando l'ordine è ambiguo — la metrica
  ci perdona.

**Prossima puntata (Notebook 02):** scegliamo tra un VLM open sul GPU Kaggle (es. Qwen2-VL)
e l'OCR classico come contro-prova, e ci costruiamo una strategia di confidence per l'AURC.

> Compito per te: apri 10 immagini a caso e prova a trascriverle *a mano*. Se fai fatica tu,
> capisci cosa stiamo chiedendo al modello. 🪲""")

# ----------------------------------------------------------------------------
nb['cells'] = cells
nb['metadata'] = {
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python'},
}
with open('museumscat_01_eda_baseline.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print('Notebook scritto: museumscat_01_eda_baseline.ipynb  (%d celle)' % len(cells))
