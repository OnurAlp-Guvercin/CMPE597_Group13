# Task 2.3(a) — Sentiment Annotation Report

## Approach

To assign emotion labels to the dataset, we used the pretrained model `j-hartmann/emotion-english-distilroberta-base`. For each meme, we fed the first `meme_captions` entry into the model and obtained one of 7 emotion labels: `anger`, `disgust`, `fear`, `joy`, `neutral`, `sadness`, `surprise`. We chose this model because it covers the most common emotional tones out of the box and requires no additional fine-tuning.

---

## Class Distribution

### Train split (5,823 samples)

| Emotion  | Count |    % |
|----------|------:|-----:|
| neutral  | 3,200 | 55.0% |
| disgust  | 1,091 | 18.7% |
| sadness  |   437 |  7.5% |
| anger    |   374 |  6.4% |
| joy      |   314 |  5.4% |
| fear     |   219 |  3.8% |
| surprise |   188 |  3.2% |

- Imbalance ratio (max / min): **17.02×** (neutral vs. surprise)
- Shannon entropy: **2.027 / 2.807** (uniform = 2.807)

### Test split (559 samples)

| Emotion  | Count |    % |
|----------|------:|-----:|
| neutral  |   293 | 52.4% |
| disgust  |   104 | 18.6% |
| sadness  |    47 |  8.4% |
| anger    |    45 |  8.1% |
| joy      |    37 |  6.6% |
| surprise |    17 |  3.0% |
| fear     |    16 |  2.9% |

- Imbalance ratio (max / min): **18.31×** (neutral vs. fear)
- Shannon entropy: **2.092 / 2.807** (uniform = 2.807)

---

## Class Imbalance

Both splits show the same pattern: `neutral` dominates at roughly 52–55%, while `fear` and `surprise` together account for only about 6%. The imbalance ratios (~17–18×) are well above the 5× warning threshold, and the Shannon entropy sits around 2.03–2.09 out of a theoretical maximum of 2.81 — meaning the distribution is noticeably far from uniform.

This is expected for meme data. Meme captions are often indirect and ironic, which makes it hard for the model to commit to a specific emotion, so it defaults to `neutral`. The practical risk is that a classifier trained on these labels could achieve ~52% accuracy just by always predicting `neutral`. To counter this, we will need either class-weighted loss or resampling of minority classes (fear, surprise, joy) before training in tasks 2.3(b) and 2.3(c).

---

## Manual Label Check

We randomly sampled 50 examples from the training set (seed=42) and inspected whether the assigned labels were plausible given the caption text. Most high-confidence predictions (≥ 0.7) looked reasonable. The problematic cases were concentrated among low-confidence predictions, as expected.

**Clear mislabels or questionable assignments:**

- **[12]** `disgust` (conf **0.383**): *"Meme poster is trying to form a person's name from the representations"* — the caption describes a word puzzle; `neutral` is the obvious correct label. The model had no basis for `disgust` here.
- **[15]** `sadness` (conf **0.308**): *"Meme poster will respond to emails saying not to reply and sowing chaos."* — this reads as mischievous or playful, not sad. Lowest confidence in the sample; clearly a guess.
- **[22]** `sadness` (conf **0.357**): *"Meme poster is tired of the world screwing them over and gives up."* — `sadness` is arguably defensible, but the frustration/resignation tone could equally be `anger`. Low confidence reflects the model's uncertainty.
- **[10]** `anger` (conf **0.511**): *"The meme poster promotes Snickers that it could satisfy hunger and get rid of being anger by hunger."* — this is a positive product endorsement, not an expression of anger. `neutral` or `joy` would fit better.
- **[19]** `anger` (conf **0.513**): *"Meme poster is trying to convey that the World Cup winning of western Europe are united now but had centuries of war."* — a historical/political observation; `neutral` is more appropriate.
- **[03]** `neutral` (conf **0.417**): *"Meme poster feels old because their McDonald's toy from four years ago is now a meme."* — the caption explicitly describes feeling old, which leans toward `sadness`. Low confidence here is a sign the model was uncertain between the two.

**General observations:**

- Samples with confidence below ~0.50 were consistently unreliable — roughly 6 of the 50 reviewed samples fell in this range, and most of them were wrong or debatable.
- High-confidence predictions (≥ 0.85) were almost always plausible, even when the caption was indirect.
- Sarcasm and irony in meme captions are a systematic source of error: the model reads the surface phrasing rather than the intended tone.

**Estimated mislabel rate from this sample:** roughly 8–12%.

One option to explore before training is dropping samples with `sentiment_score < 0.5`, which would remove the noisiest labels at the cost of losing a small fraction of training data.
