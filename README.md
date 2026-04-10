# CMPE 597 Sp. Tp. Deep Learning — Spring 2026 Term Project

## 1 Data

The [MEMECAP dataset](https://github.com/eujhwang/meme-cap) is a collection of 6,384 memes with their captions, designed to study how well vision and language models can recognize and interpret visual metaphors. The dataset is split into a training set of 5,823 memes and a test set of 559 memes. You can randomly sample a validation set from the training set for hyperparameter tuning.

You will find JSON files for training and test splits in the GitHub repository. Each entry in the dataset includes a title, a meme caption, a literal image caption, visual metaphors, and additional metadata. Meme title is its post title. The memes were scraped from Reddit (please read the [paper](https://aclanthology.org/2023.emnlp-main.89/) for more information about data acquisition[^1]). The meme caption is a sentence that conveys the meme's intended meaning. The image caption is the objective description of what is physically appearing in the image. Finally, the visual metaphors are some words that map visual elements to their underlying meanings.

An example from the training set:

```json
{
  "category": "memes",
  "img_captions": ["Spiderman climbs a tall buliding."],
  "meme_captions": ["Meme poster is trying to convey that climate change protesters don't always protest in a way that makes sense."],
  "title": "Had to do it.",
  "url": "https://i.redd.it/plchbf34saw91.jpg",
  "img_fname": "memes_yel3vo.png",
  "metaphors": [{"meaning": "climate change protesters", "metaphor": "Spiderman"}],
  "post_id": "yel3vo",
  "image": "/archive/projects/memecap/memes/memes_yel3vo.png"
}
```

---

## 2 Tasks

### 2.1 Cross-modal retrieval (30 pts)

The goal of this task is to retrieve the meme caption of a given meme and its title. We need meme and caption representations that are aligned in a latent space, so that the similarity between the meme and its corresponding caption is maximized relative to the similarity between the meme and unrelated captions. The architecture that you will design should comprise an image and a text encoder.

Please do the following while considering two types of input. **Type 1** corresponds to the meme's image, and **type 2** will be the meme's image and its title. In type 1, your query is a single embedding of the input image. In type 2, you will need to fuse the meme's image and its title representations into a single embedding. You are free to design any fusion strategy you want in type 2.

**(a)** Search the literature for the appropriate evaluation strategy for the cross-modal retrieval task. Your evaluation framework should report recall@1 and recall@5. You are free to include other related metrics.

**(b)** Find one or more pretrained architecture(s) (e.g., CLIP) trained for cross-modal retrieval. Evaluate their zero-shot performances.

**(c)** Design a custom architecture and train it from scratch with the MemeCap dataset. Please determine which loss function(s) to use. You are free to add regularization if necessary. Compare the performance of the custom architecture with that of the pretrained ones.

**(d)** Finetune one of the pretrained architectures. Explore efficient finetuning techniques, such as low-rank adaptation (LoRA)[^2]. You are free to determine the finetuning technique. Compare the finetuning performance with zero-shot and your custom model's performance.

Repeat the parts b–d for both type 1 and type 2 input, and compare their performance. At any stage, you can propose incorporating the image caption to improve retrieval performance.

---

### 2.2 Literal vs. metaphorical caption classification (30 pts)

In this task, we need to design a classifier that distinguishes image-caption pairs in which the caption literally describes the image from those in which it is a metaphorical interpretation of the image. Thus, this task corresponds to a two-class classification problem. We need to create and annotate the pairs from the MemeCap dataset. For instance, a meme image and its meme caption can be annotated as the positive class (metaphorical interpretation), while a meme image and its image caption will be considered as the negative class (literal).

**(a)** Implement the evaluation framework for the task. Include classification evaluation metrics in your framework.

**(b)** Please design an architecture that will fuse the image and caption embeddings into a single representation and classify them. You can utilize pretrained architectures to extract initial embeddings. You may consider using the embeddings you obtained in the cross-modal retrieval task.

**(c)** Please compare the performances of the architectures you tried in part b. Create baselines for yourself and try to improve them. If you have multiple components you changed to observe performance differences, such as changes in the architecture, activation functions, loss functions, regularization techniques, etc., conduct an ablation study.

---

### 2.3 Meme sentiment classification (30 pts)

This task requires designing a classifier that takes the meme image and predicts its emotion. Therefore, the task is posed as a multiclass classification problem. You may again consider multimodal inputs, e.g., taking the image and the image caption.

**(a)** First, you need to use a pretrained sentiment analysis model to map each meme's meme captions to an emotion. You can determine the number of categories. Please report if there is any class imbalance after your annotation step. Manually check the sentiment labels of a random subset for potential label noise.

**(b)** Using pretrained CLIP, extract meme caption and meme image embeddings. By training separate MLP classifiers, investigate whether the pretrained embeddings carry any sentiment information. Report the classification performances of the MLP classifier trained with only image embeddings and the MLP trained with only the meme caption embeddings.

**(c)** Design a custom architecture for the meme sentiment classification task. Output will be the class labels obtained in part a. You can try different approaches regarding the input. Report your classifier's performance and compare it with the performance in part b.

---

## 3 Project Calendar

| Date        | Expected Progress |
|-------------|-------------------|
| 24 February | Project posted |
| 3 March     | — |
| 10 March    | Task 2.1a,b — zero-shot performances (2 pts) |
| 17 March    | — |
| 24 March    | Project progress update I — custom architecture and finetuning 2.1c,d (2 pts) |
| 31 March    | — |
| 7 April     | Task 2.2a,b,c — fusion strategies (2 pts) |
| 14 April    | — |
| 21 April    | Spring Break |
| 28 April    | Project progress update II — Task 2.3a (2 pts) |
| 5 May       | — |
| 12 May      | Final project oral evaluation (2 pts) |

- All the groups should be present during in-class project discussions.
- Failing to show up or to meet the expected level of progress will cost 2 points. (Each milestone is worth 2 points. 10 points for reaching all five milestones.)
- Each group should create a private GitHub repository and add the instructor (GitHub username: **baytasin**). Before each in-class discussion, the repo should be organized to present the recent progress. Please show open and closed issues, a notebook with code and results, links to other pages that summarize your results, etc.
- Please do not forget to add the detailed references to your repo's README.

---

[^1]: Hwang, EunJeong, and Vered Shwartz. "Memecap: A dataset for captioning and interpreting memes." In *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing*, pp. 1433–1445. 2023.

[^2]: Hu et al. "LoRA: Low-Rank Adaptation of Large Language Models." https://openreview.net/pdf?id=nZeVKeeFYf9
