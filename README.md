# Online Gambling Detection for YouTube Comments

A text-classification project that detects online gambling promotion in Indonesian-language YouTube comments. Built as a research project at Universitas Multimedia Nusantara, where I was research lead and developer.

> This project studies gambling *spam* so it can be filtered. It does not promote gambling.

## Results

The full experiment is documented in the [research report](https://drive.google.com/file/d/10lEqDOEMXXhWK_2RYkgYCS-MwVk5bX-U/view?usp=drive_link):

- **97.6% accuracy** with a TF-IDF + Linear SVM pipeline
- Compared Multinomial Naive Bayes, Linear SVM, and an ensemble voting classifier, tuned with `GridSearchCV`
- Balanced the classes with SMOTE to improve minority-class detection
- Cleaned noisy social-media text: slang, emojis, code-switching, and corrupted Unicode

## Repository contents

```
src/
  preprocessing.py   load, clean, and feature-engineer the data
  model.py           train/test split and model training
  evaluation.py      accuracy and classification report
data/README.md       where to get the dataset
```

`src/` currently holds a simplified baseline pipeline (a Random Forest classifier) that shows the project structure. The TF-IDF + Linear SVM experiment and its results are in the report linked above.

## Dataset

[Judionline](https://www.kaggle.com/datasets/yaemico/judionline) on Kaggle (author: yaemico): Indonesian comments labelled gambling / non-gambling. The dataset is not included here. See [`data/README.md`](data/README.md) to reproduce.

## Tech stack

Python · scikit-learn · pandas · NumPy · imbalanced-learn · matplotlib · seaborn

## License

[MIT](LICENSE)
