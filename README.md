# Hit Song Prediction Using Spotify Data

This Machine Learning mini-project is inspired by the Stanford CS229 project **“HitPredict: Predicting Hit Songs Using Spotify Data.”**

## Problem Statement

The goal of this project is to predict whether a song is a **HIT (1)** or **NON-HIT (0)** using Spotify audio features.

### Methodological Definition

A **NON-HIT** means that the song was not matched with the Billboard dataset used in this project. It does **not** mean that the song was objectively unsuccessful or unpopular.

## Data Sources

### Billboard Hot 100 Data

The Billboard Hot 100 chart data was obtained from:

https://github.com/utdata/rwd-billboard-data/blob/main/data-out/hot-100-current.csv

The local project file is:

`data/billboard.csv`

The Billboard dataset contains chart information such as:

- song title
- performer
- chart position
- previous chart position
- peak position
- weeks on chart

### Spotify Data

The Spotify dataset was obtained from Kaggle:

https://www.kaggle.com/datasets/yamaerenay/spotify-dataset-19212020-600k-tracks

The project uses the `tracks.csv` file from this dataset.

The local project file is:

`data/spotify.csv`

The Spotify dataset contains song information and audio features including:

- danceability
- energy
- loudness
- speechiness
- acousticness
- instrumentalness
- liveness
- valence
- tempo

The Spotify and Billboard datasets were combined using normalized song title/name and artist information to identify matching songs.

## Dataset

Two source datasets were used:

- `billboard.csv`
- `spotify.csv`

Songs were matched using normalized song title/name and artist information.

The final dataset contains:

- 10,000 songs
- 4,000 HIT songs
- 6,000 NON-HIT songs
- Random seed: `42`
- No missing values
- No duplicate songs
- No release year feature

## Features

The following nine Spotify audio features were used:

- `danceability`
- `energy`
- `loudness`
- `speechiness`
- `acousticness`
- `instrumentalness`
- `liveness`
- `valence`
- `tempo`

Target:

- `hit`

## Exploratory Data Analysis

The project performs:

- Class distribution analysis
- Descriptive statistics
- Feature distribution analysis
- HIT vs NON-HIT boxplots
- Feature correlation analysis
- HIT vs NON-HIT mean comparison

## Machine Learning Models

Five classification models were evaluated:

1. Logistic Regression
2. Decision Tree
3. Support Vector Machine (SVM)
4. K-Nearest Neighbors (KNN)
5. Artificial Neural Network (ANN) using `sklearn.neural_network.MLPClassifier`

### Train-Test Split

- 80% training data
- 20% testing data
- `random_state=42`
- Stratified split

### Scaling

- Logistic Regression, SVM, and KNN use `StandardScaler`.
- The ANN uses the same training-fitted `StandardScaler`.
- Decision Tree uses unscaled features.

### ANN Architecture

- Input: 9 scaled Spotify audio features
- Hidden layers: 64 neurons followed by 32 neurons
- Hidden-layer activation: ReLU
- Solver: Adam
- `max_iter=500`
- `random_state=42`
- Early stopping with a validation fraction of `0.1`

`MLPClassifier` performs forward propagation, loss calculation, backpropagation,
and gradient-based weight updates internally.

## Baseline Results

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| SVM | 0.6660 | 0.5935 | 0.5238 | 0.5564 |
| Logistic Regression | 0.6445 | 0.5574 | 0.5400 | 0.5486 |
| KNN | 0.6335 | 0.5407 | 0.5563 | 0.5484 |
| Decision Tree | 0.6015 | 0.5018 | 0.5100 | 0.5059 |
| ANN | 0.6615 | 0.5701 | 0.6250 | 0.5963 |

The ANN baseline ROC-AUC was **0.7323**.

## Hyperparameter Tuning

`GridSearchCV` was used with:

- 5-fold cross-validation
- `scoring="f1"`

### Logistic Regression

- `C = [0.01, 0.1, 1, 10, 100]`

### Decision Tree

- `max_depth = [3, 5, 7, 10, None]`
- `min_samples_split = [2, 5, 10]`
- `min_samples_leaf = [1, 2, 4]`

### SVM

- `C = [0.1, 1, 10]`
- `kernel = ["linear", "rbf"]`

### KNN

- `n_neighbors = [3, 5, 7, 9, 11]`
- `weights = ["uniform", "distance"]`

### ANN

- `hidden_layer_sizes = [(32,), (64, 32), (128, 64)]`
- `alpha = [0.0001, 0.001]`
- `activation = "relu"`
- `solver = "adam"`
- `max_iter = 500`
- Early stopping enabled

Model selection uses the highest five-fold cross-validation F1-score. The held-out
test set is used only for final evaluation after selection.

## Tuned Results

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Decision Tree | 0.6195 | 0.5157 | 0.8013 | 0.6275 |
| SVM | 0.6390 | 0.5423 | 0.6250 | 0.5807 |
| Logistic Regression | 0.6445 | 0.5574 | 0.5400 | 0.5486 |
| KNN | 0.6190 | 0.5239 | 0.5213 | 0.5226 |
| ANN | 0.6615 | 0.5701 | 0.6250 | 0.5963 |

The tuned ANN selected `hidden_layer_sizes=(64, 32)` and `alpha=0.0001`.
Its cross-validation F1-score was **0.5829** and its test ROC-AUC was **0.7323**.

## Final Model

The tuned Decision Tree was selected as the final model based on the highest
five-fold cross-validation F1-score (**0.6226**). The held-out test set was not
used to select the final model.

### Best Parameters

- `max_depth = 3`
- `min_samples_split = 2`
- `min_samples_leaf = 1`

### Final Performance

- Accuracy: **61.95%**
- Precision: **51.57%**
- Recall: **80.13%**
- F1-score: **62.75%**

### Interpretation

The final model has relatively high recall, meaning it identifies a large proportion of actual HIT songs. However, its precision is comparatively lower, so some songs predicted as HIT are actually NON-HIT.

These results do not establish that the model is highly accurate or production-ready.

## Results

The repository contains:

- Final confusion matrix
- Baseline vs tuned F1-score comparison
- Model comparison CSV
- Saved final Decision Tree model

## Project Structure

```text
Hit-Song-Prediction/
├── data/
│   ├── billboard.csv
│   ├── spotify.csv
│   └── final_dataset.csv
├── notebooks/
│   ├── analytics.ipynb
│   ├── feature_analysis.ipynb
│   └── model_training.ipynb
├── src/
│   ├── clean_data.py
│   ├── merge_data.py
│   ├── train_models.py
│   ├── evaluate_models.py
│   ├── save_final_model.py
│   └── export_confusion_matrix.py
├── models/
│   └── decision_tree_final.pkl
├── results/
│   ├── confusion_matrix.png
│   ├── baseline_vs_tuned_f1.png
│   └── model_comparison.csv
├── requirements.txt
└── README.md
```

## How to Run

### 1. Create a Python Virtual Environment

From the project root:

```powershell
python -m venv .venv
```

Activate it on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Data Preparation Scripts

From the project root:

```bash
python src/clean_data.py
python src/merge_data.py
```

The data preparation workflow uses the source datasets in `data/` and produces the final dataset used by the project.

### 4. Open the Notebooks

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Then open the notebooks in this order:

1. `notebooks/analytics.ipynb`
2. `notebooks/feature_analysis.ipynb`
3. `notebooks/model_training.ipynb`

### 5. Run Project Scripts

The available project scripts can be run from the project root:

```bash
python src/train_models.py
python src/evaluate_models.py
python src/save_final_model.py
python src/export_confusion_matrix.py
```
