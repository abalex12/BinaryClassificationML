# Binary Classification Model Training

This project is designed to automate the training and evaluation of various binary classification models using machine learning. The code allows you to specify multiple models, perform hyperparameter tuning, and evaluate their performance on a dataset.

## Table of Contents

- [Objective](#objective)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Objective

The primary goal of this project is to facilitate the comparison of different machine learning models on binary classification tasks, enabling users to identify the best-performing model for their data.

## Features

- **Model Specification**: Support for multiple classification algorithms including K Nearest Neighbors, Support Vector Classifiers, Logistic Regression, Decision Trees, and more.
- **Hyperparameter Tuning**: Utilizes Grid Search for optimal hyperparameter selection.
- **Performance Evaluation**: Calculates accuracy and F1 score for model performance on training and test datasets.
- **Easy Integration**: Simple functions that can be integrated with any binary classification dataset.

## Installation

To run this project, you will need to have Python installed along with the required libraries. You can install the required libraries using `pip`. Here’s how:

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. Install the required packages:
   ```bash
   pip install pandas numpy matplotlib scikit-learn
   ```

## Usage

To use the provided code, follow these steps:

1. Load your dataset into a Pandas DataFrame.
2. Specify the target column name.
3. Call the `auto_train_binary_classifier` function with your DataFrame, target column name, and specified models.

Here's an example of how to use the code with the breast cancer dataset:

```python
from sklearn.datasets import load_breast_cancer
import pandas as pd

# Load the breast cancer dataset
cancer = load_breast_cancer()
cancer_df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
cancer_df['target'] = pd.Series(cancer.target)

# Specify models and train
models = specify_models()
best_model_name, best_model, train_set_score, test_set_score = auto_train_binary_classifier(
    cancer_df, 'target', models)

# Print results
print(f"Best Model: {best_model_name}")
print(f"Trained Model: {best_model}")
print(f"Training Set Score (F1): {train_set_score}")
print(f"Test Set Score (Accuracy): {test_set_score}")
```

## Dataset

This project currently uses the breast cancer dataset from the `sklearn.datasets` module. This dataset consists of features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. The task is to classify whether a tumor is malignant or benign.

## Results

After running the code, you will receive output indicating:
- The name of the best-performing model.
- The trained model object.
- Training set performance (F1 score).
- Test set performance (Accuracy).

## Contributing

Contributions are welcome! If you have suggestions for improvements or additional models, please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
