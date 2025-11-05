# Geophone Activity Classification Model

This Python script trains a machine learning model to classify human activities (**walking**, **running**, and **digging**) using time-series data from a geophone sensor.

The script demonstrates a complete (but local) machine learning pipeline:
1.  Loading and labeling multiple raw data files.
2.  Performing time-series **feature engineering** using rolling windows.
3.  Training a `RandomForestClassifier` model.
4.  Evaluating the model's performance with an accuracy score, classification report, and a visual confusion matrix.
5.  Showing which engineered features were most important for the model's predictions.

## 🚀 Core Technique: Time-Series Feature Engineering

This model's high accuracy comes from **feature engineering**. Instead of training on raw, noisy sensor data, it transforms the data first.

It calculates statistical features (like standard deviation, mean, max, etc.) over a "rolling window" of 25 samples. This allows the model to learn the unique *patterns* of each activity (e.g., "running has a high rolling standard deviation") rather than just individual data points.

## 📋 Requirements

You will need Python 3 and the following libraries:

* `pandas`
* `numpy`
* `matplotlib`
* `seaborn`
* `scikit-learn`

You can install them all using pip:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
