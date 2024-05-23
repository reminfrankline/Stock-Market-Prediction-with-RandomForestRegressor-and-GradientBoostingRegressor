# Stock Market Prediction with RandomForestRegressor and GradientBoostingRegressor

This project aims to predict stock market prices using machine learning models, specifically RandomForestRegressor and GradientBoostingRegressor. The code utilizes the cuDF library for accelerated data processing on NVIDIA GPUs.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)

## Introduction

Predicting stock market prices is a challenging task that can benefit from machine learning techniques. This project explores the use of RandomForestRegressor and GradientBoostingRegressor models for predicting stock prices. The dataset used in this project is sourced from [Yahoo Finance](https://finance.yahoo.com/) and includes historical stock price data for analysis and prediction.

## Features

- Data preprocessing using cuDF for accelerated data processing on GPUs.
- Feature engineering to create relevant features for model training.
- Model training with RandomForestRegressor and GradientBoostingRegressor.
- Hyperparameter tuning using GridSearchCV for improved model performance.
- Evaluation of model performance using Mean Squared Error (MSE) and R-squared (R2) Score.

## Requirements

- Python 3.x
- cuDF
- scikit-learn
- matplotlib
- pandas

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/reminfranklin/Stock-Market-Prediction-with-RandomForestRegressor-and-GradientBoostingRegressor.git
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Navigate to the project directory:

   ```bash
   cd Stock-Market-Prediction-with-RandomForestRegressor-and-GradientBoostingRegressor
   ```

2. Run the main Python script:

   ```bash
   python Stock_Market_Prediction.py
   ```

3. Follow the instructions on the console to preprocess the data, train the models, and evaluate their performance.

## Contributing

Contributions are welcome! If you'd like to contribute to this project, please feel free to submit a pull request or open an issue with your suggestions, bug reports, or feature requests.
