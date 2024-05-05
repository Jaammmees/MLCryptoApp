> **Important:** This project is currently under development and some functionalities may not be finished.


# Machine Learning Backtester Application

This application provides a NEAR comprehensive environment for backtesting machine learning models, particularly focused on financial data. It allows users to load models, data, and scalers, configure model training, and execute backtests.

## Features

- **Model Loading**: Load `.h5` TensorFlow/Keras models to predict or analyze time-series data.
- **Data Loading and Processing**: Load data from Excel, CSV or Parquet files, allowing for modification and scaling.
- **Training Configuration**: Set up sequence generation for LSTM or other sequence-dependent models and configure training parameters such as epochs and learning rate.
- **Backtesting**: Run simulations or backtesting strategies using historical data to evaluate model performance.
- **GUI Based Interaction**: All functionalities are accessible via a custom-built GUI, making it easy to navigate and use.

## Installation

Before running the application, ensure you have Python installed on your system. This application requires Python 3.7 or later. You can download Python from [here](https://www.python.org/downloads/).

### Dependencies

This application relies on several Python libraries, including `customtkinter`, `pandas`, `numpy`, `tensorflow`, and others. Install all required dependencies by running:

```bash
pip install customtkinter pandas numpy tensorflow
```

## Running the Application
```bash
python main.py
```
Ensure that main.py is in the current directory. This script initializes the GUI and loads the initial configuration.

## Usage
Upon launching the application, you will be greeted with a GUI window structured with a sidebar for navigation and a main display area for interaction with models, data, and configurations:


