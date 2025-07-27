# Battery Life Prediction using Support Vector Regressor with Hyperparameter Tuning and Cross-Validation

This project focuses on predicting the Remaining Useful Life (RUL) of batteries, a critical aspect for electric vehicles and other battery-dependent systems. It proposes an Artificial Intelligence (AI) driven predictive model utilizing a hyperparameter-tuned Support Vector Regressor (SVR) to enhance RUL estimation. The project compares the performance of a baseline SVR model with an optimized SVR model, demonstrating significant improvements in accuracy and reliability.

## Table of Contents

* [Problem Statement](#problem-statement)
* [Solution Approach](#solution-approach)
* [Key Features](#key-features)
* [System Architecture](#system-architecture)
* [Experimental Results](#experimental-results)
* [Future Work](#future-work)
* [Installation and Usage](#installation-and-usage)
* [Contributing](#contributing)

## Problem Statement

Battery performance is a significant concern for automobile manufacturers, especially with the increasing adoption of electric vehicles. Unmonitored and excessive use can reduce battery lifespan. Accurately predicting the Remaining Useful Life (RUL) of batteries is challenging due to numerous influencing factors and ambiguities in battery aging. Conventional statistical models and general machine learning methods often struggle to capture the complex, nonlinear deterioration patterns of batteries, resulting in inaccuracy in predictions.

## Solution Approach

This research introduces a data-driven method based on Support Vector Regression (SVR) with hyperparameter optimization and cross-validation techniques to address the limitations of current battery health monitoring systems. The core idea is to leverage AI to effectively predict battery lifespan. The research seeks to improve RUL estimation towards the ultimate aim of battery health optimization and sustainability.

The methodology involves:
1.  **Baseline SVR Model:** An initial SVR model is trained with default hyperparameters to establish a benchmark performance.
2.  **Hyperparameter Tuning:** Hyperparameter tuning is achieved using GridSearchCV, which optimizes important parameters like C (regularization), gamma (kernel coefficient), and epsilon (margin of tolerance for predictions).
3.  **Cross-Validation:** Cross-validation is applied to reduce overfitting and improve the model's response to unseen data. This technique separates the data into training and testing sections, making several folds where each fold is used to train the model, introducing it to different kinds of observations and reducing reliance on limited observations.
4.  **Performance Evaluation:** The performance of the hyperparameter-tuned SVR model is compared with the baseline SVR model using various plots and metrics to prove that the proposed model provides better accuracy and performance.

## Key Features

* **AI-driven Predictive Model:** Develops an AI-driven predictive model to enhance the estimation of battery RUL.
* **Hyperparameter-tuned SVR:** A hyperparameter-tuned and polished Support Vector Regressor (SVR) model is compared with a baseline SVR model.
* **Cross-Validation Integration:** Applies cross-validation to reduce overfitting and improve the model's response to unseen data.
* **Feature Selection:** Identifies and selects the most influential features from the dataset to ensure a comprehensive understanding of battery aging mechanisms.
* **Reliable Battery Monitoring Tool:** The proposed model can serve as a reliable tool in battery monitoring systems.
* **Predictive Maintenance:** Supports predictive maintenance strategies to mitigate failures and extend battery life to its maximum limit.
* **Performance Metrics:** Evaluates model performance using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R² Score.
* **Visualization:** Compares results using scatter plots (predict vs. actual RUL), regression line fits, learning curves (plotting training and validation errors), and residual plots (demonstrating prediction errors).

## System Architecture

The Battery RUL Prediction System is implemented as a machine learning pipeline, divided into several modular layers to ensure efficient error handling and resolution.

The architecture comprises the following layers:
1.  **Data Collection Layer:** Gathers historical battery data, user input, and data from battery sensors.
2.  **Data Preprocessing Layer:** Handles data cleaning, normalization (scaling data between 0 and 1), and managing missing values.
3.  **Exploratory Data Analysis (EDA) Layer:** Includes trend analysis, correlation analysis, and crucial feature selection to identify principal components influencing the data.
4.  **Model Development Layer:** Implements the baseline SVR model and performs hyperparameter tuning (using GridSearchCV) to optimize its performance, leading to the optimized SVR model.
5.  **Evaluation Layer:** Calculates performance metrics such as Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R² Score to assess and compare the models.
6.  **Visualization Layer:** Presents results in diagrammatic form using various plots and graphs, including Regression Fit Comparison, Error Distribution Analysis, and Learning Curve Visualization.
7.  **Prediction Output:** Displays the final RUL of a battery and performance improvements to the end-users.

The entire system operates as a two-entity model: users interact via a program script, and the system processes data to display results.

![Architecture Diagram](/img/architecture.png "Figure 1: Architecture Diagram")

## Experimental Results

The dataset used in this research contains 15,054 rows and 9 columns, with parameters significantly influencing the RUL. The SVR model successfully predicts battery RUL by studying data trends, and hyperparameter tuning further enhances its performance. It can be validated that tuning a model significantly enhances its capabilities, making it a preferable method over baseline models for battery state estimation. This upgrade will result in high standards of accuracy and reliability of the predictions and satisfy the ever-growing demand for battery monitoring and RUL systems.

### Performance Metrics

| Model | MAE ↓ | RMSE ↓ | R² Score ↑ |
| :-------------------------- | :------------------ | :------------------- | :------------------ |
| Baseline SVR (without tuning) | 199.5412502912683 | 233.30672855878518 | 0.47452132624852617 |
| Optimized SVR (with GridSearchCV) | 81.31827948395154 | 144.355780606381 | 0.7988273549390262 |

MAE calculates the average magnitude of the absolute errors between predicted and actual values; a lower MAE suggests better accuracy. RMSE is the square root of MAE, making it easier to interpret as it is in the same unit as the target variable; a lower RMSE signifies better model performance. R² Score evaluates how well the model explains the variance in the target variable, with values closer to 1 indicating stronger predictive performance. The optimized SVR model shows reduced MAE and RMSE, and an improved R² score over the baseline SVR.

### Visualizations

* **Regression Line Fit (Predicted vs. Actual RUL):**
    * **Baseline SVR:** Predicted values have a large spread from the perfect prediction line, signifying large variations from the actual RUL. The model struggles to capture intrinsic relationships, leading to increased errors.
    * **Optimized SVR:** The model, shown in green, has a denser distribution of points near the ideal prediction line, indicating enhanced predictive accuracy and reduced over/underestimation.

    ![Regression Line Fit](/img/reg_line.png "Figure 2: Baseline SVR prediction plot. & Figure 3: Tuned SVR prediction plot.")

* **Error Distribution Plot:**
    * The blue histogram indicates the baseline SVR model's error distribution, while the green histogram shows the optimized SVR model. The red dashed vertical line at zero represents perfect prediction.
    * The optimized SVR model (green) has a tighter error distribution, with predictions more closely bunched around zero, reflecting lower variance and better accuracy than the baseline SVR model. The baseline model (blue) has a wider spread, suggesting larger prediction errors and more extreme deviations.

    ![Error Distribution Plot](/img/error_dis.png "Figure 4: Error distribution plot.")

* **Learning Curve Plot:**
    * Graphically illustrates the training and validation errors of both models. The blue dashed line indicates the training error of the baseline SVR, and the red solid line indicates its validation error. The green dashed line indicates the training error of the optimized SVR, and the orange solid line indicates its validation error.
    * The baseline SVR does not generalize data as much, as there is more distance between its training error and validation error. The tuned SVR has less distance between training and validation error, resulting in more generalization and overall performance.

    ![Learning Curve Plot](/img/learn_curve.png "Figure 5: Learning curve plot.")

These results validate that tuning a model significantly enhances its capabilities, making it a preferable method over baseline models for battery state estimation. The enhanced model accuracy will be capable of minimizing downtime in operations, maximizing battery utilization, and making better decisions in energy storage systems.

## Future Work

Future efforts will focus on:
* **Deep Learning Architectures:** Investigating deep learning architectures such as LSTMs, GRUs, or Transformers to identify intricate degradation patterns.
* **Real-time BMS Integration:** Incorporating the model into real-time Battery Management Systems (BMS) for ongoing monitoring.
* **Enhanced Feature Sets:** Increasing feature sets with sensor-based parameters such as temperature and voltage variations for enhanced robustness.
* **Hybrid Methods:** Exploring hybrid methods that incorporate SVR with other machine learning models to further increase prediction accuracy.
* **Broader Battery Chemistries:** Expanding the model's use to other battery chemistries, such as Lithium-ion and Solid-State Batteries.
* **Cloud-based Platform:** Implementing a cloud-based platform for remote diagnostics to increase its practical impact.

## Installation and Usage

### Prerequisites

* Python 3.x
* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn

You can install these dependencies using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Contributing
Contributions are welcome! Please feel free to open issues or submit pull requests.
1.  Fork the repository.
2.  Create your feature branch (`git checkout -b feature/AmazingFeature`).
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the branch (`git push origin feature/AmazingFeature`).
5.  Open a Pull Request.
