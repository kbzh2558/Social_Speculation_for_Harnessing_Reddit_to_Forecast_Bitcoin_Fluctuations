# 📝🔗 Social Speculation: Harnessing Reddit to Forecast Bitcoin Fluctuations
![](https://img.shields.io/badge/python-3.10%2B-blue?logo=Python)

⚡ETL, Sentiment Analysis, Multi-Layer Perceptron Modelling, and Dropout Regularization⚡

> [!NOTE]
> This project investigates the role of Reddit sentiment as an indicator for predicting Bitcoin (BTC) price fluctuations. Recognizing the complex, sentiment-driven nature of cryptocurrency, we explore models that leverage sentiment analysis on Reddit comments to forecast BTC's closing price. Starting from a baseline regression, we refine our model by employing a multi-layer perceptron (MLP) with dropout regularization, achieving a solution that balances complexity and generalization. This README provides a structured overview of the project’s stages, data processing, and insights on the limitations and future directions for improving model performance.

## Repo Structure
```
Social_Speculation/
├── README.md
├── ETL_workflow.ipynb           # Data extraction, transformation, and loading scripts
├── Model_prediction.ipynb       # Training and evaluation of MLP model with dropout regularization
├── credentials.py               # Access credentials for API access
├── linear_model.py              # Baseline linear regression model implementation
├── mlp_model.py                 # One-layered MLP implementation
├── mlp.py                       # Two-layered MLP implementation
├── mlp_dropout.py               # Two-layered MLP implementation with dropout regularization
├── best_model.pth               # Saved state of the best-performing model during early stopping
├── mlp_trained_model.pth        # Saved state of the final trained MLP model
├── reddit.db                    # Database of Reddit comments and BTC data
├── report.md                    # Detailed report
├── image.png                    # Visual depiction of model structure or results
```

## Step-by-Step Breakdown

1. <details>
    <summary>Data Collection and ETL Process</summary>

    - Data for this study was obtained from Reddit, focusing on BTC-related posts and comments. Sentiment scores were calculated using sentiment analysis to quantify public opinion.
    - Key preprocessing steps included:
        - Filtering and structuring Reddit data to ensure relevance and consistency.
        - Calculating sentiment polarity scores for each comment to assess public sentiment on BTC.

    **NOTE:** This process streamlined data preparation for subsequent analysis, ensuring that only relevant, clean data entered the modeling pipeline.

   </details>

2. <details>
    <summary>Baseline Model: Simple Regression</summary>

    - A foundational regression model was implemented to map sentiment scores to BTC's closing price directly.
    - This model assumed an immediate impact of sentiment on price, offering a straightforward but limited approach, primarily useful as a benchmark against more complex models.
    
    - **Challenges:** This approach showed limitations in handling intricate relationships and temporal dependencies.

   </details>

3. <details>
    <summary>Transition to Multi-Layer Perceptron (MLP) with Dropout</summary>

    - Observing underfitting with a single-layer MLP, a two-layer MLP structure with dropout regularization was introduced to enhance learning complexity while mitigating overfitting.
    - Dropout layers were added between hidden layers, randomly deactivating neurons during training to prevent over-reliance on specific nodes, which allowed the model to generalize better.

    - **Results:** This MLP structure demonstrated improved ability to capture sentiment-related patterns in BTC price fluctuations, balancing model complexity and generalization.

   </details>

4. <details>
    <summary>Limitations and Potential Improvements</summary>

    - **Limitations:** While the model performed well on short-term price fluctuations, it struggled to capture broader directional trends in BTC’s price due to the lack of temporal awareness in the current structure.
    - **Future Work:** To address this, a Convolutional Neural Network (CNN) approach for time-series data could be explored to capture both short- and long-term trends in BTC price by treating sentiment scores as one-dimensional sequences.

    - **Recommendation:** Implementing CNN could enhance the model's capacity to recognize temporal patterns, providing improved prediction accuracy for long-term trends.

   </details>

## Key Takeaways
The refined MLP model represents a robust approach to capturing sentiment-driven fluctuations in Bitcoin price. However, incorporating CNN architecture in future work could strengthen predictive performance by integrating temporal dependencies, offering a valuable tool for forecasting in the volatile cryptocurrency market.

For more details, please refer to `report.md` for in-depth analyses and modeling insights.
