# SmartPricing-A-Deep-Learning-Based-Dynamic-Pricing-System-for-E-commerce-Entrepreneurs
This repository contains the code and documentation for "SmartPricing," a deep learning-based dynamic pricing model tailored for e-commerce entrepreneurs, particularly in the fashion industry. This project aims to assist entrepreneurs in understanding the market value of textile products and optimizing pricing strategies based on relevant features such as brand and category.

### Project Overview
Dynamic pricing is crucial for maximizing profits in the competitive fashion e-commerce landscape. This project utilizes a deep learning model to predict product prices based on various attributes, helping e-commerce entrepreneurs optimize pricing and gain insights into market trends. The model uses a multi-layered neural network architecture with batch normalization and dropout regularization, providing an effective balance between model complexity and generalization.

### Key Features
- Fashion Industry-Focused Pricing: Tailored to predict textile and fashion product prices, the model leverages features unique to the fashion sector, such as brand and category.
- Custom Accuracy Metric: Includes a custom accuracy function that calculates the percentage of predictions within a certain margin of actual prices, giving practical insights into the model's dynamic pricing precision.
- Data Preprocessing and Feature Engineering: Preprocessing includes handling missing values, feature extraction, one-hot encoding for categorical features, and log transformation for price normalization.
- Neural Network with Regularization: The model employs dropout layers and batch normalization to mitigate overfitting and enhance prediction accuracy.
- Early Stopping and Adam Optimizer: Uses early stopping to avoid overfitting, while the Adam optimizer facilitates fast and efficient training.
- 
### Dataset
The model is built using the Myntra Fashion Products dataset, which includes various attributes of fashion items, such as:

- p_id: Unique product identifier
- name: Product name
- products: Product category
- price: Product price
- color: Product color
- brand: Product brand
- img: Image URL of the product
- ratingCount: Number of ratings
- avg_rating: Average rating
- description: Product description
 
### Model Architecture
- Input Layer: 128 neurons with ReLU activation.
- Hidden Layers: One dense layer with 64 neurons, also using ReLU activation.
- Dropout Layers: Dropout rate of 0.2 after each dense layer to prevent overfitting.
- Output Layer: Single neuron with linear activation to predict continuous price values.
- Loss Function: Mean Absolute Error (MAE), which is robust for regression tasks.
- Optimizer: Adam optimizer, with early stopping for efficient training.

### Results
The model achieved:
Mean Absolute Error (MAE):The model achieved a MAE of 0.4108, indicating a high level of accuracy in price prediction.
- Custom Accuracy :The model's ability to predict prices within 10% of the actual price with an accuracy of 84.12% demonstrates its practical applicability in dynamic pricing. 

### Future Improvements
- Enhanced Feature Engineering: Adding more features like seasonality and competitor pricing.
- Advanced Models: Exploring transformer models or ensemble methods for enhanced performance.
- Hyperparameter Tuning: Using Grid Search or Bayesian Optimization to further refine the model.
