# Wine Quality Test Project
This project leverages machine learning techniques to predict the quality of wine based on various physicochemical properties. Using the PyCaret library, we determined the best algorithm for predicting wine quality and finalized a **Logistic Regression** model for the task. The project was a collaborative effort with another GitHub user.

## Features
1. **Data Preprocessing**:
   - Handling missing values.
   - Normalizing numerical features for consistent scaling.
   - Encoding categorical variables (if any).

2. **Algorithm Selection with PyCaret**:
   - Automated model comparison and performance evaluation using PyCaret's classification module.
   - Logistic Regression emerged as the best-performing model.

3. **Final Model**:
   - Logistic Regression was used to model the data and predict wine quality.

## Dataset
- The dataset contains physicochemical properties (e.g., pH, acidity, sugar content) and corresponding wine quality ratings.

### Preprocessing Steps
- Missing values were imputed.
- Numerical features were normalized to ensure uniform scaling.
- Any categorical data was label-encoded for compatibility with machine learning algorithms.

## Tools and Libraries Used
- **PyCaret**: For automated machine learning and model comparison.
- **Scikit-learn**: For the Logistic Regression model.
- **Pandas and NumPy**: For data preprocessing.
- **Matplotlib and Seaborn**: For visualizations.
- **Jupyter Notebook**: For exploratory data analysis and model development.

## Results
- Logistic Regression was selected as the best-performing model based on PyCaret's evaluation.
- Visualizations:
  - Feature importance plots.
    
## Collaboration
This project was done in collaboration with https://github.com/arnavtiet.

## Future Work
- Test additional algorithms such as Random Forest and Gradient Boosting for potential performance improvements.
- Fine-tune the Logistic Regression model with hyperparameter optimization.
- Extend the analysis to include other datasets or features.
- Deploy the model as a web application using Flask or Streamlit.

## Acknowledgments
- PyCaret for simplifying machine learning workflow.
- The dataset source for providing the data.
- Collaborative support from the project contributors.

