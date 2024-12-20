# **Powered Simulator for BMI and Health Predictions Based on Lifestyle Modifications**

## **Overview**
Obesity is a growing global health concern, with over 1 billion individuals affected as of 2022. In South Korea, the adult obesity rate surged from 35.1% in 2011 to 46.3% in 2021 (WHO). This project offers a personalized health management tool that empowers users to calculate their Body Mass Index (BMI) and simulate potential improvements based on lifestyle changes.

### **Key Features**
- **BMI Calculation**: Calculates BMI using user-provided data.
- **Lifestyle Simulation**: Predicts BMI changes based on lifestyle adjustments, such as increased physical activity or improved dietary habits.
- **Interactive Web Application**: Provides results through an intuitive Streamlit-based interface.

---

## **Technical Details**

### **Dataset**
- **Sources**:
  - [Kaggle: Obesity Dataset](https://www.kaggle.com/datasets/suleymansulak/obesity-dataset)
  - [UCI Repository: Obesity Levels Estimation Dataset](https://archive.ics.uci.edu/dataset/544)
- **Size**: 3,722 rows, 20 columns
- **Target Variable**: BMI

### **Preprocessing Steps**
1. **Dataset Merging**:
   - Two datasets with similar columns were merged, using **BMI as the primary criterion** for alignment and consistency.
   - Overlapping columns such as age, weight, and height were standardized and combined.
2. **Missing Data Handling**:
   - Applied **KNN Imputation** to handle missing values for numerical columns.
   - Rows with extensive missing or invalid entries were excluded.
3. **Feature Scaling**:
   - All numeric features were scaled using `StandardScaler` to normalize input for model training.
4. **Categorical Conversion**:
   - Transformed categorical data (e.g., gender, transportation) into numerical formats suitable for machine learning.

---

### **Modeling Approach**
- **Algorithm**: Multilayer Perceptron (MLP)
  - **Hidden Layers**: (200, 150, 100, 50)
  - **Activation Function**: ReLU
  - **Optimizer**: Adam
  - **Loss Function**: Mean Squared Error (MSE)
  - **Hyperparameter Tuning**: Utilized HyperOpt for parameter optimization.
- **Performance**:
  - **Train MSE**: 0.01624
  - **Test MSE**: 0.01952
  - **R² Score**: 0.9997
  - **MAPE**: 0.38%
  - **Cross-Validation**: 5-fold cross-validation ensured stability and reduced overfitting risks.

---

### **Streamlit Web Application**
#### **Interactive Features**
- Calculate current BMI based on user input.
- Simulate BMI changes with lifestyle adjustments.
- Provide personalized health recommendations based on BMI level and habits.

#### **Visualization**
- Users can explore BMI distribution within their age group and compare it to their current BMI.

---

## **Limitations and Future Enhancements**

### **Current Limitations**
1. **Data Representation**: Limited data for extreme BMI values and older demographics (50+ years).
2. **Feature Impact**: Weak correlation observed for some lifestyle variables, e.g., smoking or transportation mode.

### **Planned Improvements**
1. Expand datasets to include more diverse demographics.
2. Experiment with advanced machine learning algorithms (e.g., XGBoost, LightGBM).
3. Add features such as a gym locator or personalized diet suggestions.

---

## **How to Use**

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-repo-link
