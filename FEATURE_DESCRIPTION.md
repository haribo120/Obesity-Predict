# **Feature Description**

This document provides a detailed description of the features used in the BMI and health prediction project. Each feature is described along with its values and preprocessing steps where applicable.

---

## **Feature Details**

1. **Gender (Sex)**
   - **Description**: Gender of the individual.
   - **Values**: 
     - `1`: Male
     - `2`: Female

2. **Age**
   - **Description**: Age of the individual in years (integer).

3. **Height**
   - **Description**: Height of the individual, converted to centimeters.
   - **Action**: Ensure all height values are in cm.

4. **Weight**
   - **Description**: Weight of the individual, calculated based on BMI ranges.
   - **Action**: Use the midpoint of the BMI range for weight estimation, considering EDA results and potential errors.

5. **Family History with Overweight**
   - **Description**: Indicates if the individual has a family history of overweight or obesity.
   - **Values**:
     - `1`: Yes
     - `2`: No

6. **FAVC (Preference for High-Calorie Foods)**
   - **Description**: Indicates preference for high-calorie foods.
   - **Values**:
     - `1`: Yes
     - `2`: No
   - **Note**: Only available in dataset 2.

7. **Consumption of Fast Food**
   - **Description**: Indicates if the individual consumes fast food.
   - **Values**:
     - `1`: Yes
     - `2`: No
   - **Note**: Only available in dataset 1.

8. **FCVC (Frequency of Consuming Vegetables)**
   - **Description**: Frequency of vegetable consumption.
   - **Values**:
     - `1`: Rarely
     - `2`: Sometimes
     - `3`: Always

9. **NCP (Number of Main Meals per Day)**
   - **Description**: Number of meals consumed per day.
   - **Values**:
     - `1`: 1-2 meals
     - `2`: 3 meals
     - `3`: More than 3 meals

10. **CAEC (Food Intake Between Meals)**
    - **Description**: Frequency of snacking between meals.
    - **Values**:
      - `1`: Rarely
      - `2`: Sometimes
      - `3`: Always

11. **SMOKE (Smoking)**
    - **Description**: Indicates if the individual smokes.
    - **Values**:
      - `1`: Yes
      - `2`: No

12. **CH2O (Daily Water Consumption)**
    - **Description**: Daily water consumption.
    - **Values**:
      - `1`: Less than 1L
      - `2`: 1-2L
      - `3`: More than 2L

13. **SCC (Calorie Intake Monitoring)**
    - **Description**: Indicates if the individual monitors their calorie intake.
    - **Values**:
      - `1`: Yes
      - `2`: No

14. **FAF (Frequency of Physical Activity)**
    - **Description**: Weekly physical activity frequency.
    - **Values**:
      - `0`: No physical activity
      - `1`: 1-2 days
      - `2`: 3-4 days
      - `3`: More than 5 days
    - **Action**: Adjust values in dataset 1 where 4 and 5 are changed to 3.

15. **TUE (Technology Usage per Day)**
    - **Description**: Daily technology usage duration.
    - **Values**:
      - `0`: 0-2 hours
      - `1`: 3-5 hours
      - `2`: More than 5 hours
    - **Action**: Adjust the index values where necessary.

16. **CALC (Alcohol Consumption)**
    - **Description**: Frequency of alcohol consumption.
    - **Values**:
      - `0`: No
      - `1`: Sometimes
      - `2`: Frequently
    - **Note**: Only available in dataset 2.

17. **MTRANS (Mode of Transportation)**
    - **Description**: The individual’s primary mode of transportation.
    - **Values**:
      - `1`: Automobile
      - `2`: Motorbike
      - `3`: Bicycle
      - `4`: Public Transportation
      - `5`: Walking
    - **Action**: Adjust the index for certain values as needed.

18. **NObeyesdad (Obesity Classification)**
    - **Description**: Classification based on BMI (Obesity status).
    - **Note**: Only available in dataset 2.

19. **BMI (Body Mass Index)**
    - **Description**: The individual’s BMI, calculated as weight divided by height squared (kg/m²).
    - **Note**: Only available in dataset 2.

20. **Class (Obesity Classification)**
    - **Description**: Obesity classification based on BMI ranges.
    - **Values**:
      - **Underweight**: BMI < 18.5
      - **Normal**: 18.5 ≤ BMI < 24.9
      - **Overweight**: 25 ≤ BMI < 29.9
      - **Obesity**: BMI ≥ 30
    - **Action**: Adjust class labels in dataset 1 according to BMI values.

---

## **Usage Notes**
- **Target Variable**: `BMI` is used as the primary predictor and dependent variable.
- **Categorical Conversion**: Categorical values are converted into numerical values during preprocessing for compatibility with machine learning algorithms.
- **Merging Strategy**: Dataset merging aligns overlapping columns based on BMI and demographic features.

---
