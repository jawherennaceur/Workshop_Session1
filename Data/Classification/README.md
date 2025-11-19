# Drug Classification Dataset Description

## Overview

Predict the correct drug for a patient using medical and demographic data
Multi class classification

## Goal

Predict the drug label using patient features

## Features Description

### 1. **Age** (integer)

- Patient's age in years
- Ranges typically from 15 to 75 years
- Older patients may require different medications

### 2. **Sex** (categorical)

- Patient's biological sex
- Possible values:
  - **M**: Male
  - **F**: Female
- Gender can affect drug metabolism and effectiveness
- Important consideration in personalized medicine

### 3. **BP** (Blood Pressure) (categorical - ordinal)

- Patient's blood pressure level classification
- Possible values (in order of severity):
  - **LOW**: Below normal blood pressure (hypotension)
  - **NORMAL**: Healthy blood pressure range
  - **HIGH**: Elevated blood pressure (hypertension)
- Critical factor in determining appropriate medication
- Different drugs target different BP conditions

### 4. **Cholesterol** (categorical - ordinal)

- Patient's cholesterol level classification
- Possible values:
  - **NORMAL**: Healthy cholesterol levels
  - **HIGH**: Elevated cholesterol (hypercholesterolemia)
- High cholesterol increases cardiovascular disease risk
- Influences drug selection for heart health management

### 5. **Na_to_K** (Sodium to Potassium Ratio) (float)

- Ratio of sodium (Na) to potassium (K) concentration in blood
- Continuous numerical value
- Typically ranges from approximately 6 to 38
- Higher ratios may indicate sodium retention or potassium deficiency
- Strong predictor for certain drug prescriptions

## Target Variable (Drug Classes)

### **Drug** (categorical - multi-class)

classes: DrugY, drugX, drugA, drugB, drugC
