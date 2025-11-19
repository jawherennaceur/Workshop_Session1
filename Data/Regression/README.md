# California Housing Dataset Description

## Overview

Housing data from the 1990 California Census
Used for regression to predict median house value

## Goal

Predict median_house_value using housing and demographic features

## Features Description

### 1. **longitude** (float)

- Longitudinal coordinate of the housing block
- Represents the east-west position
- Ranges approximately from -124 to -114
- Western values (more negative) are closer to the Pacific Ocean

### 2. **latitude** (float)

- Latitudinal coordinate of the housing block
- Represents the north-south position
- Ranges approximately from 32 to 42
- Higher values indicate northern California locations

### 3. **housing_median_age** (float)

- Median age of houses in the block (in years)
- Represents how old the housing stock is
- Values typically range from 1 to 52 years
- Higher values indicate older housing developments

### 4. **total_rooms** (float)

- Total number of rooms in all houses within the block
- Includes all living spaces (bedrooms, living rooms, kitchens, etc.)
- Can be used to derive average rooms per household

### 5. **total_bedrooms** (float)

- Total number of bedrooms in all houses within the block
- Subset of total_rooms
- **Note**: This field may contain missing values in the full dataset
- Useful for understanding housing capacity

### 6. **population** (float)

- Total population living in the block
- Represents population density when combined with geographic data
- Helps understand crowding and neighborhood characteristics

### 7. **households** (float)

- Total number of households (occupied housing units) in the block
- A household is defined as a group of people residing together
- Used to calculate average household size and occupancy rates

### 8. **median_income** (float)

- Median income for households in the block
- Measured in tens of thousands of U.S. dollars
- For example: 3.5 represents $35,000
- Strong predictor of house values
- Typically ranges from about 0.5 to 15

### 9. **ocean_proximity** (categorical)

- Categorical variable indicating proximity to the ocean
- Possible values:
  - **NEAR BAY**: Close to San Francisco Bay area
  - **<1H OCEAN**: Less than 1 hour drive to the ocean
  - **INLAND**: Interior locations away from water
  - **NEAR OCEAN**: Close to the Pacific Ocean
  - **ISLAND**: Island locations (very rare)
- Significant impact on property values

## Target Variable

### **median_house_value** (float)

- Median house value for households in the block
- Measured in U.S. dollars
- The variable we want to predict
- Typically ranges from about $15,000 to $500,000+
- Values may be capped at $500,000 in some versions
