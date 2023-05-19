# power-consumption

## DataSource
https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014

## Description
From the title, the time range covered by this data source is from 2011 to 2014.
Values are in kW of each 15 min. To convert values in kWh values must be divided by 4.

## Problem Assumption
Predict the total electricity consumption of units MT_004, MT_005, and MT_006 in the next month.<br>

Assuming the current time is 22:00 on May 18th, does 'predicting the consumption for the next month' refer to predicting the total electricity consumption from 22:00 on May 18th to 22:00 on June 18th? Or does it refer to predicting the total electricity consumption from 00:00 on June 1st to the end of June? However, the second approach would result in predicting the data only at the end of each month, which is less flexible.<br>

If it is the second case, the data can be compressed to 1/2880. However, if it is the first case, this approach cannot be used.<br>

## Project Structure
In the directory, saod is the abbreviation for small_amount_of_data.

## Evaluating Indicator
Coefficient of Determination is also known as R-squared (R²), is used to evaluate the goodness-of-fit of a regression model. It represents the proportion of the variance in the target variable that can be explained by the model. It ranges from 0 to 1, where a value closer to 1 indicates a better fit of the model, and a value closer to 0 indicates a poorer fit. There is no fixed threshold to determine the acceptability of the coefficient of determination as it depends on the specific application and characteristics of the data. Generally, a higher coefficient of determination implies that the model can better explain the variability in the target variable, while a lower coefficient of determination indicates a weaker explanatory power of the model.