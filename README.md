# power-consumption

**DataSource**<br>
https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014

**Description**<br>
From the title, the time range covered by this data source is from 2011 to 2014.
Values are in kW of each 15 min. To convert values in kWh values must be divided by 4.

**Problem Assumption**<br>
Predict the total electricity consumption of units MT_004, MT_005, and MT_006 in the next month.<br>

Assuming the current time is 22:00 on May 18th, does 'predicting the consumption for the next month' refer to predicting the total electricity consumption from 22:00 on May 18th to 22:00 on June 18th? Or does it refer to predicting the total electricity consumption from 00:00 on June 1st to the end of June? However, the second approach would result in predicting the data only at the end of each month, which is less flexible.<br>

If it is the second case, the data can be compressed to 1/2880. However, if it is the first case, this approach cannot be used.<br>


