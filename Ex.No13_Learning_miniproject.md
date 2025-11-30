# Ex.No: 13 Learning – Use Supervised Learning  

#### DATE:                         
#### NAME: DHARUN KUMAR K
#### REGISTER NUMBER: 212223060051

## AIM: 
To write a program to train the classifier for Exoplanet Detection

## Algorithm:
1.  **Start**
2.  **Import Libraries:** Load all necessary Python libraries, including `pandas`, `numpy`, `matplotlib`, `seaborn`, `sklearn` (for models, metrics, PCA, and scaling), and `imblearn` (for SMOTE).
3.  **Load Data:** Read the `nasa exoplanet data.csv` file into a pandas DataFrame.
4.  **Initial Preprocessing:**
    * Handle any missing values (e.g., using `fillna(0)`).
    * Encode the target variable `LABEL` from {2, 1} to {1, 0}, where 1 represents an exoplanet host and 0 represents a non-host.
5.  **Data Splitting:**
    * Separate the data into features (X, all `FLUX` columns) and the target label (y, the `LABEL` column).
    * Split the *original, unprocessed* data into a training set and a testing set (e.g., `train_X`, `test_X`, `train_y`, `test_y`). This is critical to prevent data leakage.
6.  **Handle Class Imbalance (on Training Data only):**
    * Initialize the `SMOTE` (Synthetic Minority Oversampling Technique) object.
    * Apply `fit_resample` *only* on the training data (`train_X`, `train_y`) to create a balanced training set (`train_X_resampled`, `train_y_resampled`).
7.  **Feature Scaling (Fit on Train, Transform on Both):**
    * Initialize a `StandardScaler`.
    * Fit the scaler on the resampled training data (`train_X_resampled`) using `fit_transform`.
    * Transform the test data (`test_X`) using the *same fitted scaler* with `.transform()`.
8.  **Dimensionality Reduction (Fit on Train, Transform on Both):**
    * Initialize `PCA` (Principal Component Analysis) with the desired number of components (e.g., `n_components=23`).
    * Fit the PCA model on the *scaled* training data (`train_X_scaled`) using `fit_transform`.
    * Transform the *scaled* test data (`test_X_scaled`) using the *same fitted PCA model* with `.transform()`.
9.  **Model Training:**
    * Initialize the desired classifier (e.g., `KNeighborsClassifier`, `LogisticRegression`, `DecisionTreeClassifier`).
    * Train the classifier by calling the `.fit()` method on the processed training data (`train_X_pca`, `train_y_resampled`).
10. **Model Evaluation:**
    * Use the trained model to make predictions on the processed test data (`test_X_pca`) using the `.predict()` method.
    * Calculate and display performance metrics:
        * Accuracy Score
        * Confusion Matrix
        * Classification Report (Precision, Recall, F1-Score)
        * ROC Curve and AUC Score
11. **Result:** Analyze the metrics to assess the model's performance in identifying exoplanets.

We will use data science techniques and machine learning to predict potential exoplanets in star systems using light intensity curves data derived from observations made by the NASA Kepler space telescope.

![image](https://github.com/user-attachments/assets/b5174068-da61-474c-bd58-4487fdddf0af)


## Introduction

Exoplanets are planets located outside our solar system. These exoplanets are diverse in size and orbit where some are as large as giants closely orbiting their stars, while others are icy or rocky. 

A key focus of astronomical research is finding Earth-like exoplanets in habitable zones, areas around stars where conditions are just right for liquid water to exist. The search for exoplanets, planets beyond our solar system, is driven by questions about their existence, diversity, and the potential for life. 


### The Kepler Space Telescope
The Kepler mission marked a significant breakthrough in exoplanet discovery. Prior to Kepler, only a few exoplanets were known. Kepler, using the Transit method, dramatically increased this number by continuously monitoring star brightness.


### NASA's Transit Method

Unlike planets in our solar system that reflect sunlight, exoplanets are too distant and dim to be observed directly. Scientists at NASA use the Transit method to detect these distant worlds. 

This method involves observing stars for tiny dips in brightness, which occur when a planet crosses in front of a star. By analyzing these brightness changes, astronomers can deduce the existence, size, and orbit of exoplanets.

![Transit](https://github.com/user-attachments/assets/4bbb17e3-85be-42dd-9928-14bc6e9db2ce)


## Dataset Description

The [Keplar Dataset](https://www.kaggle.com/datasets/keplersmachines/kepler-labelled-time-series-data), publicly available from NASA, includes flux readings from over 3000 stars, each labeled as either housing an exoplanet or not. We will be analysing this data from the Kepler mission to identify potentially habitable exoplanets. Each star has a binary label of 2 or 1. 2 indicated that that the star is confirmed to have at least one exoplanet in orbit; some observations are in fact multi-planet systems.

As you can imagine, planets themselves do not emit light, but the stars that they orbit do. If said star is watched over several months or years, there may be a regular 'dimming' of the flux (the light intensity). This is evidence that there may be an orbiting body around the star; such a star could be considered to be a 'candidate' system. Further study of our candidate system, for example by a satellite that captures light at a different wavelength, could solidify the belief that the candidate can in fact be 'confirmed'.

<img width="649" height="299" alt="light-curves-star" src="https://github.com/user-attachments/assets/11d405e7-d13b-45d0-a7b6-80b9213f75e5" />


In the above diagram, a star is orbited by a planet. At `t = 2:30`, the starlight intensity drops because it is partially obscured by the planet, given our position. The starlight rises back to its original value at `t = 5:00`. The graph in each box shows the measured flux (light intensity) at each time interval.

## 1. Project Setup and Configuration

### 1.1 Importing Libraries


```python
# Importing Libraries
import math
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import ndimage
from imblearn.over_sampling import SMOTE

3197 from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import recall_score, precision_score, classification_report,accuracy_score,confusion_matrix, roc_curve, auc
```


```python
# Configurations
warnings.filterwarnings('ignore')
plt.style.use('fivethirtyeight')
```

### 1.2 Loading Dataset into Pandas DataFrame


```python
exoplanet_data = pd.read_csv('./nasa exoplanet data.csv').fillna(0)
exoplanet_data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LABEL</th>
      <th>FLUX.1</th>
      <th>FLUX.2</th>
      <th>FLUX.3</th>
      <th>FLUX.4</th>
      <th>FLUX.5</th>
      <th>FLUX.6</th>
      <th>FLUX.7</th>
      <th>FLUX.8</th>
      <th>FLUX.9</th>
      <th>...</th>
      <th>FLUX.3188</th>
      <th>FLUX.3189</th>
      <th>FLUX.3190</th>
      <th>FLUX.3191</th>
      <th>FLUX.3192</th>
      <th>FLUX.3193</th>
      <th>FLUX.3194</th>
      <th>FLUX.3195</th>
      <th>FLUX.3196</th>
      <th>FLUX.3197</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>93.85</td>
      <td>83.81</td>
      <td>20.10</td>
      <td>-26.98</td>
      <td>-39.56</td>
      <td>-124.71</td>
      <td>-135.18</td>
      <td>-96.27</td>
      <td>-79.89</td>
      <td>...</td>
      <td>-78.07</td>
      <td>-102.15</td>
      <td>-102.15</td>
      <td>25.13</td>
      <td>48.57</td>
      <td>92.54</td>
      <td>39.32</td>
      <td>61.42</td>
      <td>5.08</td>
      <td>-39.54</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>-38.88</td>
      <td>-33.83</td>
      <td>-58.54</td>
      <td>-40.09</td>
      <td>-79.31</td>
      <td>-72.81</td>
      <td>-86.55</td>
      <td>-85.33</td>
      <td>-83.97</td>
      <td>...</td>
      <td>-3.28</td>
      <td>-32.21</td>
      <td>-32.21</td>
      <td>-24.89</td>
      <td>-4.86</td>
      <td>0.76</td>
      <td>-11.70</td>
      <td>6.46</td>
      <td>16.00</td>
      <td>19.93</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>532.64</td>
      <td>535.92</td>
      <td>513.73</td>
      <td>496.92</td>
      <td>456.45</td>
      <td>466.00</td>
      <td>464.50</td>
      <td>486.39</td>
      <td>436.56</td>
      <td>...</td>
      <td>-71.69</td>
      <td>13.31</td>
      <td>13.31</td>
      <td>-29.89</td>
      <td>-20.88</td>
      <td>5.06</td>
      <td>-11.80</td>
      <td>-28.91</td>
      <td>-70.02</td>
      <td>-96.67</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>326.52</td>
      <td>347.39</td>
      <td>302.35</td>
      <td>298.13</td>
      <td>317.74</td>
      <td>312.70</td>
      <td>322.33</td>
      <td>311.31</td>
      <td>312.42</td>
      <td>...</td>
      <td>5.71</td>
      <td>-3.73</td>
      <td>-3.73</td>
      <td>30.05</td>
      <td>20.03</td>
      <td>-12.67</td>
      <td>-8.77</td>
      <td>-17.31</td>
      <td>-17.35</td>
      <td>13.98</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>-1107.21</td>
      <td>-1112.59</td>
      <td>-1118.95</td>
      <td>-1095.10</td>
      <td>-1057.55</td>
      <td>-1034.48</td>
      <td>-998.34</td>
      <td>-1022.71</td>
      <td>-989.57</td>
      <td>...</td>
      <td>-594.37</td>
      <td>-401.66</td>
      <td>-401.66</td>
      <td>-357.24</td>
      <td>-443.76</td>
      <td>-438.54</td>
      <td>-399.71</td>
      <td>-384.65</td>
      <td>-411.79</td>
      <td>-510.54</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>5082</th>
      <td>1</td>
      <td>-91.91</td>
      <td>-92.97</td>
      <td>-78.76</td>
      <td>-97.33</td>
      <td>-68.00</td>
      <td>-68.24</td>
      <td>-75.48</td>
      <td>-49.25</td>
      <td>-30.92</td>
      <td>...</td>
      <td>139.95</td>
      <td>147.26</td>
      <td>156.95</td>
      <td>155.64</td>
      <td>156.36</td>
      <td>151.75</td>
      <td>-24.45</td>
      <td>-17.00</td>
      <td>3.23</td>
      <td>19.28</td>
    </tr>
    <tr>
      <th>5083</th>
      <td>1</td>
      <td>989.75</td>
      <td>891.01</td>
      <td>908.53</td>
      <td>851.83</td>
      <td>755.11</td>
      <td>615.78</td>
      <td>595.77</td>
      <td>458.87</td>
      <td>492.84</td>
      <td>...</td>
      <td>-26.50</td>
      <td>-4.84</td>
      <td>-76.30</td>
      <td>-37.84</td>
      <td>-153.83</td>
      <td>-136.16</td>
      <td>38.03</td>
      <td>100.28</td>
      <td>-45.64</td>
      <td>35.58</td>
    </tr>
    <tr>
      <th>5084</th>
      <td>1</td>
      <td>273.39</td>
      <td>278.00</td>
      <td>261.73</td>
      <td>236.99</td>
      <td>280.73</td>
      <td>264.90</td>
      <td>252.92</td>
      <td>254.88</td>
      <td>237.60</td>
      <td>...</td>
      <td>-26.82</td>
      <td>-53.89</td>
      <td>-48.71</td>
      <td>30.99</td>
      <td>15.96</td>
      <td>-3.47</td>
      <td>65.73</td>
      <td>88.42</td>
      <td>79.07</td>
      <td>79.43</td>
    </tr>
    <tr>
      <th>5085</th>
      <td>1</td>
      <td>3.82</td>
      <td>2.09</td>
      <td>-3.29</td>
      <td>-2.88</td>
      <td>1.66</td>
      <td>-0.75</td>
      <td>3.85</td>
      <td>-0.03</td>
      <td>3.28</td>
      <td>...</td>
      <td>10.86</td>
      <td>-3.23</td>
      <td>-5.10</td>
      <td>-4.61</td>
      <td>-9.82</td>
      <td>-1.50</td>
      <td>-4.65</td>
      <td>-14.55</td>
      <td>-6.41</td>
      <td>-2.55</td>
    </tr>
    <tr>
      <th>5086</th>
      <td>1</td>
      <td>323.28</td>
      <td>306.36</td>
      <td>293.16</td>
      <td>287.67</td>
      <td>249.89</td>
      <td>218.30</td>
      <td>188.86</td>
      <td>178.93</td>
      <td>118.93</td>
      <td>...</td>
      <td>71.19</td>
      <td>0.97</td>
      <td>55.20</td>
      <td>-1.63</td>
      <td>-5.50</td>
      <td>-25.33</td>
      <td>-41.31</td>
      <td>-16.72</td>
      <td>-14.09</td>
      <td>27.82</td>
    </tr>
  </tbody>
</table>
<p>5087 rows × 3198 columns</p>
</div>



## 2. Exploratory Data Analysis

### 2.1 Initial Data Analysis and Feature Engineering


```python
exoplanet_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LABEL</th>
      <th>FLUX.1</th>
      <th>FLUX.2</th>
      <th>FLUX.3</th>
      <th>FLUX.4</th>
      <th>FLUX.5</th>
      <th>FLUX.6</th>
      <th>FLUX.7</th>
      <th>FLUX.8</th>
      <th>FLUX.9</th>
      <th>...</th>
      <th>FLUX.3188</th>
      <th>FLUX.3189</th>
      <th>FLUX.3190</th>
      <th>FLUX.3191</th>
      <th>FLUX.3192</th>
      <th>FLUX.3193</th>
      <th>FLUX.3194</th>
      <th>FLUX.3195</th>
      <th>FLUX.3196</th>
      <th>FLUX.3197</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>93.85</td>
      <td>83.81</td>
      <td>20.10</td>
      <td>-26.98</td>
      <td>-39.56</td>
      <td>-124.71</td>
      <td>-135.18</td>
      <td>-96.27</td>
      <td>-79.89</td>
      <td>...</td>
      <td>-78.07</td>
      <td>-102.15</td>
      <td>-102.15</td>
      <td>25.13</td>
      <td>48.57</td>
      <td>92.54</td>
      <td>39.32</td>
      <td>61.42</td>
      <td>5.08</td>
      <td>-39.54</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>-38.88</td>
      <td>-33.83</td>
      <td>-58.54</td>
      <td>-40.09</td>
      <td>-79.31</td>
      <td>-72.81</td>
      <td>-86.55</td>
      <td>-85.33</td>
      <td>-83.97</td>
      <td>...</td>
      <td>-3.28</td>
      <td>-32.21</td>
      <td>-32.21</td>
      <td>-24.89</td>
      <td>-4.86</td>
      <td>0.76</td>
      <td>-11.70</td>
      <td>6.46</td>
      <td>16.00</td>
      <td>19.93</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>532.64</td>
      <td>535.92</td>
      <td>513.73</td>
      <td>496.92</td>
      <td>456.45</td>
      <td>466.00</td>
      <td>464.50</td>
      <td>486.39</td>
      <td>436.56</td>
      <td>...</td>
      <td>-71.69</td>
      <td>13.31</td>
      <td>13.31</td>
      <td>-29.89</td>
      <td>-20.88</td>
      <td>5.06</td>
      <td>-11.80</td>
      <td>-28.91</td>
      <td>-70.02</td>
      <td>-96.67</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>326.52</td>
      <td>347.39</td>
      <td>302.35</td>
      <td>298.13</td>
      <td>317.74</td>
      <td>312.70</td>
      <td>322.33</td>
      <td>311.31</td>
      <td>312.42</td>
      <td>...</td>
      <td>5.71</td>
      <td>-3.73</td>
      <td>-3.73</td>
      <td>30.05</td>
      <td>20.03</td>
      <td>-12.67</td>
      <td>-8.77</td>
      <td>-17.31</td>
      <td>-17.35</td>
      <td>13.98</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>-1107.21</td>
      <td>-1112.59</td>
      <td>-1118.95</td>
      <td>-1095.10</td>
      <td>-1057.55</td>
      <td>-1034.48</td>
      <td>-998.34</td>
      <td>-1022.71</td>
      <td>-989.57</td>
      <td>...</td>
      <td>-594.37</td>
      <td>-401.66</td>
      <td>-401.66</td>
      <td>-357.24</td>
      <td>-443.76</td>
      <td>-438.54</td>
      <td>-399.71</td>
      <td>-384.65</td>
      <td>-411.79</td>
      <td>-510.54</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 3198 columns</p>
</div>




```python
# Label Encoding of Target Feature
categ = {2: 1, 1: 0}
exoplanet_data.LABEL = [categ[item] for item in exoplanet_data.LABEL]
```


```python
print("Dataset Shape: ", exoplanet_data.shape)

print("Dataset Description: ")
exoplanet_data.describe()
```

    Dataset Shape:  (5087, 3198)
    Dataset Description: 





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LABEL</th>
      <th>FLUX.1</th>
      <th>FLUX.2</th>
      <th>FLUX.3</th>
      <th>FLUX.4</th>
      <th>FLUX.5</th>
      <th>FLUX.6</th>
      <th>FLUX.7</th>
      <th>FLUX.8</th>
      <th>FLUX.9</th>
      <th>...</th>
      <th>FLUX.3188</th>
      <th>FLUX.3189</th>
      <th>FLUX.3190</th>
      <th>FLUX.3191</th>
      <th>FLUX.3192</th>
      <th>FLUX.3193</th>
      <th>FLUX.3194</th>
      <th>FLUX.3195</th>
      <th>FLUX.3196</th>
      <th>FLUX.3197</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>5087.000000</td>
      <td>5.087000e+03</td>
      <td>5.087000e+03</td>
      <td>5.087000e+03</td>
      <td>5.087000e+03</td>
      <td>5.087000e+03</td>
      <td>5.087000e+03</td>
      <td>5.087000e+03</td>
      <td>5.087000e+03</td>
      <td>5.087000e+03</td>
      <td>...</td>
      <td>5.087000e+03</td>
      <td>5.087000e+03</td>
      <td>5.087000e+03</td>
      <td>5.087000e+03</td>
      <td>5.087000e+03</td>
      <td>5.087000e+03</td>
      <td>5087.000000</td>
      <td>5087.000000</td>
      <td>5087.000000</td>
      <td>5087.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.007273</td>
      <td>1.445054e+02</td>
      <td>1.285778e+02</td>
      <td>1.471348e+02</td>
      <td>1.561512e+02</td>
      <td>1.561477e+02</td>
      <td>1.469646e+02</td>
      <td>1.168380e+02</td>
      <td>1.144983e+02</td>
      <td>1.228639e+02</td>
      <td>...</td>
      <td>3.485578e+02</td>
      <td>4.956476e+02</td>
      <td>6.711211e+02</td>
      <td>7.468790e+02</td>
      <td>6.937372e+02</td>
      <td>6.553031e+02</td>
      <td>-494.784966</td>
      <td>-544.594264</td>
      <td>-440.239100</td>
      <td>-300.536399</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.084982</td>
      <td>2.150669e+04</td>
      <td>2.179717e+04</td>
      <td>2.191309e+04</td>
      <td>2.223366e+04</td>
      <td>2.308448e+04</td>
      <td>2.410567e+04</td>
      <td>2.414109e+04</td>
      <td>2.290691e+04</td>
      <td>2.102681e+04</td>
      <td>...</td>
      <td>2.864786e+04</td>
      <td>3.551876e+04</td>
      <td>4.349963e+04</td>
      <td>4.981375e+04</td>
      <td>5.087103e+04</td>
      <td>5.339979e+04</td>
      <td>17844.469520</td>
      <td>17722.339334</td>
      <td>16273.406292</td>
      <td>14459.795577</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>-2.278563e+05</td>
      <td>-3.154408e+05</td>
      <td>-2.840018e+05</td>
      <td>-2.340069e+05</td>
      <td>-4.231956e+05</td>
      <td>-5.975521e+05</td>
      <td>-6.724046e+05</td>
      <td>-5.790136e+05</td>
      <td>-3.973882e+05</td>
      <td>...</td>
      <td>-3.240480e+05</td>
      <td>-3.045540e+05</td>
      <td>-2.933140e+05</td>
      <td>-2.838420e+05</td>
      <td>-3.288214e+05</td>
      <td>-5.028894e+05</td>
      <td>-775322.000000</td>
      <td>-732006.000000</td>
      <td>-700992.000000</td>
      <td>-643170.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>-4.234000e+01</td>
      <td>-3.952000e+01</td>
      <td>-3.850500e+01</td>
      <td>-3.505000e+01</td>
      <td>-3.195500e+01</td>
      <td>-3.338000e+01</td>
      <td>-2.813000e+01</td>
      <td>-2.784000e+01</td>
      <td>-2.683500e+01</td>
      <td>...</td>
      <td>-1.760000e+01</td>
      <td>-1.948500e+01</td>
      <td>-1.757000e+01</td>
      <td>-2.076000e+01</td>
      <td>-2.226000e+01</td>
      <td>-2.440500e+01</td>
      <td>-26.760000</td>
      <td>-24.065000</td>
      <td>-21.135000</td>
      <td>-19.820000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000000</td>
      <td>-7.100000e-01</td>
      <td>-8.900000e-01</td>
      <td>-7.400000e-01</td>
      <td>-4.000000e-01</td>
      <td>-6.100000e-01</td>
      <td>-1.030000e+00</td>
      <td>-8.700000e-01</td>
      <td>-6.600000e-01</td>
      <td>-5.600000e-01</td>
      <td>...</td>
      <td>2.600000e+00</td>
      <td>2.680000e+00</td>
      <td>3.050000e+00</td>
      <td>3.590000e+00</td>
      <td>3.230000e+00</td>
      <td>3.500000e+00</td>
      <td>-0.680000</td>
      <td>0.360000</td>
      <td>0.900000</td>
      <td>1.430000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.000000</td>
      <td>4.825500e+01</td>
      <td>4.428500e+01</td>
      <td>4.232500e+01</td>
      <td>3.976500e+01</td>
      <td>3.975000e+01</td>
      <td>3.514000e+01</td>
      <td>3.406000e+01</td>
      <td>3.170000e+01</td>
      <td>3.045500e+01</td>
      <td>...</td>
      <td>2.211000e+01</td>
      <td>2.235000e+01</td>
      <td>2.639500e+01</td>
      <td>2.909000e+01</td>
      <td>2.780000e+01</td>
      <td>3.085500e+01</td>
      <td>18.175000</td>
      <td>18.770000</td>
      <td>19.465000</td>
      <td>20.280000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>1.439240e+06</td>
      <td>1.453319e+06</td>
      <td>1.468429e+06</td>
      <td>1.495750e+06</td>
      <td>1.510937e+06</td>
      <td>1.508152e+06</td>
      <td>1.465743e+06</td>
      <td>1.416827e+06</td>
      <td>1.342888e+06</td>
      <td>...</td>
      <td>1.779338e+06</td>
      <td>2.379227e+06</td>
      <td>2.992070e+06</td>
      <td>3.434973e+06</td>
      <td>3.481220e+06</td>
      <td>3.616292e+06</td>
      <td>288607.500000</td>
      <td>215972.000000</td>
      <td>207590.000000</td>
      <td>211302.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 3198 columns</p>
</div>




```python
exoplanet_data.isnull().sum()
```




    LABEL        0
    FLUX.1       0
    FLUX.2       0
    FLUX.3       0
    FLUX.4       0
                ..
    FLUX.3193    0
    FLUX.3194    0
    FLUX.3195    0
    FLUX.3196    0
    FLUX.3197    0
    Length: 3198, dtype: int64



<div class="alert alert-block alert-info">Initial Reading Analysis: Everything appears to be in order at this stage. Our label values are neatly categorized as 0 and 1, and there are no null values to contend with.</div> 

### 2.2 Geting an idea about the Class Distribution


```python
print(exoplanet_data['LABEL'].value_counts())

fig, ax = plt.subplots(1, 2, figsize=(18,6))
sns.countplot(x = 'LABEL', data=exoplanet_data, palette = "Set2", ax = ax[0])
ax[0].set_xlabel('')
exoplanet_data['LABEL'].value_counts().plot.pie(explode = [0,0.1], autopct ='%1.1f%%',ax = ax[1], shadow = True)
fig.suptitle('0: Not Exoplanet; 1: Exoplanet\n', fontsize = 24, fontweight = 'bold')
```

    LABEL
    0    5050
    1      37
    Name: count, dtype: int64





    Text(0.5, 0.98, '0: Not Exoplanet; 1: Exoplanet\n')




    

    
<img width="1558" height="589" alt="exoplanet detection_20_2" src="https://github.com/user-attachments/assets/3c906ef4-786a-4ee5-a6ed-f0657040a2e8" />


<div class="alert alert-block alert-danger">Huge disproportion in the data: 99.3% not exoplanet while only 0.7% is an exoplanet.</div>

### 2.3 Correlation in the data


```python
plt.figure(figsize=(5,5))
sns.heatmap(exoplanet_data.corr())
plt.title('Correlation in the data')
plt.show()
```


    
<img width="584" height="564" alt="exoplanet detection_23_0" src="https://github.com/user-attachments/assets/6f61533f-632b-44e0-8569-87b3171cd651" />

    


<div class="alert alert-block alert-warning">Analysis: The correlation matrix doesn't provide much insight in this case. Since all the variables are fluxes, they represent independent measurements.</div>

### 2.4 Investigating Flux


```python
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

train_y=exoplanet_data[exoplanet_data['LABEL'] == 1]
train_n=exoplanet_data[exoplanet_data['LABEL'] < 1]
train_t_n=train_n.iloc[:,1:].T
train_t_y=train_y.iloc[:,1:].T
train_t_y.head(1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>27</th>
      <th>28</th>
      <th>29</th>
      <th>30</th>
      <th>31</th>
      <th>32</th>
      <th>33</th>
      <th>34</th>
      <th>35</th>
      <th>36</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>FLUX.1</th>
      <td>93.85</td>
      <td>-38.88</td>
      <td>532.64</td>
      <td>326.52</td>
      <td>-1107.21</td>
      <td>211.1</td>
      <td>9.34</td>
      <td>238.77</td>
      <td>-103.54</td>
      <td>-265.91</td>
      <td>...</td>
      <td>124.39</td>
      <td>-63.5</td>
      <td>31.29</td>
      <td>-472.5</td>
      <td>194.82</td>
      <td>26.96</td>
      <td>43.07</td>
      <td>-248.23</td>
      <td>22.82</td>
      <td>26.24</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 37 columns</p>
</div>




```python
# Flux Variations of Non Exoplanets Stars
fig = make_subplots(rows=2, cols=2,subplot_titles=("Flux variation of star 37", "Flux variation of star 5086", 
                                                   "Flux variation of star 3000", "Flux variation of star 3001"))
fig.add_trace(
    go.Scatter(y=train_t_n[37], x=train_t_n.index),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(y=train_t_n[5086], x=train_t_n.index),
    row=1, col=2
)
fig.add_trace(
    go.Scatter(y=train_t_n[3000], x=train_t_n.index),
    row=2, col=1
)
fig.add_trace(
    go.Scatter(y=train_t_n[3001], x=train_t_n.index),
    row=2, col=2
)
fig.update_layout(height=600, width=800, title_text="Flux Variations of Non Exoplanets Stars",showlegend=False)
fig.show()
```


    
<img width="890" height="600" alt="exoplanet detection_27_0" src="https://github.com/user-attachments/assets/cf255240-6b01-4d66-be22-0cea896d31d0" />

    



```python
# Flux Variations of Exoplanets Stars
fig = make_subplots(rows=2, cols=2,subplot_titles=("Flux variation of star 0", "Flux variation of star 1", 
                                                   "Flux variation of star 35", "Flux variation of star 36"))
fig.add_trace(
    go.Scatter(y=train_t_y[0], x=train_t_y.index),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(y=train_t_y[1], x=train_t_y.index),
    row=1, col=2
)
fig.add_trace(
    go.Scatter(y=train_t_y[35], x=train_t_y.index),
    row=2, col=1
)
fig.add_trace(
    go.Scatter(y=train_t_y[36], x=train_t_y.index),
    row=2, col=2
)
fig.update_layout(height=600, width=800, title_text="Flux Variations of Exoplanets Stars",showlegend=False)
```


    

  <img width="890" height="600" alt="exoplanet detection_28_0" src="https://github.com/user-attachments/assets/4c893724-0d84-41a5-967f-5def388435e5" />
  


<div class="alert alert-block alert-success">We see clear periodic motion: We still see clear anomalies from detection error, but there is periodic motion evident in all the plots. Even star 35 shows periodic motion, albeit on a smaller amplitude. This is due to the fact that there is a planet orbiting in front of the star periodically, therefore decreasing the flux received. </div>

## 3. Data Preprocessing

### 3.1 Handling Missing Values


```python
sns.heatmap(exoplanet_data.isnull())
```




    <Axes: >


<img width="644" height="527" alt="exoplanet detection_32_1" src="https://github.com/user-attachments/assets/4ec9fd79-65f9-4e6e-bcc3-c7e355305bb0" />


    

    


<div class="alert alert-block alert-info">We can see from the heat map that we dont have any missing values in our dataset.</div>

### 3.2 Outlier Detection and Removal


```python
fig, axes = plt.subplots(1, 5,figsize=(15, 6), sharey=True)
fig.suptitle('Distribution of FLUX')

sns.boxplot(ax=axes[0], data=exoplanet_data, x='LABEL', y='FLUX.1',palette="Set2")
sns.boxplot(ax=axes[1], data=exoplanet_data, x='LABEL', y='FLUX.2',palette="Set2")
sns.boxplot(ax=axes[2], data=exoplanet_data, x='LABEL', y='FLUX.3',palette="Set2")
sns.boxplot(ax=axes[3], data=exoplanet_data, x='LABEL', y='FLUX.4',palette="Set2")
sns.boxplot(ax=axes[4], data=exoplanet_data, x='LABEL', y='FLUX.5',palette="Set2")
```




    <Axes: xlabel='LABEL', ylabel='FLUX.5'>




   <img width="1418" height="618" alt="exoplanet detection_35_1" src="https://github.com/user-attachments/assets/781c1cb0-6401-42bc-a76d-52130b0cdefc" />
 

    



```python
exoplanet_data.drop(exoplanet_data[exoplanet_data['FLUX.1']>250000].index, axis=0, inplace=True)
```

### 3.3 Handling Imbalance using SMOTE (Synthetic Minority Oversampling Technique)


```python
from imblearn.over_sampling import SMOTE
model = SMOTE()
input_features, output_feature = model.fit_resample(exoplanet_data.drop('LABEL',axis=1), exoplanet_data['LABEL'])
output_feature = output_feature.astype('int')
```


```python
output_feature.value_counts()
```




    LABEL
    1    5049
    0    5049
    Name: count, dtype: int64



### 3.4 Data Normalization

**Data Normalization** is a technique often applied as part of data preparation for machine learning. The goal of normalization is to change the values of numeric columns in the dataset to a common scale, without distorting differences in the ranges of values.


```python
input_features = normalized = normalize(input_features)
```

### 3.5 Apply Gaussian Filters

In probability theory, the normal (or Gaussian or Gauss or Laplace–Gauss) distribution is a very common continuous probability distribution. Normal distributions are important in statistics and are often used in the natural and social sciences to represent real-valued random variables whose distributions are not known.


```python
input_features = filtered = ndimage.filters.gaussian_filter(input_features, sigma=10)
```

### 3.6 Feature Scaling

We will use feature scaling so that all the values remain in the comparable range.


```python
std_scaler = StandardScaler()
input_features = scaled = std_scaler.fit_transform(input_features)
```

### 3.7 Dimensionality Reduction using PCA (Principal Component Analysis)


```python
from sklearn.decomposition import PCA
pca = PCA() 
input_features = pca.fit_transform(input_features)
input_features = pca.transform(input_features)
total=sum(pca.explained_variance_)
k=0
current_variance=0
while current_variance/total < 0.90:
    current_variance += pca.explained_variance_[k]
    k=k+1
print(k)
```

    23



```python
pca = PCA(n_components=23)
input_features = pca.fit_transform(input_features)
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') 
plt.title('Exoplanet Dataset Explained Variance')
plt.show()
```


    
<img width="641" height="491" alt="exoplanet detection_51_0" src="https://github.com/user-attachments/assets/7afda289-f873-4392-b7c2-26c2d4faf827" />

    



```python
input_features.shape
```




    (10098, 23)



### 3.8 Splitting into Testing Data and Training Data


```python
train_X, test_X, train_y, test_y = train_test_split(input_features, output_feature, test_size=0.33, random_state=42)
```

## 4. Model Building and Evaluation

In this section, we will build and evaluate several machine learning models. We will start with traditional models and proceed to more advanced models. The performance of these models will be assessed based on their accuracy, precision, recall, and F1-score.

### Model Building


```python
def model(classifier,dtrain_x,dtrain_y,dtest_x,dtest_y):
    classifier.fit(dtrain_x,dtrain_y)

    prediction=classifier.predict(dtest_x)

    print('Validation accuracy of model is', accuracy_score(prediction,dtest_y))
    print ("\nClassification report :\n",(classification_report(dtest_y,prediction)))

    #Confusion matrix
    plt.figure(figsize=(13,10))
    plt.subplot(221)
    sns.heatmap(confusion_matrix(dtest_y,prediction),annot=True,cmap="viridis",fmt = "d",linecolor="k",linewidths=3)
    plt.title("CONFUSION MATRIX",fontsize=20)

    #ROC curve and Area under the curve plotting
    predicting_probabilites = classifier.predict_proba(dtest_x)[:,1]
    fpr,tpr,thresholds = roc_curve(dtest_y,predicting_probabilites)
    plt.subplot(222)
    plt.plot(fpr,tpr,label = ("Area_under the curve :",auc(fpr,tpr)),color = "r")
    plt.plot([1,0],[1,0],linestyle = "dashed",color ="k")
    plt.legend(loc = "best")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title("ROC - CURVE & AREA UNDER CURVE",fontsize=20)
```


```python
from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier()
model(knn_model,train_X,train_y,test_X,test_y)
```

    Validation accuracy of model is 0.9993999399939995
    
    Classification report :
                   precision    recall  f1-score   support
    
               0       1.00      1.00      1.00      1709
               1       1.00      1.00      1.00      1624
    
        accuracy                           1.00      3333
       macro avg       1.00      1.00      1.00      3333
    weighted avg       1.00      1.00      1.00      3333
    



    

<img width="1174" height="470" alt="exoplanet detection_59_1" src="https://github.com/user-attachments/assets/e45f44f4-eadc-4dc6-9506-64a6dbb601c8" />



### Logistic Regression


```python
from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression(solver='lbfgs', max_iter=1000)
model(lr_model,train_X,train_y,test_X,test_y)
```

    Validation accuracy of model is 0.9996999699969997
    
    Classification report :
                   precision    recall  f1-score   support
    
               0       1.00      1.00      1.00      1709
               1       1.00      1.00      1.00      1624
    
        accuracy                           1.00      3333
       macro avg       1.00      1.00      1.00      3333
    weighted avg       1.00      1.00      1.00      3333
    



    
<img width="1174" height="470" alt="exoplanet detection_61_1" src="https://github.com/user-attachments/assets/7a980f8e-96f1-4010-b02e-d9ca653a2b29" />

    



```python
from sklearn.naive_bayes import BernoulliNB
bnb = BernoulliNB()
model(bnb,train_X,train_y,test_X,test_y)
```

    Validation accuracy of model is 0.9996999699969997
    
    Classification report :
                   precision    recall  f1-score   support
    
               0       1.00      1.00      1.00      1709
               1       1.00      1.00      1.00      1624
    
        accuracy                           1.00      3333
       macro avg       1.00      1.00      1.00      3333
    weighted avg       1.00      1.00      1.00      3333
    



    
<img width="1218" height="470" alt="exoplanet detection_62_1" src="https://github.com/user-attachments/assets/777e5329-bbcf-4c3e-9938-da36ec82c960" />

    



```python
from sklearn.tree import DecisionTreeClassifier
ds_model = DecisionTreeClassifier(max_depth=5, random_state=13)
model(ds_model,train_X,train_y,test_X,test_y)
```

    Validation accuracy of model is 0.9996999699969997
    
    Classification report :
                   precision    recall  f1-score   support
    
               0       1.00      1.00      1.00      1709
               1       1.00      1.00      1.00      1624
    
        accuracy                           1.00      3333
       macro avg       1.00      1.00      1.00      3333
    weighted avg       1.00      1.00      1.00      3333
    



    
<img width="1174" height="470" alt="exoplanet detection_63_1" src="https://github.com/user-attachments/assets/18fd30bd-7033-47bf-8d50-5d411da40e46" />

    


<div class="alert alert-block alert-info">In conclusion, this project demonstrated the application of various machine learning models to the task of exoplanet detection using data from the Kepler Space Telescope. 

This study also highlights the potential and challenges of machine learning in astronomical data analysis. Future work could explore more complex models, feature engineering techniques, and larger datasets to further enhance the detection of exoplanets.</div>


### Result:
Thus the system was trained successfully and the prediction was carried out.
