# ImageProcessing_EDA_SVM-Model
## Dataset

The dataset used for this project can be found on Kaggle: [Gender Classification Dataset](https://www.kaggle.com/datasets/cashutosh/gender-classification-dataset)

## Exploratory Data Analysis and Image Processing for Gender Classification

After Cropping and saving the dataset Exploratory Data Analysis (EDA) on a gender classification dataset, followed by image processing tasks such as resizing and removing outliers is needed for better model trainin

## Steps

### 1. Distribution of Male and Female

We start by analyzing the distribution of male and female samples in the dataset. The results are visualized using:
- **Bar Chart**
- **Pie Chart**

### 2. Distribution of Image Sizes

Next, we examine the distribution of image sizes within the dataset. This analysis includes:
- **Histogram**: To show the frequency distribution of image sizes.
- **Box Plot**: To summarize the distribution and identify any outliers.
- **Split by Gender**: To see if there are any differences in image size distributions between male and female samples.

### 3. Decision on Image Resizing

Based on the insights obtained from the EDA, we make an informed decision on the width and height to which all images should be resized.

### 4. Removal of Outliers

Images with significantly small sizes that may adversely affect the analysis or model performance are identified and removed.

## EDA Results

### Distribution of Male and Female

![Screenshot 2024-07-15 143706](https://github.com/user-attachments/assets/bd9377d8-88dc-4fc1-ab8f-66fcc92f5e3d)

### Distribution of Image Sizes

#### Histogram and Box Plot

![Screenshot 2024-07-15 143718](https://github.com/user-attachments/assets/5c1a84f3-c7da-4da0-9f9d-e226d6e40a67)

#### Split by Gender

![Screenshot 2024-07-15 143748](https://github.com/user-attachments/assets/ba29197c-7c52-431b-96bb-6e178da1983a)

### Resizing Decision

- 1. We have almost equal distribution of gender (numbers)
- 2. Most of the images are having dimensions more than 60 x 60
- 3. Size of the images lie near 100 x 100

Based on the EDA, the images will be resized to 100 x 100 and images less than 60 x 60 will be filtered out (Outliers Removal).

