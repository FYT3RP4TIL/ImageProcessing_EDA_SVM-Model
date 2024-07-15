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

## Eigen Face and Principle Component Analysis

![image](https://github.com/user-attachments/assets/7780b4c1-b743-49be-a9e9-d4147a6738e0)

### PCA Algorithm 
An approach to face recognition that uses dimensionality reduction and linear algebra concepts to recognize faces. This approach is computationally less expensive and easy to implement and thus used in various applications at that time such as handwritten recognition, lip-reading, medical image analysis, etc.It uses Eigenvalues and EigenVectors to reduce dimensionality and project a training sample/data on small feature space. Let’s look at the algorithm in more detail (in a face recognition perspective).

## Steps

### 1. Convert Images to Vectors

Convert the images into vectors of size $N^2$ :  $x_1, x_2, x_3, \ldots, x_m$

<p align="center">
  <img src="https://github.com/user-attachments/assets/0ccbdc75-75d5-44eb-ad25-6724a7d64a4e"/>
</p>

### 2. Calculate the Average Face

Calculate the average of all face vectors and subtract it from each vector:

<p align="center">
  <img src="https://github.com/user-attachments/assets/578aefb3-614c-4500-ac2a-34ab337b0d65"/>
</p>

$$\psi = \dfrac{1}{m}\sum_{i=1}^{m}x_i$$
$$a_i = x_i - \psi$$

### 3. Create Matrix A

Construct matrix $A$ of size $N^2 \times M$:

$$A = \begin{bmatrix} a_1 & a_2 & a_3 & \ldots & a_m \end{bmatrix}$$

### 4. Compute the Covariance Matrix

Calculate the covariance matrix by multiplying $A$ with $A^T$:

$$\text{Cov} = A^T A$$

### 5. Calculate Eigenvalues and Eigenvectors

Find the eigenvalues and eigenvectors of the covariance matrix:

$$A^T A \nu_i = \lambda_i \nu_i$$
$$AA^T A \nu_i = \lambda_i A \nu_i$$
$$C'u_i = \lambda_i u_i$$
$$\text{where, } C' = AA^T \text{ and } u_i = A \nu_i$$

### 6. Map Eigenvectors

Calculate eigenvectors of the reduced covariance matrix and map them into $C'$:

$$u_i = A \nu_i$$

### 7. Select Top K Eigenvectors

Select the $K$ eigenvectors of $C'$ corresponding to the $K$ largest eigenvalues. These eigenvectors have size $N^2$.

### 8. Represent Faces Using Eigenfaces

Use the eigenvectors obtained in the previous step to represent each face vector as a linear combination of the best $K$ eigenvectors:

<p align="center">
  <img src="https://github.com/user-attachments/assets/a2191165-26eb-4cfc-a6cd-f3147c709ea8"/>
</p>

$$x_i - \psi = \sum_{j=1}^{K} w_j u_j$$

These $u_j$ are called Eigenfaces.

### 9. Represent Training Faces

Take the coefficient of Eigenfaces and represent the training faces in the form of a vector of those coefficients:

$$x_i = \begin{bmatrix} w_1^i \\ w_2^i \\ w_3^i \\ \ldots \\ w_k^i \end{bmatrix}$$

## Testing/Detection Algorithm

<p align="center">
  <img src="https://github.com/user-attachments/assets/90bf5cea-2b17-4976-93c1-95112832698c"/>
</p>

### 1. Preprocess Test Images

Given an unknown face $y$, preprocess the face to make it centered in the image and have the same dimensions as the training face. Subtract the average face $\psi$:

<p align="center">
  <img src="https://github.com/user-attachments/assets/c56f587c-0669-4184-add0-c7608808abae"/>
</p>

$$\phi = y - \psi$$

### 2. Project into Eigenspace

Project the normalized vector into eigenspace to obtain the linear combination of Eigenfaces:

$$\phi = \sum_{i=1}^{k} w_i u_i$$

### 3. Generate Coefficient Vector

Generate the vector of coefficients:

$$\Omega = \begin{bmatrix} w_1 \\ w_2 \\ w_3 \\ \ldots \\ w_k \end{bmatrix}$$

### 4. Calculate Minimum Distance

Calculate the minimum distance between the training vectors and testing vectors:

$$e_r = \min_{l} \left \| \Omega - \Omega_l \right \|$$

If $e_r$ is below the tolerance level $T_r$, it is recognized with $l$ face from the training image; otherwise, the face is not matched with any faces in the training set.

## Advantages

- Easy to implement and computationally less expensive.
- No knowledge of the image (such as facial features) is required (except id).

## Limitations

- Proper centered face is required for training/testing.
- The algorithm is sensitive to lighting, shadows, and the scale of the face in the image.
- A front view of the face is required for this algorithm to work properly.

## Reasults :

### Explained Variance and Cumulative Variance Plot :

![image](https://github.com/user-attachments/assets/8b5ce132-fef4-4f24-90e1-60e6b86ea45e)

Out of the principle components elbow plot shows that a good number of components would come around 15 but in cumulative plot at 15 components we might get around 50% explained variance. Which is not good as for a good model the explained variance should be >= 80%. Hence, we should take 50 principle components which gives us 80% coverage of explained variance.

Hence, PCA will be trained with 50 principle components. This model will be saved as well as the mean face (here npz format numpyzip - data, pca_50 with mean face via pickle (pca_dict.pickle), eigen images visulized in the notebook(03)).

## SVM (Support Vector Machine) Classifier Model

![image](https://github.com/user-attachments/assets/f810313a-b7b2-4baa-87e2-4c2ed3df58cb)

- Support Vector Machine (SVM) is a powerful machine learning algorithm used for linear or nonlinear classification, regression, and even outlier detection tasks. SVMs can be used for a variety of tasks, such as text classification, image classification, spam detection, handwriting identification, gene expression analysis, face detection, and anomaly detection. SVMs are adaptable and efficient in a variety of applications because they can manage high-dimensional data and nonlinear relationships.

- The main objective of the SVM algorithm is to find the optimal hyperplane in an N-dimensional space that can separate the data points in different classes in the feature space. The hyperplane tries that the margin between the closest points of different classes should be as maximum as possible. The dimension of the hyperplane depends upon the number of features. If the number of input features is two, then the hyperplane is just a line. If the number of input features is three, then the hyperplane becomes a 2-D plane. It becomes difficult to imagine when the number of features exceeds three.

### Hyperparameter Tuning with GridSearchCV

- GridSearchCV acts as a valuable tool for identifying the optimal parameters for a machine learning model. Imagine you have a machine learning model with adjustable settings, known as hyperparameters, that can enhance its performance. GridSearchCV aids in pinpointing the best combination of these hyperparameters automatically.

- You provide GridSearchCV with a set of Scoring parameter to experiment with, and it systematically explores each possible combination. For every combination, it evaluates the model’s performance by testing it on various sections of the dataset to gauge its accuracy.

- After exhaustively trying out all the combinations, GridSearchCV presents you with the combination of settings that yielded the most favorable outcomes. This streamlines the process of fine-tuning your model, ensuring it operates optimally for your specific task without incurring excessive computational expenses.

The grid search provided by GridSearchCV exhaustively generates candidates from a grid of parameter values specified with the param_grid parameter. For instance, the following param_grid

```python 
model_svc = SVC(probability=True)

param_grid = {'C':[0.5,1,10,20,30,50],
             'kernel':['rbf','poly'],
             'gamma':[0.1,0.05,0.01,0.001,0.002,0.005],
             'coef0':[0,1]}

model_grid = GridSearchCV(model_svc,
                          param_grid=param_grid,
                          scoring='accuracy',cv=3,verbose=2)

model_grid.fit(x_train,y_train)
```

```python
model_grid.best_params_
```

Output : 

```python
{'C': 20, 'coef0': 0, 'gamma': 0.002, 'kernel': 'rbf'}
```

## Model Evaluation :

- **Classification Report**
  - Precision, Recall, F1-Score

- **Kappa Score (Good for Multiclass Problems)**
  - -ve (worst model)
  - 0 to 0.5 (bad model)
  - 0.5 to 0.7 (good model)
  - 0.7 to 0.9 (excellent model)
  - 0.9 to 1.0 (perfect model)

- **AUC**
  - Less than 0.5 (worst model)
  - 0.5 to 0.6 (bad model)
  - 0.6 to 0.8 (good model)
  - 0.8 to 0.9 (excellent model)
  - 0.9 to 1.0 (perfect model)
 
### Classification Report

|              | Precision | Recall   | F1-Score | Support |
|--------------|-----------|----------|----------|---------|
| **Female**   | 0.805611  | 0.842767 | 0.823770 | 477.000 |
| **Male**     | 0.794521  | 0.749354 | 0.771277 | 387.000 |
| **Accuracy** | 0.800926  | 0.800926 | 0.800926 | 0.800926|
| **Macro Avg**| 0.800066  | 0.796061 | 0.797524 | 864.000 |
| **Weighted Avg** | 0.800644  | 0.800926 | 0.800258 | 864.000 |

### Additional Metrics

| Metric       | Value         |
|--------------|---------------|
| **Kappa Score** | 0.595314   |
| **AUC**         | 0.796066   |
