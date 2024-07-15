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

