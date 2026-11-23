#!/usr/bin/env python
# coding: utf-8

# Brianna Aceves | DSC355 | Professor F. Neugebauer | Final- Project Milestone 5 Predicting Donor Churn for Nonprofit Organizations

# To start off my final project, I would like to add that this project is a continuation of my work from this semester. Notes will advise when my milestone week's were started and completed. 

# This final project contains my week to week submissions that produce Milestone 5.

# In[1]:


#Install missing library in the current Jupyter environment
get_ipython().run_line_magic('pip', 'install seaborn')


# In[2]:


# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# I add to adjust some things for Week 7's work. I needed to reinstall the seaborn library for this to run again. 

# For this exercise, I will use this dataset to build a classification model that predicts the target variable, y. I started by importing the dataset into a Pandas DataFrame. I wanted to look at the first few rows to get a better idea of how the data is organized and what types of values each column contains.

# In[3]:


#load the dataset
df = pd.read_csv("eda_classification.csv")

df.head()


# Next, I wanted to look at the overall structure of the dataset. This should help me identify which variables are numerical and which ones might need to be cleaned up before building the model.

# In[4]:


# Reviewe number of rows and columns
print("Dataset shape:", df.shape)

# Review column names
print("\nColumn names:")
print(df.columns.tolist())

# Review data types
print("\nDataset information:")
df.info()


# In[5]:


#Descriptive statistics for numerical variables
df.describe().T


# Before starting on the model, I checked for missing values and duplicates. The descriptive statistics showed that the dataset has 9,999 rows and 19 columns. Also, from looking at the descriptive statistics, I noticed that some numerical values have pretty large ranges, like x8 going from about -661.76 to 476.91, and several columns have fewer than 9,999 non-null values, meaning there are some missing values I'll need to look into before starting the model.

# In[6]:


#Check missing values in each column
print("Missing values by column:")
print(df.isnull().sum())

print("\nTotal missing values:", df.isnull().sum().sum())


# In[7]:


#Count duplicate rows
print("Duplicate rows:", df.duplicated().sum())


# There are no duplicate rows, but this does show me there are 31 missing values. 

# In[8]:


#Target distribution
#Count each target value
print(df["y"].value_counts(dropna=False))

#Percentage of each class
print("\nTarget percentages:")
print(df["y"].value_counts(normalize=True) * 100)


# In[9]:


#Plot the target distribution
sns.countplot(data=df, x="y")
plt.title("Distribution of Target Variable (y)")
plt.xlabel("Target Class")
plt.ylabel("Number of Observations")
plt.show()


# After looking at the target variable y, I noticed that the two classes are pretty evenly split, with 50.81% being 0 and 49.19% being 1. I want to take a look at the categorical variables to see what kind of values they have and if anything needs to be cleaned up before moving on to the model.

# In[10]:


#Identify categorical columns
categorical_cols = df.select_dtypes(
    include=["object", "string"]
).columns.tolist()

print("Categorical columns:", categorical_cols)

#Review unique values and their counts
for col in categorical_cols:
    print(f"\nColumn: {col}")
    print(df[col].value_counts(dropna=False))


# After looking at the categorical columns, I noticed that some values are written differently even though they mean the same thing, like abbreviations for months or weekdays. I also noticed that some numbers are being treated as text because they include dollar signs or percentages. Next, I wanted to look at the numerical variables a little closer to see how their values are spread out and whether there are any unusual patterns.

# In[11]:


#Numerical Distribution
#Select numerical columns, excluding y
numeric_cols = df.select_dtypes(
    include="number"
).columns.drop("y").tolist()

#Create histograms
df[numeric_cols].hist(
    figsize=(15, 12),
    bins=30,
    edgecolor="black"
)

plt.suptitle("Numerical Variable Distributions")
plt.tight_layout()
plt.show()


# I found a few things that need to be cleaned up before building the model. Some columns have currency amounts or percentages that are being read as text, and others have inconsistent abbreviations or spelling. Fixing these will make the data more consistent and then it can actually be used by the classification model.

# In[12]:


#Create a copy
df_clean = df.copy()

#Remove extra spaces from column names
df_clean.columns = df_clean.columns.str.strip()


# In[13]:


#Check the copy
df_clean.head()


# Instead of modfying the original dataframe, I named the copy df_clean and modified the copy instead. 

# In[14]:


# Review column x7
# Convert x7 currency amounts to numbers
currency = df_clean["x7"].astype("string").str.strip()

# Identify negative amounts in parentheses
negative = currency.str.startswith("(").fillna(False)

# Remove dollar signs, commas, and parentheses
currency = currency.str.replace(
    r"[$,()\s]", "", regex=True
)

# Convert to numerical values
df_clean["x7"] = pd.to_numeric(
    currency, errors="coerce"
)

# Keep negative amounts
df_clean.loc[negative, "x7"] = (
    -df_clean.loc[negative, "x7"].abs()
)

df_clean["x7"].head(10)


# For x7, I noticed that the values included dollar signs, commas, and parentheses, which caused Python to read them as text instead of numbers. I removed the extra formatting and converted the values into numerical data while keeping the negative amounts negative. This should make the column easier to work with when building the classification model.

# In[15]:


#Cleaning percentages on x11
#Remove percentage signs and extra spaces
percent = (
    df_clean["x11"]
    .astype("string")
    .str.replace("%", "", regex=False)
    .str.strip()
)

#Convert percentages into decimals
df_clean["x11"] = (
    pd.to_numeric(percent, errors="coerce") / 100
)

#Display the first 10 cleaned values
df_clean["x11"].head(10)


# Similar to x7, I noticed that x11 had percentage signs, which caused Python to treat the values as text. I removed the percentage signs and converted everything into decimals so the model can recognize them as numerical values.

# In[16]:


#Standardize
#Identify remaining categorical columns
category_cols = df_clean.select_dtypes(
    include=["object", "string"]
).columns.tolist()

#Convert to lowercase and remove extra spaces
for col in category_cols:
    df_clean[col] = (
        df_clean[col]
        .astype("string")
        .str.strip()
        .str.lower()
    )

#Standardize month names
month_map = {
    "jan": "january",
    "feb": "february",
    "mar": "march",
    "apr": "april",
    "jun": "june",
    "jul": "july",
    "aug": "august",
    "sep": "september",
    "sept": "september",
    "sept.": "september",
    "oct": "october",
    "nov": "november",
    "dec": "december"
}

df_clean["x1"] = df_clean["x1"].replace(month_map)

#Standardize weekday names
day_map = {
    "mon": "monday",
    "tue": "tuesday",
    "tues": "tuesday",
    "wed": "wednesday",
    "weds": "wednesday",
    "thur": "thursday",
    "thurs": "thursday",
    "thurday": "thursday",
    "fri": "friday",
    "sat": "saturday",
    "sun": "sunday"
}

df_clean["x14"] = df_clean["x14"].replace(day_map)

#Standardize car brand spelling
df_clean["x13"] = df_clean["x13"].replace({
    "volkswagon": "volkswagen",
    "chrystler": "chrysler"
})

#Review the cleaned categories
for col in category_cols:
    print(f"\nColumn: {col}")
    print(df_clean[col].value_counts(dropna=False))


# I cleaned up the categorical variables because some of the values were written differently. For example, some months and weekdays were abbreviated, and a few categories had spelling differences. I converted everything to lowercase and standardized the names so they would be treated as the same category instead of separate ones.

# In[17]:


#Review updated data types
print("Updated data types:")
print(df_clean.dtypes)

#Check remaining missing values
print("\nMissing values after cleaning:")
print(df_clean.isnull().sum())

#Compare missing values before and after cleaning
print("\nChange in missing values:")
print(
    df_clean.isnull().sum() - df.isnull().sum()
)

#Display first five rows
df_clean.head()


# After cleaning the data, I checked the column types and missing values again to make sure everything worked. Both x7 and x11 are now numerical and the cleaning didn't create any new missing values. There are still a few missing values in some columns but I'll handle those during preprocessing instead of removing the rows.

# In[18]:


#Checking correlations
#Select numerical columns
numerical_data = df_clean.select_dtypes(
    include="number"
)

#Calculate correlations
correlation_matrix = numerical_data.corr()

#Create correlation heatmap
plt.figure(figsize=(14, 10))

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)

plt.title("Correlation Between Numerical Variables")
plt.tight_layout()
plt.show()


# I used a correlation heatmap to see how the numerical variables relate to each other and to the target variable, y. I wanted to get a better idea of whether any features have stronger positive or negative relationships that could help with the classification model.

# In[19]:


#Separate the predictor variables from the target
X = df_clean.drop(columns=["y"]).copy()
y = df_clean["y"]

#Check target values
print("Target values:", y.unique())

#Review dataset shapes
print("Predictor shape:", X.shape)
print("Target shape:", y.shape)


# I've finished cleaning and exploring the data, I separated the predictors from the target variable. X contains the 18 features the model will use to make predictions, while y contains the actual 0 or 1 outcomes. This will help me prepare everything for training and testing.

# In[20]:


#Identify categorical columns
cat_cols = X.select_dtypes(
    include=["object", "string"]
).columns.tolist()

#Convert categorical columns to object type
#use np.nan for missing values
for col in cat_cols:
    X[col] = X[col].astype(object)
    X[col] = X[col].where(
        X[col].notna(), np.nan
    )

#Display categorical columns
print("Categorical columns:", cat_cols)

#Check missing values
print("\nMissing values in predictors:")
print(X.isnull().sum())


# I also made sure the categorical columns were in a format that Scikit-learn could work with. Since there are still a few missing values in the dataset, I'm keeping track of those so they can be filled in during preprocessing. This way, I don't have to delete rows just because they're missing a small amount of information.

# In[21]:


#Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

#Review the sizes
print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

#Check the target distribution
print("\nTraining target percentages:")
print(y_train.value_counts(normalize=True) * 100)

print("\nTesting target percentages:")
print(y_test.value_counts(normalize=True) * 100)


# I split the data into 80% for training and 20% for testing, which gave me 7,999 rows for training and 2,000 for testing. I also used stratification since the original target was almost evenly split between 0 and 1. This helps keep both classes fairly equal when training and testing the model.

# In[22]:


#Identify numerical and categorical features
numeric_features = X_train.select_dtypes(
    include="number"
).columns.tolist()

categorical_features = X_train.select_dtypes(
    include=["object", "string"]
).columns.tolist()

print("Numerical features:", numeric_features)
print("Categorical features:", categorical_features)

#Prepare numerical features
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

#Prepare categorical features
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(
        strategy="most_frequent"
    )),
    ("onehot", OneHotEncoder(
        handle_unknown="ignore"
    ))
])

#Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

print("\nPreprocessing pipeline created successfully!")


# Since the dataset has both numerical and categorical variables, I needed to prepare them differently before training the model. I set up one-hot encoding for the categorical columns and standardization for the numerical ones since their values have pretty different ranges. I also filled in the remaining missing values with the median.

# In[23]:


#Create the logistic regression model
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=1000,
        random_state=42
    ))
])

#Train the model
model.fit(X_train, y_train)

print("Logistic Regression model trained successfully!")


# Now that the features are prepared, I trained a logistic regression model to predict y. I chose logistic regression because the target is binary.

# In[24]:


#Make predictions using the testing data
y_pred = model.predict(X_test)

#Compare predictions to actual values
results = pd.DataFrame({
    "Actual": y_test.to_numpy(),
    "Predicted": y_pred
})

#Check whether the prediction was correct
results["Correct"] = (
    results["Actual"] == results["Predicted"]
)

#Display the first 15 predictions
results.head(15)


# After training the model, I compared the predicted values with the actual values of y to see how well it performed. Looking at the first 15 predictions, the model got 10 correct and 5 incorrect. I want to check the overall accuracy to get a better idea of how well the model performed on the entire testing dataset.

# In[25]:


#Evaluate model
#Calculate overall accuracy
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy:.2%}")

#Display the classification report
print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    zero_division=0
))


# After checking the model's performance, I found that it had an overall accuracy of 51.60%, which is pretty low. Looking at the classification report, the model did a little better predicting 0, with a recall of 0.60 compared to 0.43 for 1. The F1-scores were also fairly low at 0.56 for 0 and 0.47 for 1. Overall, it looks like the model is struggling to identify both classes accurately, especially 1.

# In[26]:


#Create confusion matrix
cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1]
)

#Display the confusion matrix
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[0, 1]
)

disp.plot(cmap="Blues", values_format="d")

plt.title("Logistic Regression Confusion Matrix")
plt.show()


# Looking at the confusion matrix, the model correctly predicted 610 values of 0 and 422 values of 1, but it also made quite a few mistakes. It incorrectly predicted 406 values as 1 and 562 values as 0, which shows that it struggled more with identifying class 1. This lines up with the 51.60% accuracy from the classification report and shows that the model isn't making very reliable predictions yet.

# In[27]:


#Compare to Baseline
#Find the most common class in the training data
majority_class = y_train.mode()[0]

#Create baseline predictions
baseline_pred = np.full(
    len(y_test),
    majority_class
)

#Calculate baseline accuracy
baseline_accuracy = accuracy_score(
    y_test,
    baseline_pred
)

#Compare baseline and logistic regression
print(f"Baseline Accuracy: {baseline_accuracy:.2%}")
print(f"Logistic Regression Accuracy: {accuracy:.2%}")

#Calculate the difference
improvement = (accuracy - baseline_accuracy) * 100

print(
    f"Accuracy Difference: "
    f"{improvement:.2f} percentage points"
)


# I compared my logistic regression model to a baseline that always predicts the most common class. The baseline had an accuracy of 50.80%, while my logistic regression model reached 51.60%, which is only a 0.80 percentage point improvement. Although the model did slightly better, the difference is pretty small, which tells me it might not be finding many useful patterns in the data.

# Starting Week 7 Exercise | In Week 6, I created a logistic regression classification model to predict the binary target variable, y. The initial model achieved an accuracy of about 51.6%, which was only slightly higher than the majority-class baseline of about 50.8%. For this exercise in Week 7, I will evaluate the model using additional classification metrics and then use cross-validation and hyperparameter tuning to see whether the model's performance can be improved. I will compare the tuned model with the original logistic regression model using the same testing dataset.

# In[28]:


#Import 
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)

print("Additional evaluation and tuning tools imported successfully.")


# The original model was evaluated using accuracy, a classification report, and a confusion matrix. For this exercise, I also want to calculate precision, recall, F1-score, and ROC-AUC. I also imported GridSearchCV and StratifiedKFold so I can test different logistic regression settings using cross-validation. Stratified cross-validation is useful for classification because it helps keep the proportion of each target class similar across the folds.

# In[29]:


#Generate predictions from the original logistic regression model
original_pred = model.predict(X_test)

#Probability that each observation belongs to class 1
original_prob = model.predict_proba(X_test)[:, 1]

print("Original model predictions generated successfully.")


# I first generated predictions from the original logistic regression model. These predictions will be used to calculate a more complete set of classification metrics. This gives me a baseline model that I can compare with the tuned model later. And to avoid confusion I used original_pred over baseline_pred. 

# In[30]:


#Calculate baseline classification metrics
original_accuracy = accuracy_score(
    y_test,
    original_pred
)

original_precision = precision_score(
    y_test,
    original_pred,
    zero_division=0
)

original_recall = recall_score(
    y_test,
    original_pred,
    zero_division=0
)

original_f1 = f1_score(
    y_test,
    original_pred,
    zero_division=0
)

original_auc = roc_auc_score(
    y_test,
    original_prob
)

print("Original Logistic Regression Evaluation")
print(f"Accuracy:  {original_accuracy:.4f}")
print(f"Precision: {original_precision:.4f}")
print(f"Recall:    {original_recall:.4f}")
print(f"F1-Score:  {original_f1:.4f}")
print(f"ROC-AUC:   {original_auc:.4f}")


# Accuracy tells me how many predictions were correct overall, but accuracy alone does not show what kinds of mistakes the model is making. I calculated precision, recall, F1-score, and ROC-AUC in addition to accuracy. Accuracy measures the percentage of all predictions that were correct. Precision measures how often predictions of class 1 were actually class 1. Recall measures how many of the actual class 1 observations the model successfully identified.F1-score balances precision and recall into one measurement. ROC-AUC measures how well the model separates the two classes across different classification thresholds. Looking at several metrics gives me a more complete picture of how well the classification model performs.

# In[31]:


#Re-evaluate the model 
print(
    classification_report(
        y_test,
        original_pred,
        zero_division=0
    )
)


# I reviewed the classification report to compare how the model performed for both classes. The model performed better at identifying class 0 than class 1 in my original report.For class 0, the recall was 0.60, which means the model correctly identified about 60% of the actual class 0 observations. For class 1, the recall was lower at 0.43, meaning the model missed more of the actual class 1 observations. The F1-score was also higher for class 0 at 0.56 compared with 0.47 for class 1. Overall accuracy was about 52%, and the macro and weighted averages were both around 0.51–0.52. This tells me that the model is only slightly better than random guessing and has more difficulty identifying class 1 correctly.

# In[32]:


#Create confusion matrix
original_cm = confusion_matrix(
    y_test,
    original_pred
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=original_cm,
    display_labels=[0, 1]
)

disp.plot(values_format="d")

plt.title("Original Logistic Regression Confusion Matrix")
plt.show()


# The confusion matrix gives me a closer look at the predictions made by the original logistic regression model. The model correctly predicted 610 observations as class 0 and 422 observations as class 1. It incorrectly predicted 406 class 0 observations as class 1 and incorrectly predicted 562 class 1 observations as class 0. The largest error was the 562 false negatives, meaning the model had more difficulty identifying actual class 1 observations. This matches the classification report, where class 1 had a lower recall score of 0.43 compared with 0.60 for class 0.

# In[33]:


#Create ROC curve for the original model
RocCurveDisplay.from_predictions(
    y_test,
    original_prob
)

plt.title("ROC Curve - Original Logistic Regression")
plt.show()


# I created an ROC curve to see how well the original logistic regression model can separate class 0 from class 1 across different classification thresholds. The ROC-AUC score was 0.52. Since an AUC of 0.50 represents performance close to random guessing, this shows me that the model currently has very little ability to distinguish between the two classes. The ROC curve also stays fairly close to the diagonal line, which supports this result. This matches what I saw with the other evaluation metrics. The model's accuracy was only about 52%, and it had difficulty correctly identifying class 1. This gives me another reason to test whether tuning the logistic regression model can improve its performance.

# In[34]:


#Create a summary table of the original model metrics
original_metrics = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score",
        "ROC-AUC"
    ],
    "Original Model": [
        original_accuracy,
        original_precision,
        original_recall,
        original_f1,
        original_auc
    ]
})

display(original_metrics)


# I placed the main classification metrics into one table so I could review the overall performance of the original logistic regression model more easily. This table also gives me a baseline that I can use later when comparing the original model with the tuned model. Instead of looking only at accuracy, I can compare changes in precision, recall, F1-score, and ROC-AUC as well.

# In[35]:


#Create 5-fold stratified cross-validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

print("5-fold stratified cross-validation created.")


# I created a 5-fold stratified cross-validation process to use during model tuning. This divides the training data into five sections and repeatedly trains and validates the model using different sections of the data. I used StratifiedKFold because this is a classification problem. Stratification helps keep a similar proportion of class 0 and class 1 observations in each fold, which makes the validation results more consistent.

# In[36]:


#Create the parameter grid for logistic regression
param_grid = {
    "classifier__C": [
        0.01,
        0.1,
        1,
        10,
        100
    ],
    "classifier__penalty": [
        "l1",
        "l2"
    ],
    "classifier__solver": [
        "liblinear"
    ]
}

print("Parameter grid created.")


# I created a parameter grid to test different regularization settings for logistic regression. The C parameter controls the strength of regularization. Smaller C values apply stronger regularization, while larger values apply less. I also tested both L1 and L2 penalties to see whether either approach improved the model. This gives GridSearchCV a lot of combinations to compare instead of relying only on the default logistic regression settings.

# In[37]:


#Create GridSearchCV
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=cv,
    scoring="f1",
    n_jobs=-1,
    return_train_score=True
)

#Fit using training data only
grid_search.fit(
    X_train,
    y_train
)

print("Hyperparameter tuning complete.")


# I used GridSearchCV to test the different logistic regression parameter combinations using 5-fold stratified cross-validation. I chose F1-score as the scoring metric because my original model had more difficulty identifying class 1. F1-score considers both precision and recall, so it gives me a better way to evaluate the balance between the two. The grid search was performed only on the training data so the testing data could remain separate for the final evaluation.

# In[38]:


#Review the best Parameters
#Display the best parameters and cross-validation score
print("Best Parameters:")
print(grid_search.best_params_)

print("\nBest Cross-Validation F1-Score:")
print(
    round(
        grid_search.best_score_,
        4
    )
)


# From the grid search, I found that the best settings were a C value of 100, an L2 penalty, and the liblinear solver. The best cross-validation F1-score was 0.4283. The larger C value tells me that the better-performing version of the model used less regularization. I also found that L2 performed better than L1 for the combinations I tested. Even though this was the best result from the grid search, the F1-score was still only about 0.43. This makes me think the tuning may only improve the model a little, so I still need to compare the tuned model with the original model on the test data before deciding if it really performed better.

# In[39]:


#Convert GridSearchCV results into a DataFrame
cv_results = pd.DataFrame(
    grid_search.cv_results_
)

#Select the most useful columns
tuning_results = cv_results[[
    "param_classifier__C",
    "param_classifier__penalty",
    "mean_test_score",
    "std_test_score",
    "rank_test_score"
]].copy()

#Rename columns
tuning_results = tuning_results.rename(
    columns={
        "param_classifier__C": "C",
        "param_classifier__penalty": "Penalty",
        "mean_test_score": "Mean CV F1",
        "std_test_score": "CV F1 Std Dev",
        "rank_test_score": "Rank"
    }
)

#Display results from best to worst
display(
    tuning_results.sort_values("Rank")
)


# The best result used a C value of 100 with an L2 penalty and had a mean F1-score of about 0.4283. Most of the stronger models had C values of 10 or 100, while smaller C values generally performed worse. The C value of 0.01 with an L1 penalty performed the worst with an F1-score of 0. This suggests that too much regularization hurt the model. Overall, tuning changed the results slightly, but the best F1-scores still stayed around 0.43.

# In[40]:


#Save the best model found by GridSearchCV
tuned_model = grid_search.best_estimator_

#Generate predictions using the test set
tuned_pred = tuned_model.predict(
    X_test
)

#Generate probabilities for class 1
tuned_prob = tuned_model.predict_proba(
    X_test
)[:, 1]

print("Tuned model predictions generated.")


# I saved the best model selected by GridSearchCV and used it to make predictions on the original testing dataset. The testing data was not used during hyperparameter tuning, so it provides a better way to evaluate how well the tuned model performs on new data.

# In[41]:


#Calculate evaluation metrics for the tuned model
tuned_accuracy = accuracy_score(
    y_test,
    tuned_pred
)

tuned_precision = precision_score(
    y_test,
    tuned_pred,
    zero_division=0
)

tuned_recall = recall_score(
    y_test,
    tuned_pred,
    zero_division=0
)

tuned_f1 = f1_score(
    y_test,
    tuned_pred,
    zero_division=0
)

tuned_auc = roc_auc_score(
    y_test,
    tuned_prob
)

print("Tuned Logistic Regression Evaluation")
print(f"Accuracy:  {tuned_accuracy:.4f}")
print(f"Precision: {tuned_precision:.4f}")
print(f"Recall:    {tuned_recall:.4f}")
print(f"F1-Score:  {tuned_f1:.4f}")
print(f"ROC-AUC:   {tuned_auc:.4f}")


# The tuned logistic regression model had an accuracy of 0.5160, with precision at 0.5097, recall at 0.4278, and an F1-score of 0.4652. The ROC-AUC score was 0.5155. These results are very close to the original model, so the tuning did not make a major difference in performance. The recall and F1-score for class 1 are still fairly low, which shows that the model continues to have difficulty correctly identifying class 1.

# In[42]:


#Display classification report for tuned model
print(
    classification_report(
        y_test,
        tuned_pred,
        zero_division=0
    )
)


# The tuned classification report looks almost the same as the original model. Class 0 still had a recall of 0.60 and an F1-score of 0.56, while class 1 had a lower recall of 0.43 and an F1-score of 0.47. This shows that tuning did not really change how the model handled the two classes. The model still performs better at identifying class 0 and continues to have more difficulty correctly identifying class 1.

# In[43]:


#Create confusion matrix for the tuned model
tuned_cm = confusion_matrix(
    y_test,
    tuned_pred
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=tuned_cm,
    display_labels=[0, 1]
)

disp.plot(values_format="d")

plt.title("Tuned Logistic Regression Confusion Matrix")
plt.show()


# The tuned confusion matrix was almost identical to the original model. The tuned model correctly predicted 611 class 0 observations and 421 class 1 observations. It also produced 405 false positives and 563 false negatives. Compared with the original model, the changes were only by one prediction in each area. This confirms that hyperparameter tuning did not meaningfully improve the model. The model still has the most difficulty with false negatives, meaning it continues to miss many actual class 1 observations.

# In[44]:


#Create comparison table
comparison = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score",
        "ROC-AUC"
    ],
    "Original Model": [
        original_accuracy,
        original_precision,
        original_recall,
        original_f1,
        original_auc
    ],
    "Tuned Model": [
        tuned_accuracy,
        tuned_precision,
        tuned_recall,
        tuned_f1,
        tuned_auc
    ]
})

#Calculate the change
comparison["Change"] = (
    comparison["Tuned Model"]
    - comparison["Original Model"]
)

display(comparison)


# The comparison shows that the tuned model performed almost the same as the original model. Accuracy stayed the same, precision increased a very slight amount, while recall, and F1-score decreased by very small (almost insignificant) amounts. ROC-AUC stayed essentially unchanged. These differences are so small that I would not consider the tuned model to be meaningfully better. Overall, changing the logistic regression settings did not improve the model much.

# In[45]:


#Compare with the Majority class Baseline
#Find the most common class in the training data
majority_class = y_train.mode()[0]

#Predict the majority class for every test observation
majority_pred = np.full(
    len(y_test),
    majority_class
)

#Calculate majority baseline accuracy
majority_accuracy = accuracy_score(
    y_test,
    majority_pred
)

print(
    f"Majority Baseline Accuracy: "
    f"{majority_accuracy:.4f}"
)

print(
    f"Original Logistic Regression Accuracy: "
    f"{original_accuracy:.4f}"
)

print(
    f"Tuned Logistic Regression Accuracy: "
    f"{tuned_accuracy:.4f}"
)


# The majority-class baseline had an accuracy of 0.5080, while both logistic regression models had an accuracy of about 0.5160. So the models performed slightly better than simply predicting the most common class every time, but the improvement was super small. This supports the other results showing that the current logistic regression model is only learning a weak pattern from the predictor variables.

# In[46]:


#Compare ROC curves on the same graph
fig, ax = plt.subplots(figsize=(7, 5))

RocCurveDisplay.from_predictions(
    y_test,
    original_prob,
    name="Original Logistic Regression",
    ax=ax
)

RocCurveDisplay.from_predictions(
    y_test,
    tuned_prob,
    name="Tuned Logistic Regression",
    ax=ax
)

plt.title("ROC Curves: Original vs. Tuned Model")
plt.show()


# The ROC curves for the original and tuned logistic regression models are almost identical, and both have an AUC of about 0.52. This shows that tuning did not improve the model’s ability to separate class 0 from class 1. Since an AUC close to 0.50 is similar to random guessing, both models still have weak classification performance.

# In[47]:


#Display final model comparison
print("----FINAL MODEL COMPARISON----")

print("\nOriginal Logistic Regression")
print(f"Accuracy:  {original_accuracy:.4f}")
print(f"Precision: {original_precision:.4f}")
print(f"Recall:    {original_recall:.4f}")
print(f"F1-Score:  {original_f1:.4f}")
print(f"ROC-AUC:   {original_auc:.4f}")

print("\nBest Hyperparameters")
print(grid_search.best_params_)

print("\nTuned Logistic Regression")
print(f"Accuracy:  {tuned_accuracy:.4f}")
print(f"Precision: {tuned_precision:.4f}")
print(f"Recall:    {tuned_recall:.4f}")
print(f"F1-Score:  {tuned_f1:.4f}")
print(f"ROC-AUC:   {tuned_auc:.4f}")

print(
    f"\nMajority Baseline Accuracy: "
    f"{majority_accuracy:.4f}"
)


# The final comparison shows that the original and tuned logistic regression models performed almost the same. Both had an accuracy of 0.5160, and the precision, recall, F1-score, and ROC-AUC values were also nearly identical. The tuned model used a C value of 100 with an L2 penalty and the liblinear solver, but these settings did not lead to a meaningful improvement. Both models were only slightly better than the majority baseline accuracy of 0.5080. Overall, the results suggest that changing the logistic regression settings was not enough to improve the model much. The next step may be to revisit the features or try a different classification algorithm.

# START of Week 10 (10.2 Exercise)

# In[48]:


#Import Joblib to save the trained model
import joblib

#Prepare the model and test data for the dashboard
dashboard_data = {
    "model": tuned_model,
    "X_test": X_test,
    "y_test": y_test
}

#Save the model and test data
joblib.dump(dashboard_data, "week7_dashboard_data.joblib")

#Confirm the file was saved
print("Week 7 model and test data saved successfully!")


# For Week 10, I decided to use Streamlit to create my predictive model dashboard. Since I already built and evaluated a logistic regression model in Week 7, I wanted to reuse that model instead of starting over. I used Joblib to save my tuned model along with the testing data. The file saved successfully, which means I can now use my previous model in an interactive dashboard without having to retrain it every time.

# In[49]:


#Install Streamlit for the predictive model dashboard
get_ipython().run_line_magic('pip', 'install streamlit joblib')


# I installed Streamlit so I could turn my Week 7 logistic regression model into an interactive dashboard. I chose Streamlit because it seemed easier to work with than Dash and allows me to create an application using Python. The installation was successful, so I can now start setting up the application and connecting it to my saved model.

# In[50]:


#Import libraries
import streamlit as st
import joblib
import os

#Check the Streamlit version
print("Streamlit Version:", st.__version__)

#Check that the saved model exists
print("Model File Exists:", os.path.exists("week7_dashboard_data.joblib"))

#Confirm the environment is ready
print("Dashboard environment checked successfully!")


# After installing Streamlit, I checked that the library was working and that my saved Week 7 model was still available. The results showed that Streamlit version 1.67.0 was installed, and the model file returned True, confirming that it was found. This was important because I needed to make sure the application could access my trained model before I started building the dashboard.

# In[51]:


#Import libraries
from pathlib import Path

#Create the Streamlit application code
app_code = '''
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    ConfusionMatrixDisplay
)

#Set up the dashboard
st.set_page_config(
    page_title="Predicting Donor Churn for Nonprofit Organizations",
    layout="wide"
)

#Display dashboard title
st.title("Predicting Donor Churn for Nonprofit Organizations")
st.write("Logistic Regression Classification")

#Load the saved model and test data
dashboard_data = joblib.load("week7_dashboard_data.joblib")

model = dashboard_data["model"]
X_test = dashboard_data["X_test"]
y_test = dashboard_data["y_test"]

#Generate model predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

#Calculate model evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
auc = roc_auc_score(y_test, y_prob)

#Display model performance
st.header("Model Performance")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Accuracy", f"{accuracy:.4f}")
col2.metric("Precision", f"{precision:.4f}")
col3.metric("Recall", f"{recall:.4f}")
col4.metric("F1-Score", f"{f1:.4f}")
col5.metric("ROC-AUC", f"{auc:.4f}")

#Explain model limitations
st.info(
    "The model performed only slightly better than "
    "the majority-class baseline."
)

#Display confusion matrix
st.header("Confusion Matrix")

fig, ax = plt.subplots()

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    labels=[0, 1],
    display_labels=[0, 1],
    cmap="Blues",
    ax=ax
)

st.pyplot(fig)
plt.close(fig)

#Create interactive prediction section
st.header("Make a Prediction")
st.write("Select a test record and adjust the feature values.")

#Select a test record
record = st.selectbox(
    "Choose a Test Record",
    range(len(X_test)),
    format_func=lambda x: f"Record {x + 1}"
)

selected_data = X_test.iloc[record]

#Create predictor input fields
input_values = {}

with st.form("prediction_form"):

    for column in X_test.columns:

        if pd.api.types.is_numeric_dtype(X_test[column]):

            value = selected_data[column]

            if pd.isna(value):
                value = X_test[column].median()

            input_values[column] = st.number_input(
                column,
                value=float(value)
            )

        else:

            options = sorted(
                X_test[column].dropna().astype(str).unique()
            )

            if X_test[column].isna().any():
                options.append("(missing)")

            current_value = selected_data[column]

            if pd.isna(current_value):
                current_value = "(missing)"
            else:
                current_value = str(current_value)

            input_values[column] = st.selectbox(
                column,
                options,
                index=options.index(current_value)
                if current_value in options else 0
            )

    #Create prediction button
    predict_button = st.form_submit_button("Predict")

#Generate prediction
if predict_button:

    #Prepare the selected input data
    input_df = pd.DataFrame(
        [input_values],
        columns=X_test.columns
    )

    #Restore missing values
    input_df = input_df.replace("(missing)", np.nan)

    #Calculate prediction and probabilities
    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]

    #Display prediction results
    st.subheader("Prediction Results")
    st.success(f"Predicted Class: {prediction}")

    #Display prediction probabilities
    probability_df = pd.DataFrame({
        "Class": [str(x) for x in model.classes_],
        "Probability": probabilities
    }).set_index("Class")

    st.bar_chart(probability_df)

    #Display input data
    st.subheader("Input Data")
    st.dataframe(input_df)

#Add dashboard footer
st.write("---")
st.caption("Predictive Model Dashboard")
'''

#Save the application as a Python file
Path("app.py").write_text(app_code, encoding="utf-8")

#Confirm the file was created
print("Streamlit dashboard application saved successfully!")
print("File Exists:", Path("app.py").exists())


# I created the Python application file for my Streamlit dashboard. I used my tuned logistic regression model from Week 7 and included sections for the evaluation metrics, confusion matrix, and interactive predictions. I also added input fields so users can select a test record, change the feature values, and see what classification the model predicts. The output confirmed that the application file was saved successfully. Now that the file is created, I can check that everything is in place before launching the dashboard.

# In[52]:


#Verify dashboard files
#Import libraries
from pathlib import Path
import os

#Check that both dashboard files exist
app_exists = Path("app.py").exists()
model_exists = Path("week7_dashboard_data.joblib").exists()

#Display the file results
print("Application File Exists:", app_exists)
print("Model File Exists:", model_exists)

#Display the working directory
print("Working Directory:", os.getcwd())

#Confirm the files are ready
if app_exists and model_exists:
    print("Both dashboard files are ready!")
else:
    print("One or more dashboard files are missing.")


# After creating the Streamlit application, I checked that both the Python application file and my Week 7 model were available. Both files returned True, basically confirming that they were created and saved in the directory. This is important because Streamlit needs access to the previous model to make predictions. I can move forward with launching the dashboard and see how it works.

# In[53]:


#Launch Streamlit
#Import libraries
import subprocess
import sys
from pathlib import Path

#Find the Streamlit file
app_path = Path("app.py").resolve()

#Launch the Streamlit dashboard
dashboard_process = subprocess.Popen(
    [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.headless=true"
    ],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL
)

#Display dashboard information
print("Streamlit launch command started!")
print("Dashboard URL: http://localhost:8501")
print("Process ID:", dashboard_process.pid)


# I was able to launch my Streamlit dashboard successfully! I opened it in my browser and reviewed the dashboard which displayed my Week 7 model along with its performance metrics and confusion matrix. The model had an accuracy of 51.60%, precision of 50.97%, recall of 42.78%, F1-score of 46.52%, and ROC-AUC of 51.55%. These results matched what I previously calculated in Week 7, which confirmed that the dashboard was using the correct model.
# 
# Looking at the confusion matrix, the model correctly classified 611 records as class 0 and 421 records as class 1. However, it also incorrectly classified 405 records as class 1 and 563 records as class 0. This showed that the model was still struggling to distinguish between the two classes. Although, it worked successfully, the model's predictive performance still has room for improvement. 

# I tested the interactive prediction feature by selecting a record and clicking the Predict button. The application successfully generated a prediction of Class 0 and displayed a probability chart comparing both classes. The model estimated approximately a 50.5% probability for Class 0 and a 49.5% probability for Class 1. Since the probabilities were so close, the prediction showed that the model had difficulty clearly separating the two classes. This also matched what I observed during Week 7, when my logistic regression model had an accuracy of only 51.60%. Overall, the prediction feature worked as expected. I was able to use the model through the dashboard instead of having to manually run the prediction code in Jupyter.

# #Verify prediction
# #Import libraries
# import joblib
# import pandas as pd
# 
# #Load the saved Week 7 model and testing data
# dashboard_data = joblib.load("week7_dashboard_data.joblib")
# 
# model = dashboard_data["model"]
# X_test = dashboard_data["X_test"]
# 
# #Select the first test record
# test_record = X_test.iloc[[0]]
# 
# #Generate a prediction
# prediction = model.predict(test_record)
# 
# #Calculate prediction probabilities
# probabilities = model.predict_proba(test_record)
# 
# #Display the results
# print("Selected Test Record: 1")
# print("Predicted Class:", prediction[0])
# 
# #Display probabilities for each class
# for class_label, probability in zip(
#     model.classes_,
#     probabilities[0]
# ):
#     print(
#         f"Class {class_label} Probability: "
#         f"{probability:.4f}"
#     )
# 
# #Confirm the prediction process completed
# print("Prediction verification completed successfully!")

# After testing the dashboard, I wanted to double-check that the predictions matched what my model produced. I loaded the saved Week 7 model and used the first record from the testing dataset to generate another prediction to make sure it would pull that data. The model predicted Class 0, with a probability of 50.47%, compared to 49.53% for Class 1. These results matched what I saw in the Streamlit dashboard, confirming that both were using the same saved model. The probabilities were very close, which showed that the model was not strongly favoring either class. Even though the model's accuracy still needs improvement, I was able to confirm that the dashboard was working correctly and producing consistent results.

# One thing I learned from this exercise is that creating an application and building an accurate predictive model are two different challenges. Streamlit made it easier to display my results and interact with the model, but it did not improve the model's actual performance. In the future, I would like to explore different classification algorithms or improve the features to see whether I can achieve better predictions. Overall, this exercise helped me better understand how a machine learning model can be saved, reused, and turned into an interactive application outside of Jupyter Notebook. 

# START Milestone 5: Final Predictive Application
# For the final stage of my project, I will use Streamlit to create an interactive application that uses the tuned logistic regression model developed earlier in the project. The application will allow a user to enter predictor values and receive a classification prediction of Class 0 or Class 1. This application is part of my Predicting Donor Churn for Nonprofit Organizations project and demonstrates how the final predictive model can be saved, reused, and connected to an interactive application.

# In[54]:


#Install Streamlit and Joblib
get_ipython().run_line_magic('pip', 'install streamlit joblib')


# I installed Streamlit to create the interactive application and Joblib to save the trained model. Saving the model allows the application to use the model created during the project without repeating the entire training process every time the application runs.

# In[55]:


#Import libraries for the final application
import joblib
import os
from pathlib import Path

print("Application libraries imported successfully.")


# In[56]:


#Recreate the tuned logistic regression model
#Create the tuned model using the best settings
tuned_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            LogisticRegression(
                C=100,
                penalty="l2",
                solver="liblinear",
                max_iter=1000,
                random_state=42
            )
        )
    ]
)

#Train the tuned model
tuned_model.fit(
    X_train,
    y_train
)

print(
    "Tuned logistic regression model recreated successfully."
)


# I recreated the final tuned model, because I restarted the kernel, so I needed to recreate the tuned logistic regression model before building the application. I used the same preprocessing pipeline and the best hyperparameters found earlier with GridSearchCV: a C value of 100, an L2 penalty, and the liblinear solver. The model was then trained again using the existing training data. This makes sure the final model is available in memory and ready to be saved and used by the Streamlit application.

# In[57]:


#Confirm the final tuned model exists
print(tuned_model)

print(
    "\nModel Type:",
    type(tuned_model).__name__
)


# I confirmed that the final tuned model is stored as a pipeline.

# In[58]:


#Save the final tuned model
import joblib

joblib.dump(
    tuned_model,
    "tuned_logistic_model.joblib"
)

print(
    "Final tuned model saved successfully."
)


# In[59]:


#Save the feature lists used by the model
app_features = {
    "numeric_features": numeric_features,
    "categorical_features": categorical_features
}

print(
    "Numerical Features:",
    app_features["numeric_features"]
)

print(
    "\nCategorical Features:",
    app_features["categorical_features"]
)


# I saved the numerical and categorical feature lists used by the model. These lists will help the Streamlit application know which inputs should be treated as numbers and which ones should use categories.

# In[60]:


#Save the category choices for the application
category_choices = {}

for column in categorical_features:

    category_choices[column] = sorted(
        X_train[column]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

category_choices


# I reviewed the available values for each categorical feature so the Streamlit application can use them as dropdown options. The categories include months for "x1", vehicle brands for "x13", weekdays for "x14", and size categories for "x17". Using the values directly from the training data helps make sure the application only provides category choices that the model already recognizes.

# In[61]:


#Calculate median values for numerical inputs
numeric_defaults = {}

for column in numeric_features:

    numeric_defaults[column] = float(
        X_train[column].median()
    )

numeric_defaults


# I calculated the median value for each numerical feature in the training data. These median values will be used as the default numbers shown in the Streamlit application. Using median values gives the app reasonable starting points for each numerical input while still allowing the user to change the values before making a prediction.

# In[62]:


#Create the application model bundle
app_bundle = {
    "model": tuned_model,
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
    "category_choices": category_choices,
    "numeric_defaults": numeric_defaults
}

#Save the application bundle
joblib.dump(
    app_bundle,
    "classification_app_bundle.joblib"
)

print(
    "Application model bundle saved successfully."
)


# I combined the tuned model with the numerical features, categorical features, dropdown choices, and default numerical values into one application bundle. I then saved that bundle so the Streamlit application can load everything it needs from one file.

# In[63]:


#Verify the saved files
from pathlib import Path

model_file = Path(
    "tuned_logistic_model.joblib"
)

bundle_file = Path(
    "classification_app_bundle.joblib"
)

print(
    "Saved Model Exists:",
    model_file.exists()
)

print(
    "Application Bundle Exists:",
    bundle_file.exists()
)


# I checked that both the tuned logistic regression model and the application bundle were saved successfully. Both files are needed for the final application, so confirming that they exist helps prevent errors coming up when Streamlit tries to load the model.

# In[64]:


get_ipython().run_cell_magic('writefile', 'classification_app.py', '\n#Create Streamlit app\nimport streamlit as st\nimport pandas as pd\nimport joblib\n\n#Set up the page\nst.set_page_config(\n    page_title="Predicting Donor Churn for Nonprofit Organizations",\n    page_icon="📊",\n    layout="wide"\n)\n\n#Load the saved application bundle\nbundle = joblib.load(\n    "classification_app_bundle.joblib"\n)\n\nmodel = bundle["model"]\nnumeric_features = bundle["numeric_features"]\ncategorical_features = bundle["categorical_features"]\ncategory_choices = bundle["category_choices"]\nnumeric_defaults = bundle["numeric_defaults"]\n\n#Application title\nst.title(\n    "📊 Predicting Donor Churn for Nonprofit Organizations"\n)\n\nst.write(\n    """\n    This application uses the tuned logistic regression\n    model developed during my project,\n    Predicting Donor Churn for Nonprofit Organizations.\n\n    Enter predictor values below and the model will\n    predict whether the observation belongs to Class 0\n    or Class 1.\n    """\n)\n\nst.divider()\n\n#Input section\nst.subheader(\n    "Enter Predictor Values"\n)\n\nuser_input = {}\n\n#Numerical inputs\nst.write(\n    "### Numerical Features"\n)\n\nnum_cols = st.columns(2)\n\nfor index, feature in enumerate(\n    numeric_features\n):\n\n    with num_cols[index % 2]:\n\n        user_input[feature] = st.number_input(\n            feature,\n            value=float(\n                numeric_defaults[feature]\n            )\n        )\n\n#Categorical inputs\nst.write(\n    "### Categorical Features"\n)\n\ncat_cols = st.columns(2)\n\nfor index, feature in enumerate(\n    categorical_features\n):\n\n    with cat_cols[index % 2]:\n\n        user_input[feature] = st.selectbox(\n            feature,\n            category_choices[feature]\n        )\n\nst.divider()\n\n#Prediction button\nif st.button(\n    "Make Prediction",\n    type="primary"\n):\n\n    input_data = pd.DataFrame(\n        [user_input]\n    )\n\n    feature_order = (\n        numeric_features\n        + categorical_features\n    )\n\n    input_data = input_data[\n        feature_order\n    ]\n\n    prediction = model.predict(\n        input_data\n    )[0]\n\n    probability = model.predict_proba(\n        input_data\n    )[0, 1]\n\n    st.subheader(\n        "Prediction Result"\n    )\n\n    result_col1, result_col2 = (\n        st.columns(2)\n    )\n\n    result_col1.metric(\n        "Predicted Class",\n        int(prediction)\n    )\n\n    result_col2.metric(\n        "Probability of Class 1",\n        f"{probability:.1%}"\n    )\n\n    st.progress(\n        float(probability)\n    )\n\n    if prediction == 1:\n\n        st.success(\n            "The model predicts Class 1."\n        )\n\n    else:\n\n        st.info(\n            "The model predicts Class 0."\n        )\n\n#Model information\nwith st.expander(\n    "About the Model"\n):\n\n    st.write(\n        """\n        This application uses the tuned logistic regression\n        pipeline developed during the project,\n        Predicting Donor Churn for Nonprofit Organizations.\n\n        Hyperparameter tuning was completed using\n        GridSearchCV with 5-fold stratified\n        cross-validation.\n\n        Best model settings:\n\n        - C = 100\n        - L2 penalty\n        - liblinear solver\n\n        The pipeline also handles missing values,\n        numerical scaling, and one-hot encoding for\n        categorical variables.\n        """\n    )\n\nst.caption(\n    "Predicting Donor Churn for Nonprofit Organizations"\n)\n')


# In[65]:


#Verify the Streamlit application file
from pathlib import Path

app_file = Path(
    "classification_app.py"
)

print(
    "Application File Exists:",
    app_file.exists()
)


# I confirmed that the Python file for the Streamlit application was created successfully. This file contains the interface and prediction code that will allow the tuned logistic regression model to run outside of Jupyter Notebook.

# In[66]:


#Launch the Streamlit application
import subprocess
import sys

app_path = Path(
    "classification_app.py"
).resolve()

streamlit_process = subprocess.Popen(
    [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.headless=true"
    ]
)

print(
    "Streamlit application started."
)

print(
    "Open this address in your browser:"
)

print(
    "http://localhost:8501"
)


# In[67]:


#Test the saved application bundle

#Load the saved bundle
test_bundle = joblib.load(
    "classification_app_bundle.joblib"
)

#Get the saved model
saved_model = test_bundle["model"]

#Select one record from the test data
sample_record = X_test.iloc[[0]]

#Make a prediction
sample_prediction = saved_model.predict(
    sample_record
)[0]

#Get probability of Class 1
sample_probability = saved_model.predict_proba(
    sample_record
)[0, 1]

print(
    "Sample Prediction:",
    sample_prediction
)

print(
    "Probability of Class 1:",
    round(
        sample_probability,
        4
    )
)


# After creating and launching the Streamlit application, I tested the saved application bundle using one record from the testing data. The model predicted Class 0, with a 0.4953 probability of Class 1. This confirmed that the saved model could be loaded successfully and still generate predictions using the same preprocessing pipeline from the project. Since the Class 1 probability was close to 50%, this particular prediction was fairly close to the model's decision threshold.

# For this final project, I built and tested a classification model and then connected the final tuned logistic regression model to an interactive Streamlit application. The project included data cleaning, preprocessing, model evaluation, hyperparameter tuning, saving the model, and creating an application that can generate predictions from new inputs. The final tuned model had an accuracy of 0.5160, precision of 0.5097, recall of 0.4278, F1-score of 0.4652, and ROC-AUC of 0.5155. These results show that the model only performed slightly better than a basic baseline, so there is still room for improvement. One of the biggest things I learned from this project was that creating a model is only part of the predictive analytics process. The model also needs to be evaluated carefully and connected to something that can actually be used. Building the Streamlit application helped me see how a trained model can be turned into an interactive tool. If I continued this project, I would test additional algorithms, explore stronger feature engineering, and use a dataset with clearer donor-specific variables so the model could better support the original goal of predicting donor churn. With further development and more donor-specific data, I could see this type of model being useful in my work outside of school.

# In[ ]:




