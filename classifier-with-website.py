import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import streamlit as st

# Read CSV file
path = "data-3.csv"
data = pd.read_csv(path)

# Clean unnecessary data
columns_to_keep = ['diagnosis', 'perimeter_mean', 'radius_mean', 'texture_mean', 'area_mean',
                   'smoothness_mean', 'concavity_mean', 'symmetry_mean']
data = data.drop(columns=[col for col in data.columns if col not in columns_to_keep], axis = 1)


# Convert "Malignant" and "Benign" to 1 and 0
data.diagnosis = [1 if val == 'M' else 0 for val in data.diagnosis]

# Create predictors and target variable
X = data.drop(['diagnosis'], axis = 1)
y = data.diagnosis
# Normalize the predictor values
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

from sklearn.model_selection import train_test_split as tts

X_train, X_test, y_train, y_test = tts(X_scaled, y, test_size = 0.75, random_state=100)

from sklearn.linear_model import LogisticRegression
LogReg = LogisticRegression()

LogReg.fit(X_train, y_train)
y_predict = LogReg.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_predict)
print("Accuracy: ", accuracy*100)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_predict))

st.title('Breast Cancer Classification Model')
st.write('Our model uses many demographics of a cell to predict whether the cell is cancerous. Check out the model in action here! ')

# Section 1: model in action on testing data
st.header('Testing the Model')
st.write("We trained our model and then tested it on separate, unseen data. Here's how the model performed.")

st.code(classification_report(y_test, y_predict), language='html')

# Section 2: User Input and Model Classification
st.header('Classify a Cell of Your Own')
st.write('Fill in the following fields with cell data to get a prediction!')
user_input1 = st.text_input('Enter a perimeter mean:')
if user_input1 == "":
    user_input1 = 0
else:
    user_input1 = float(user_input1)

user_input2 = st.text_input('Enter a radius mean:')
if user_input2 == "":
    user_input2 = 0
else:
    user_input2 = float(user_input2)

user_input3 = st.text_input('Enter a texture mean:')
if user_input3 == "":
    user_input3 = 0
else:
    user_input3 = float(user_input3)

user_input4 = st.text_input('Enter an area mean:')
if user_input4 == "":
    user_input4 = 0
else:
    user_input4 = float(user_input4)

user_input5 = st.text_input('Enter a smoothness mean:')
if user_input5 == "":
    user_input5 = 0
else:
    user_input5 = float(user_input5)

user_input6 = st.text_input('Enter a concavity mean:')
if user_input6 == "":
    user_input6 = 0
else:
    user_input6 = float(user_input6)

user_input7 = st.text_input('Enter a symmetry mean:')
if user_input7 == "":
    user_input7 = 0
else:
    user_input7 = float(user_input7)


if st.button('Classify...'):
    user_cell = np.array([[user_input1, user_input2, user_input3, user_input4, user_input5, user_input6, user_input7]])
    prediction = LogReg.predict(user_cell)
    word = "cancerous."
    if prediction == 0:
        word = "not cancerous."
    st.write('Our model predicts that this cell is ',word)

    if prediction == 1:
        st.header('Optimal and Personalized Treatment')
        st.write("Based on your input data, you are best suited for...")
        st.subheader("Neo-Adjuvant Chemotherapy")
        st.write("Chemotherapy treatment that shrinks the size of the tumor. Before surgery, neoadjuvant chemo is taken in the presence of an inflammatory tumor or in cases where the tumor's size or extent makes it difficult to perform surgery initially. This increases the likelihood of a successful surgical removal (BCS/Mastectomy), and should generally be done before BCS.")
        st.subheader("Breast-Conserving Surgery (BCS)/ Lumpectomy")
        st.write("Surgery that removes the tumor while also preserving healthy breast tissue by stripping only a tiny margin of the normal tissue surrounding the tumor. Taken if the tumor is small and in the earlier stage of invasive (stage 0 to stage II). Ideally, the tumor should be relatively small and confined to the breast or nearby lymph nodes. Before performing BCS, several diagnostic tests are conducted to assess the tumor's characteristics, including mammography, ultrasound, and MRI to confirm the cancer diagnosis and determine hormone receptor status (specific receptors on the surface of cancer cells that can bind to estrogen and progesterone, which lays the foundation for guiding the treatment procedure).")
        st.subheader("Hormone Therapy")
        st.write("Treatment that blocks the effects of hormones to reduce their production and inhibit the growth of hormone-sensitive cancer cells. All women with tumor sizes larger than 0.5 cm are recommended to continue hormone therapy for a range of 5-10 years.")
        if user_input2>0.5:
            st.write("Because your inputted tumor size is larger than 0.5 cm, we recommend hormone therapy in addition to the treatments above.")

# Section 3: More Info / research stuff
st.header('More Info')
st.subheader("Introduction")
st.write("Our project investigates proper treatment and classifications for Breast Cancer—one of the most prevalent and life-threatening diseases affecting women worldwide. Essentially, Malignant (cancerous) cells typically exhibit abnormal characteristics, including increased cell size, irregular shape, rapid cell division, etc. Benign cells, on the other hand, tend to maintain more regular and organized features.")
st.subheader("Model 1 (Original Model)")
st.write("Our first Machine Learning model implements a Logistic Regression Classification model that uses an API dataset of 32 columns to train. Other libraries used were numpy for linear algebra and pandas for data preprocessing. Overall, in optimizing our model to the greatest prediction accuracy possible, we found that dropping 0 data columns—except the id—produced a prediction accuracy of 98.6% with a runtime of 20.7 sec. More specific data reports are organized below:")
st.subheader("Imported Libraries")
st.write("StandardScaler (sklearn): Used for feature scaling.")
st.write("Train_test_split (sklearn): Used to split the dataset into training and testing sets.")
st.write("Accuracy_score (sklearn): Used to evaluate the accuracy of the classifier.")
st.write("Classification_report (sklearn): Used to generate a report on the classifications.")

st.subheader("Reading Data and Preprocessing")
st.write("Our model first reads a CSV file of our wisconsin breast cancer API dataset with a pandas DataFrame. Then, the target variable (diagnosis) is converted from categorical values ('Malignant' and 'Benign') to binary numeric values (1 and 0, respectively) Predictors (data columns) are stored in X while the target variable is stored in y.")

st.subheader("Feature Scaling and Setup")
st.write("Predictor values in X are normalized using StandardScaler. Our dataset is then split into training and testing sets. The test set size is 75% of the entire dataset, while the random_state is set to 100 for reproducibility.")

st.subheader("Model Training and Predictions")
st.write("First, we initialize our Logistic Regression model as LogReg. Our model is then trained on the training data using the fit method with X_train and y_train passed in as arguments. Now, our trained model is used to predict the target labels on the test set (X_test), and the predictions are stored in y_predict.")

st.subheader("Accuracy Reports")
st.write("The accuracy of our trained model is calculated using the accuracy_score function by comparing the predicted labels (y_predict) with the actual labels in the test set (y_test). Then, we use the classification_report function to provide a summary of the precision, recall, F1-score, and support for each class (0 and 1) in the test set.")

st.subheader("Model 2 (Optimized Model)")
st.write("Unlike our original model, our optimized model drops many unnecessary dataset columns in order to increase overall performance. While this does sacrifice 4.7% of our model’s prediction accuracy, it makes up for it with a reduced runtime of more than 3 seconds. Also, reducing the number of dataset columns to train on prevents the risk of overfitting, which occurs when a model strictly memorizes the training data instead of learning its underlying patterns, resulting in poor performance on other new datasets. In other words, even though we lost 4.7% prediction accuracy on this specific test set, on other test sets we are more likely to have a consistent and high prediction accuracy along with a faster runtime—as our model learns the actual patterns instead of just straight memorization from the training data. After many trial runs and test cases, we found that using predictors of 'perimeter_mean', 'radius_mean', 'texture_mean', 'area_mean', 'smoothness_mean', 'concavity_mean', and 'symmetry_mean' generate the optimal model with the fastest runtime while also maintaining the highest prediction accuracy.")
