#在Streamlit中生成数据可视化网站
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.write("""
# Simple Iris Flower Prediction App
This app predicts the **Iris flower** type!
""")
st.sidebar.header('User Input Parameters')

def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features
df = user_input_features()
st.subheader('User Input parameters')
st.write(df)

iris = datasets.load_iris()
X = iris.data
Y = iris.target
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
clf = RandomForestClassifier()
clf.fit(X_train, Y_train)
prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.subheader('Class labels and their corresponding index number')
st.write(iris.target_names)

st.subheader('Prediction')
st.write(iris.target_names[prediction])
st.write(prediction)

st.subheader('Prediction Probability')

st.write(prediction_proba)
st.write('Accuracy: ', accuracy_score(Y_test, clf.predict(X_test)))

st.subheader('Data Visualization')
st.write('Correlation Matrix')
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True)
st.pyplot()
st.write('Pairplot')

sns.pairplot(df)
st.pyplot()
st.write('Histogram')

plt.hist(df)
st.pyplot()
st.write('Barplot')
plt.bar(df.columns, df.values[0])
st.pyplot()
st.write('Boxplot')
plt.boxplot(df.values[0])
st.pyplot()
