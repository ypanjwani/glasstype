import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score 

# ML classifier Python modules
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Loading the dataset.
@st.cache()
def load_data():
    file_path = "https://s3-student-datasets-bucket.whjr.online/whitehat-ds-datasets/glass-types.csv"
    df = pd.read_csv(file_path, header = None)
    # Dropping the 0th column as it contains only the serial numbers.
    df.drop(columns = 0, inplace = True)
    column_headers = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'GlassType']
    columns_dict = {}
    # Renaming columns with suitable column headers.
    for i in df.columns:
        columns_dict[i] = column_headers[i - 1]
        # Rename the columns.
        df.rename(columns_dict, axis = 1, inplace = True)
    return df

glass_df = load_data() 

# Creating the features data-frame holding all the columns except the last column.
X = glass_df.iloc[:, :-1]

# Creating the target series that holds last column.
y = glass_df['GlassType']

# Spliting the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

@st.cache()
def prediction(model,ri,na,mg,al,si,k,ca,ba,fe):
  glass_pred=model.predict([[ri,na,mg,al,si,k,ca,ba,fe]])
  glass_type=glass_pred[0]
  if glass_type == 1:
    return "building windows float processed".upper()
  elif glass_type == 2:
    return "building windows non float processed".upper()

  elif glass_type == 3:
    return "vehicle windows float processed".upper()

  elif glass_type == 4:
    return "vehicle windows non float processed".upper()

  elif glass_type == 5: 
    return "containers".upper()

  elif glass_type == 6: 
    return "tableware".upper()
  
  else:
    return "headlamp".upper()


st.title("Glass Type Predictor")
st.sidebar.title("Exploratory Data Analysis")


if st.sidebar.checkbox("Show raw data"):
  st.subheader("Full Datasets")
  st.dataframe(glass_df)


st.sidebar.subheader("Scatter plot")
# Choosing x-axis values for the scatter plot.
# Add a multiselect in the sidebar with the 'Select the x-axis values:' label
# and pass all the 9 features as a tuple i.e. ('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe') as options.
# Store the current value of this widget in the 'features_list' variable.
feature_lst=st.sidebar.multiselect("select x values",('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'))

st.set_option('deprecation.showPyplotGlobalUse', False)
for feature in feature_lst:
  st.subheader(f"Scatter plot between {feature} and glass type")
  plt.figure(figsize=(15,6))
  sns.scatterplot(x=feature,y="GlassType",data=glass_df)
  st.pyplot()

st.sidebar.subheader("Histogram")
# Choosing features for histograms.
hist_features=st.sidebar.multiselect("select feature values",('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'))
# Create histograms.
for i in hist_features:
  st.subheader(f"Histogram of {i}")
  plt.figure(figsize=(15,6))
  plt.hist(glass_df[i],bins="sturges",edgecolor="green")
  st.pyplot()

st.sidebar.subheader("Box Plot")

# Choosing columns for box plots.
box_plot_cols = st.sidebar.multiselect("Select the columns to create box plots:",
                                            ('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'GlassType'))

# Create box plots.
for col in box_plot_cols:
    st.subheader(f"Box plot for {col}")
    plt.figure(figsize = (12, 2))
    sns.boxplot(glass_df[col])
    st.pyplot()

plot_types=st.sidebar.multiselect("Select the plot :",("Countplot","Piechart","Correlation Heatmap"))

if "Countplot" in plot_types:
    st.subheader("Countplot")
    sns.countplot("GlassType",data=glass_df)
    st.pyplot()
if "Piechart" in plot_types:
    st.subheader("Piechart")
    pie_value=glass_df["GlassType"].value_counts()
    plt.figure(figsize=(16,6))
    plt.pie(pie_value,labels=pie_value.index,autopct="%1.2f%%")
    st.pyplot()
if "Correlation Heatmap" in plot_types:
    st.subheader("Correlation Heatmap")
    sns.heatmap(glass_df.corr(),annot=True)
    st.pyplot()

st.sidebar.subheader("Select your values:")
ri = st.sidebar.slider("Input Ri", float(glass_df['RI'].min()), float(glass_df['RI'].max()))
na = st.sidebar.slider("Input Na", float(glass_df['Na'].min()), float(glass_df['Na'].max()))
mg = st.sidebar.slider("Input Mg", float(glass_df['Mg'].min()), float(glass_df['Mg'].max()))
al = st.sidebar.slider("Input Al", float(glass_df['Al'].min()), float(glass_df['Al'].max()))
si = st.sidebar.slider("Input Si", float(glass_df['Si'].min()), float(glass_df['Si'].max()))
k = st.sidebar.slider("Input K", float(glass_df['K'].min()), float(glass_df['K'].max()))
ca = st.sidebar.slider("Input Ca", float(glass_df['Ca'].min()), float(glass_df['Ca'].max()))
ba = st.sidebar.slider("Input Ba", float(glass_df['Ba'].min()), float(glass_df['Ba'].max()))
fe = st.sidebar.slider("Input Fe", float(glass_df['Fe'].min()), float(glass_df['Fe'].max()))

st.sidebar.subheader("Choose Classifier")

# Add a selectbox in the sidebar with label 'Classifier'.
# and with 2 options passed as a tuple ('Support Vector Machine', 'Random Forest Classifier').
# Store the current value of this slider in a variable 'classifier'.

classifier = st.sidebar.selectbox("Classifier", 
                                 ('Support Vector Machine', 'Random Forest Classifier', 'Logistic Regression'))

if classifier == 'Support Vector Machine':
    st.sidebar.subheader("Model Hyperparameters")
    c_value = st.sidebar.number_input("C (Error Rate)", 1, 100, step = 1)
    kernel_input = st.sidebar.radio("Kernel", ("linear", "rbf", "poly"))
    gamma_input = st. sidebar.number_input("Gamma", 1, 100, step = 1)

    # If the user clicks 'Classify' button, perform prediction and display accuracy score and confusion matrix.
    # This 'if' statement must be inside the above 'if' statement.
    if st.sidebar.button('Classify'):
        st.subheader("Support Vector Machine")
        svc_model = SVC(C = c_value, kernel = kernel_input, gamma = gamma_input)
        svc_model.fit(X_train,y_train)
        y_pred = svc_model.predict(X_test)
        accuracy = svc_model.score(X_test, y_test)
        glass_type = prediction(svc_model, ri, na, mg, al, si, k, ca, ba, fe)
        st.write("The Type of glass predicted is:", glass_type)
        st.write("Accuracy", accuracy.round(2))
        plot_confusion_matrix(svc_model, X_test, y_test)
        st.pyplot()


if classifier=="Random Forest Classifier":
    st.sidebar.subheader("Model Hyperparameters")
    estimators=st.sidebar.number_input("Estimators",50,100,step=10)
    depth=st.sidebar.number_input("Depth",1,100,step=1)

    if st.sidebar.button('Classify'):
        st.subheader("Random Forest Classifier")
        svc_model = RandomForestClassifier(n_estimators=estimators,max_depth=depth,n_jobs=-1)
        svc_model.fit(X_train,y_train)
        y_pred = svc_model.predict(X_test)
        accuracy = svc_model.score(X_test, y_test)
        glass_type = prediction(svc_model, ri, na, mg, al, si, k, ca, ba, fe)
        st.write("The Type of glass predicted is:", glass_type)
        st.write("Accuracy", accuracy.round(2))
        plot_confusion_matrix(svc_model, X_test, y_test)
        st.pyplot()

