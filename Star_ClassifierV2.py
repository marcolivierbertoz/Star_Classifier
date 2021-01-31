# Loading packages ##########################################################################
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

pd.set_option('display.max_columns', None) # Viewing all the columns

# Loading Machine Learning Packages
# Classifier
from sklearn.svm import LinearSVC


###############################################################################################

# Writing title 
st.title('Star Classification')
# Writng description
st.header('Description')
st.write("""
This web application is created with the only goal to train the use of the _Streamlit_ package. 

With this web app, it's going to be used a machine learning classifier, with the aim to classify a new star created by the user.

Data inside the dataframe:
""")

left_column, middle_column, right_column = st.beta_columns(3)
left_column.write(""" 
* Absolute Temperature [K];
* Relative Luminosity [L/Lo];
* Relative Radius [R/Ro]; 
* Absolute Magnitude [Mv];
* Star Color:
    * White;
    * Red;
    * Blue;
    * Yellow;
    * Yellow-Orange.
""")

middle_column.write("""
* Spectral Class:
    * O (Hottest);
    * B;
    * A;
    * F;
    * G;
    * K;
    * M (Coolest Star)
""")

right_column.write("""
* Star Type: 
    * Red Dwarf;
    * Brown Dwarf;
    * White Dwarf;
    * Main Sequence;
    * SuperGiants;
    * HyperGiants.
""")
st.write(""" 
> Data Source: https://www.kaggle.com/deepu1109/star-dataset  
""")




# Creating function to change name
def change_name(df):
    for i in range(len(df)):
        if (df.loc[i,'Star_type'] == 0):
            df.loc[i,'Star_type'] = 'Brown Dwarf'
        elif (df.loc[i,'Star_type'] == 1):
            df.loc[i,'Star_type'] = 'Red Dwarf'
        elif (df.loc[i,'Star_type'] == 2):
            df.loc[i,'Star_type'] = 'White Dwarf'
        elif (df.loc[i,'Star_type'] == 3):
            df.loc[i,'Star_type'] = 'Main Sequence'
        elif (df.loc[i,'Star_type'] == 4):
            df.loc[i,'Star_type'] = 'Supergiant'
        elif (df.loc[i,'Star_type'] == 5):
            df.loc[i,'Star_type'] = 'Hyperginat'
    return df        

# Creating funciton to load the data and store it in the cache
@st.cache
def load_data():
    df=pd.read_csv('Star_Classes.csv')
    df=df.rename(columns={"Absolute magnitude(Mv)":"Absolute_magnitude(Mv)",
    "Temperature (K)":"Temperature_(K)","Star type":"Star_type","Star color":"Star_color",
    "Spectral Class":"Spectral_Class"})
    df=change_name(df)
    return df
    
df=load_data()

# Plotting the data of df on 3d graph ######################################################################################################
st.subheader('3D Plot of the stars present inside the dataset')
fig = px.scatter_3d(df, x='Temperature_(K)',y='Luminosity(L/Lo)',z='Absolute_magnitude(Mv)', color='Star_type')
st.plotly_chart(fig)


############################################################################################################################################


# Creating Sidebar ######################################################################
st.sidebar.header('User Input')
# Sidebar parameters
params = {
    'Temperature':st.sidebar.slider('Temperature [K]', min(df['Temperature_(K)']),max(df['Temperature_(K)']),1),
    'Luminosity': st.sidebar.slider('Luminosity [L/lo]', min(df['Luminosity(L/Lo)']),max(df['Luminosity(L/Lo)']),1.0),
    'Radius': st.sidebar.slider('Radius [R/Ro]', min(df['Radius(R/Ro)']),max(df['Radius(R/Ro)']),1.0),
    'Absolute_magnitude': st.sidebar.slider('Absolute magnitude [Mv]', min(df['Absolute_magnitude(Mv)']),max(df['Absolute_magnitude(Mv)']),1.0),
    'Star_color': st.sidebar.selectbox('Star color', ('White', 'Red', 'Blue', 'Yellow', 'Yellow-Orange')),
    'Spectral_class': st.sidebar.selectbox('Spectral class', ('O', 'B', 'A', 'F', 'G', 'K', 'M')),
}

########################################################################################

# Defining the user input ############################################################

def user_input():
    temperature = params['Temperature']
    luminosity = params['Luminosity']
    radius = params['Radius']
    absolute_magnitude = params['Absolute_magnitude']
    star_color = params['Star_color']
    spectral_class = params['Spectral_class']

    data = {
        'Temperature_(K)': temperature,
        'Luminosity(L/Lo)' : luminosity,
        'Radius(R/Ro)' : radius,
        'Absolute_magnitude(Mv)': absolute_magnitude,
        'Star_color': star_color,
        'Spectral_Class': spectral_class
    }

    features = pd.DataFrame(data, index=[0])

    return features

user_input_df = user_input()
###########################################################################################

# Implementing Machine Learning ###########################################################

# Preparing the data
X_data = df.drop('Star_type', axis=1)
y_target = df.Star_type

# Concat the user data
X_data = pd.concat([user_input_df,X_data], axis=0)

# One-hot encoding
X_data = pd.get_dummies(X_data, columns=['Star_color','Spectral_Class'], drop_first=True)

# Retriving the data to be predictied
Data_predict = X_data[:1]

# Removing the first row
X_data = X_data.iloc[1:]


# Creatin function for training the classifier and storing in the cache

@st.cache
def train_class():
    # Creating the classifier
    lsvc_classifier = LinearSVC(C=1.0, max_iter = 1000000)
    # Training the classifier
    lsvc_classifier.fit(X_data,y_target)
    return lsvc_classifier

lsvc = train_class()
###########################################################################################

# Predicting star type ##################################################################
# Adding button
btn = st.sidebar.button("Predict")  

# Defyining function for prediction

def predict_star():
    prediction = lsvc.predict(Data_predict)
    st.subheader('Prediction')
    st.write(prediction)

if btn:
    predict_star()
	
else:
	pass

###########################################################################################

        
