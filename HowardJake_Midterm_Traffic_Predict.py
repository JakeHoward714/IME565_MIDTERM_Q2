# Import necessary libraries
import streamlit as st
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Set up the title and description of the app
st.title('Traffic Volume Predictor') 
st.write("Utilize our advance Machine Learning application to predict traffic volume.")

# Display an image of traffic
st.image('traffic_image.gif', use_column_width=True)


# Load the pre-trained model from the pickle file
dt_pickle = open('xgboost_traffic_volume.pickle', 'rb') 
bst = pickle.load(dt_pickle) 
dt_pickle.close()


# SIDEBAR CREATION
# Display an image of traffic
st.sidebar.image('traffic_sidebar.jpg', width = 400)


# Create a sidebar for input collection
st.sidebar.header('Traffic Features Input')

#Write short text on sidebar
st.sidebar.write('You can manually select traffic features or upload a data file (csv)')

# CREATE EXPANDERS 
# file upload option
with st.sidebar.expander("Option 1: Upload CSV File"):
    # Create option for csv upload
    traffic_upload = st.file_uploader("Upload a CSV containing the traffic details")

    #DISPLAY EXAMPLE CSV
    st.header('Sample Data Format for Upload')
    # Import example dataframe
    df = pd.read_csv('Traffic_Volume.csv')

    # Convert the 'date_time' column to month, weekday, hour
    df['date_time'] = pd.to_datetime(df['date_time'], format='%m/%d/%y %H:%M')

    # Extract specific month, day, hour and add to dataframe as new column
    df['month'] = df['date_time'].dt.month_name()       # Month name
    df['weekday'] = df['date_time'].dt.day_name()   # Day name 
    df['hour'] = df['date_time'].dt.hour                # Hour (0-23)
    
    # Format correctly
    df_clean = df.drop(columns=['date_time','traffic_volume'])
    #Write to streamlit
    st.write(df_clean.head())
    # Message
    st.warning('Ensure your uploaded file has the same column names and data types as shown above')

# Manual entry option
with st.sidebar.expander("Option 2: Fill Out Form"):
    with st.form('Enter the traffic details manually using the form below'):
        st.write('Enter the traffic details manually using the form below')
        #inputs
        holiday = st.selectbox('Choose whether today is a designated holiday or not', options=['None','Labor Day','Thanksgiving Day','Christmas Day',
         'New Years Day','Martin Luther King Jr Day','Columbus Day','Veterans Day','Washingtons Birthday','Memorial Day',
         'Independence Day','State Fair'], help="Is today a holiday? If not leave unasnwered")
        # If no holiday is selected (empty string), set holiday to None
        if holiday == "None":
            holiday = None

        temp = st.number_input('Average temperature in Kelvin', min_value = df['temp'].min(), max_value = df['temp'].max(), value = 293.15, help="Range [0, 310.07] K")
        rain = st.number_input('Amount in mm of rain that occurred in the hour', min_value = df['rain_1h'].min(), max_value = df['rain_1h'].max(), help="Range [0, 9831.3] mm")
        snow = st.number_input('Amount in mm of snow that occurred in the hour', min_value = df['snow_1h'].min(), max_value = df['snow_1h'].max(), help="Range [0, 0.51] mm")
        cloud = st.number_input('Percentage of cloud cover', min_value = df['clouds_all'].min(), max_value = df['clouds_all'].max(), help="Give a percentage from 0-100%")
        weather = st.selectbox('Choose the current weather', options=['Clouds','Clear','Mist','Rain','Snow','Drizzle','Haze','Thunderstorm',
                                                            'Fog','Smoke','Squall'], help='Choose the current type of weather from selection box')
        month = st.selectbox('Choose Month' ,options=['January','February','March','April','May','June','July','September','October','November','December'], help='Choose current month')
        day = st.selectbox('Choose day of week' ,options=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'], help='Choose current day of week')
        hour = st.selectbox('Choose hour' ,options=['0','1','2','3','4','5','6','7','8','9','10',
                                        '11','12','13','14','15','16','17','18','19','20','21',
                                        '22','23'], help='Choose current hour of day (0-23)')
        submit_button = st.form_submit_button("Predict")


# Inform user to select a input method if not yet selected
#if traffic_upload is None and not submit_button:
    #st.info("Please select a data input method to proceed")
    
    
# Recall original data frame for dummy creation
df_orig = df.drop(columns=['date_time','traffic_volume'])
len_orig = len(df_orig)

# RUN PREDICTIONS
# Check to see if csv file as been uploaded
if traffic_upload is not None:
   # Say success
   st.success('CSV file uploaded successfully')
   user_df = pd.read_csv(traffic_upload)

   #Create slider for alpha value selection
   alpha_input = st.slider('Select alpha value for prediction intervals', min_value=0.01, max_value=0.50, step=0.01)

   #Create Neccessary Dummies
   # Input features
   features = user_df[['holiday', 'temp', 'rain_1h', 'snow_1h', 'clouds_all', 'weather_main', 'month', 'weekday', 'hour']] 

   # Combine input data into orginal data frame 
   # Append features to df_orig
   combined_df = pd.concat([df_orig, features], ignore_index=True)

   # One-hot encoding to handle categorical variables
   combined_features_encoded = pd.get_dummies(combined_df)

   # Pull out user CSV from combined dataframe
   features_encoded  = combined_features_encoded.iloc[len_orig:].reset_index(drop=True)

   #Run predictions for each row of data
   # Use predict() on the entire DataFrame at once
   predictions = bst.predict(features_encoded)

   # Get the prediction with its intervals
   predictions, intervals = bst.predict(features_encoded, alpha = alpha_input)
   
   # Storing results in original dataframe
   user_df["Predicted Traffic Volume"] = predictions.round(0)
   user_df["Lower Traffic Volume Limit"] = np.maximum(intervals[:, 0], 0).round(0)  # Ensure no negative lower limit
   user_df["Upper Traffic Volume Limit"] = intervals[:, 1].round(0)

   #Display
   st.header(f"Here is your Traffic Volume Prediction Results with {(1 - alpha_input) * 100:.2f}% Confidence Interval")
   st.write(user_df)

else:
    # write success
    st.success("Form data submitted successfully")

    #Create slider for alpha value selection
    alpha_input = st.slider('Select alpha value for prediction intervals', min_value=0.01, max_value=0.50, step=0.01)
    
    # Encode the form inputs for model prediction
    encode_df = df_orig.copy()
    
    # Combine the list of user data as a row to default_df
    encode_df.loc[len(encode_df)] = [holiday,temp,rain,snow,cloud,weather,month,day,hour]

    # Ensure the data types match the original data's types
    encode_df = encode_df.astype(df_orig.dtypes)
    
    # Get Dummies 
    df_dummy_form = pd.get_dummies(encode_df)

    # Extract encoded user data
    userform_encoded_df = df_dummy_form.tail(1)

    #Run prediction for data
    prediction, intervals = bst.predict(userform_encoded_df, alpha = alpha_input)
    pred_value = prediction[0]
    lower_limit = intervals[:, 0]
    upper_limit = intervals[:, 1][0][0]

    # Ensure limits are within [0, 1]
    lower_limit = max(0, lower_limit[0][0])
    
    #Display
    st.header("Predicted Traffic Volume...")
    st.metric('Predicted Traffic Volume', f"{pred_value:.2f}")
    st.write(f"CONFIDENCE INTERVAL ({(1 - alpha_input)*100:.2f}%): [{lower_limit:.2f}, {upper_limit:.2f}]")


    

# Additional tabs for DT model performance
st.subheader("Model Insights")
tab1, tab2, tab3, tab4 = st.tabs(["Feature Importance", 
                            "Histogram of Residuals", 
                            "Predicted Vs. Actual", 
                            "Coverage Plot"])
with tab1:
    st.write("### Feature Importance")
    st.image('feature_imp.svg')
    st.caption("Relative importance of features in prediction.")
with tab2:
    st.write("### Histogram of Residuals")
    st.image('residual_plot.svg')
    st.caption("Distribution of residuals to evaluate prediction quality.")
with tab3:
    st.write("### Plot of Predicted Vs. Actual")
    st.image('pred_vs_actual.svg')
    st.caption("Visual comparison of predicted and actual values.")
with tab4:
    st.write("### Coverage Plot")
    st.image('coverage.svg')
    st.caption("Range of predictions with confidence intervals.")