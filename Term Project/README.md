This project is presented in the form of a websited created by using the Streamlit API. The Streamlit API only run on Python 3.7 and Python 3.8. Please update the python version before running the project. To run the project, please follow these steps:

- Click on the file app.py, then install the Streamlit API in the Python environment using this command : pip install streamlit or this command: pip3 install streamlit to ensure that streamlit is installed.

- Here are the list of libraries that needed to be installed for the project: sklearn, numpy, tensorflow, matplotlib.pyplot, json, os, pandas, math, alpha_vantage, datetime, pandas_datareader, and urllib.request.

- Run the website on the localhost by using this command: streamlit run app.py.

- Note: as the Streamlit API has recently been updated to a new version, a small bug, which is not yet resovled, has occured that sometimes, the sliders won't return the selected value. When an error occured because of the file dataPreprocess.py, please change the value parameters of the sliders in the file app.py, save the changes, and reload the page.

- Here are all of the sliders, in the file app.py, they are from line 45 to line 48, please change the 'value' parameter if the error occured:

selectedSplit = st.sidebar.slider('Choose a spilt ratio to split the data set into input and output set', min_value=0.1, max_value=0.99, value=0.85)

selectedTimeSteps = st.sidebar.slider('Choose a number for the time step', min_value=10, max_value=100, value=50)

selectedBatchSize = st.sidebar.slider('Choose a number for the batch size', min_value=10, max_value=100, value=32)

selectedEpochNum = st.sidebar.slider('Choose the number of epoch', min_value=1, max_value=100, value=10)

The dataset used in this project is provided by Alpha Vantage API: https://www.alphavantage.co/documentation/

To predict the stock trend:

- Enter a letter in the search bar to search for a company. You can either select a different company in the drop down bar or don't.

- You should not change the sliders. If you change the sliders, the bug mentioned above may occurs. If the bug occurs, please follow the above instruction to solve the error.

- You can also check the checkbox to use the K-Fold cross-validation. Have not detected any bug with this checkbox. If you check the box and an error occurs, please replace the line of code at line 49 to this line: selectedKFold = st.sidebar.checkbox('Do you want to use K-Fold cross-validation?', value=True)

- Click the button Predict the stock trend to run the program.

- Different companies will lead to different predictions of log return.

- The models are saved in the savedModel folder.