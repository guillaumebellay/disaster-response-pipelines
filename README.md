# Disaster response pipelines

## Table of contents  
1- Installation  
2- File description  
3- Project objective  
4- Instructions 


## File descriptions

```bash
│   .gitattributes
│   README.md
│
├───app
│   │   run.py                      # flask file that runs app
│   │
│   └───templates
│           go.html                 # classification result page of web app
│           master.html             # main page of web app
│
├───data
│       DisasterResponse.db         # database to save clean data
│       disaster_categories.csv     # data to process (categories)
│       disaster_messages.csv       # data to process (messages)
│       process_data.py             # clean data
│
└───models
        classifier.pkl              # saved model
        train_classifier.py         # train ML model
```


## Project objective

In this course, you've learned and built on your data engineering skills to expand your opportunities and potential as a data scientist.
In this project, you'll apply these skills to analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages.
In the Project Workspace, you'll find a data set containing real messages that were sent during disaster events. 
You will be creating a machine learning pipeline to categorize these events so that you can send the messages to an appropriate disaster relief agency.
Your project will include a web app where an emergency worker can input a new message and get classification results in several categories. 
The web app will also display visualizations of the data. 
This project will show off your software skills, including your ability to create basic data pipelines and write clean, organized code!

Below are a few screenshots of the web app.


Project Components
There are three components you'll need to complete for this project.

1. ETL Pipeline
In a Python script, process_data.py, write a data cleaning pipeline that:

Loads the messages and categories datasets
Merges the two datasets
Cleans the data
Stores it in a SQLite database
2. ML Pipeline
In a Python script, train_classifier.py, write a machine learning pipeline that:

Loads data from the SQLite database
Splits the dataset into training and test sets
Builds a text processing and machine learning pipeline
Trains and tunes a model using GridSearchCV
Outputs results on the test set
Exports the final model as a pickle file
3. Flask Web App
We are providing much of the flask web app for you, but feel free to add extra features depending on your knowledge of flask, html, css and javascript.
For this part, you'll need to:

Modify file paths for database and model as needed
Add data visualizations using Plotly in the web app. One example is provided for you

Project Workspace - ETL
The first part of your data pipeline is the Extract, Transform, and Load process. Here, you will read the dataset, clean the data, and then store it in a SQLite database. We expect you to do the data cleaning with pandas. 
To load the data into an SQLite database, you can use the pandas dataframe .to_sql() method, which you can use with an SQLAlchemy engine.
Feel free to do some exploratory data analysis in order to figure out how you want to clean the data set. 
Though you do not need to submit this exploratory data analysis as part of your project, you'll need to include your cleaning code in the final ETL script, process_data.py.

Project Workspace - Machine Learning Pipeline
For the machine learning portion, you will split the data into a training set and a test set. 
Then, you will create a machine learning pipeline that uses NLTK, as well as scikit-learn's Pipeline and GridSearchCV to output a final model that uses the message column to predict classifications for 36 categories (multi-output classification). Finally, you will export your model to a pickle file. After completing the notebook, you'll need to include your final machine learning code in train_classifier.py.

Data Pipelines: Python Scripts
After you complete the notebooks for the ETL and machine learning pipeline, you'll need to transfer your work into Python scripts, process_data.py and train_classifier.py.
If someone in the future comes with a revised or new dataset of messages, they should be able to easily create a new model just by running your code. 
These Python scripts should be able to run with additional arguments specifying the files used for the data and model.

Example:
python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db

python train_classifier.py ../data/DisasterResponse.db classifier.pkl
Templates for these scripts are provided in the Resources section, as well as the Project Workspace IDE.
The code for handling these arguments on the command line is given to you in the templates.

Flask App
In the last step, you'll display your results in a Flask web app. We have provided a workspace for you with starter files.
You will need to upload your database file and pkl file with your model.
This is the part of the project that allows for the most creativity. 
So if you are comfortable with html, css, and javascript, feel free to make the web app as elaborate as you would like.

In the starter files, you will see that the web app already works and displays a visualization.
You'll just have to modify the file paths to your database and pickled model file as needed.

There is one other change that you are required to make. We've provided code for a simple data visualization. 
Your job will be to create two additional data visualizations in your web app based on data you extract from the SQLite database. 
You can modify and copy the code we provided in the starter files to make the visualizations.





