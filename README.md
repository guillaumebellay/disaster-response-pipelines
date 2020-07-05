# Disaster response pipelines

## Table of contents  
1- Installation  
2- File descriptions  
3- Project objective  
4- Instructions 

## Installation
You need to install the required libraries in the `requirements.txt`. Run the command below.  
```
pip install -r requirements.txt
```

## File descriptions

```bash
│   .gitattributes
│   README.md
│   requirements.txt
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
│       process_data.py             # clean data (ETL)
│
└───models
        classifier.pkl              # saved model
        train_classifier.py         # train ML model
```

## Project objective

In this project, I analyzed disaster data from Figure Eight to build a model for an API that classifies disaster messages. These real messages were sent during disaster events.  
I created a machine learning pipeline to categorize these events so that you can send the messages to an appropriate disaster relief agency.  

The first part of the data pipeline is the Extract, Transform, and Load process. Here, we read the dataset, clean the data, and then store it in a **SQLite database**. 
The second step is to create a machine learning pipeline that uses **NLTK**, as well as **scikit-learn's Pipeline and GridSearchCV** to output a final model that uses the message column to predict classifications for 36 categories (multi-output classification).  
Finally we display the results in a **Flask web app**. 

## Instructions  

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        ```
        python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
        ```
    - To run ML pipeline that trains classifier and saves model
        ```
        python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
        ```

2. Run the following command in the app's directory to run your web app.
    ```
    python run.py
    ```

3. Go to [http://127.0.0.1:3001/](http://127.0.0.1:3001/)  



Below are a few screenshots of the web app.   

![Web app](https://user-images.githubusercontent.com/60384891/86534795-a72bc400-bedb-11ea-97d3-1edb1e83bc4b.png)  


### Example:  

*How can we find help and food ?* 

![example](https://user-images.githubusercontent.com/60384891/86534822-db9f8000-bedb-11ea-9d92-a5db95103e88.png)




