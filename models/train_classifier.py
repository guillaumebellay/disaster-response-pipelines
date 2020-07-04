import sys
import pandas as pd
from sqlalchemy import create_engine

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.externals import joblib

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download(['words','punkt','stopwords','wordnet'])

def load_data(database_filepath):
    """
    Load data from database
    
    Parameters:
    -----------
    database_filepath: str. Filepath of the database.
    
    Returns:
    --------
    X: DataFrame. Input variables
    y: DataFrame. Output variables.
    """
    # Create engine
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('disaster', engine)

    # Select input and output data
    X = df['message']
    y = df.drop(['message', 'id', 'original', 'genre'], axis=1)
    categories = y.columns
    return X, y, categories

def tokenize(text):
    """
    Process text data
    
    Parameters:
    -----------
    text: str. Messages
    
    Returns:
    --------
    clean_tokens: list. List of words
    """
    # Split text into words.
    tokens = word_tokenize(text)

    # Lemmatize words and convert all uppercase caracters into
    # lowercase caracters.
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    Build model.

    Parameters:
    -----------
    None

    Returns:
    --------
    grid. Optimal model
    """
 
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    ## Find the optimal model
    parameters = { 
    'clf__estimator__n_estimators': [20, 50],
    'clf__estimator__max_features': ['auto', 'sqrt']}

    grid = GridSearchCV(pipeline, param_grid=parameters,
                        n_jobs=-1)

    return grid


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate performances : print classification report
    
    Parameters:
    -----------
    model. Model to evaluate
    X_test: DataFrame. Dataset used to evaluate our model.
    y_test: DataFrame. Dataset used to evaluate our model.
    category_names: list. List of categories.
    
    Returns:
    -------
    None    
    """
    y_pred = model.predict(X_test)
    print(classification_report(Y_test,y_pred,target_names=category_names))


def save_model(model, model_filepath):
    """
    Save model
    
    Parameters:
    -----------
    model. Model to save. 
    model_filepath : str. Filepath of the model.
    
    Returns:
    --------
    None
    """
    joblib.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
