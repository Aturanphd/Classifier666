import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,classification_report
from config import DEFAULT_MODELS

warnings.filterwarnings("ignore")

def test_data(X,y,test_size=.3,random_state=42, scale_data=True)-> pd.DataFrame:

    """
    Test multiple classification models and return their performance metric values.

    :param X: Feature matrix
    :param y: Target vector
    :param test_size: Proportion of the dataset to include in the test set
    :param random_state: Random seed for reproducibility
    :param scale_data: Whether or not to scale the data before fitting models
    :return:
        DataFrame: Results comparing all classifiers
    """

    X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=test_size, random_state=random_state,stratify=y)
    if scale_data:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    results=[]

    for name, model in DEFAULT_MODELS:
        try:
            model.fit(X_train,y_train)
            y_pred=model.predict(X_test)
            acc=accuracy_score(y_test,y_pred)
            report=classification_report(y_test,y_pred,output_dict=True)

            results.append({
                'Classifier': name,
                'Accuracy': acc,
                'Precision (weighted)': report['weighted avg']['precision'],
                'Recall (weighted)': report['weighted avg']['recall'],
                'F1-Score (weighted)': report['weighted avg']['f1-score'],
            })

        except Exception as e:
            print(f"Error with {name}: {str(e)}")
            results.append({
                'Classifier': name,
                'Accuracy': None,
                'Precision (weighted)': None,
                'Recall (weighted)': None,
                'F1-Score (weighted)': None,
            })
    result_df = pd.DataFrame(results)
    result_df=result_df.sort_values(by='Accuracy',ascending=False).reset_index(drop=True)
    return result_df


if __name__=="__main__":
        from sklearn.datasets import load_iris

        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)

        iris=load_iris()
        X=iris.data
        y=iris.target
        df=test_data(X,y)
        print(df)
