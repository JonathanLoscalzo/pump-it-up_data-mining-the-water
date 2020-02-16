from datetime import datetime
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, make_scorer, f1_score, classification_report, plot_confusion_matrix


def timeit(method):
    def timed(*args, **kw):
        ts = datetime.now()
        print(10*'*', 'Start', 10*'*')
        result = method(*args, **kw)
        te = datetime.now()
        elapsed = te - ts
        print("Time spent : ", elapsed)
        print(10*'*', 'End', 10*'*')
        print()
        return result
    return timed

def compute_weights(y):
    weights = compute_class_weight("balanced", np.unique(y), y)
    return { i:k for i,k in enumerate(weights)}

def plot_feature_importances(clf, columns):
    plt.figure()
    plt.title("Feature importances")
    feat_importances = pd.Series(clf.feature_importances_, index=columns)
    feat_importances.nlargest(10).plot(
        kind='barh', 
        xerr=feat_importances.std(),
        align="center"
    )
    plt.show()
    
def print_line(text="***"):
    print(10*'*', text, 10*'*')
    
def print_metrics(clf, X, y, title=""):
    print_line()
    print_line(f'Metrics for: {title}')
    print(f'accuracy: {accuracy_score(clf.predict(X), y)}')
    print_line("Classification Report")
    print(classification_report(clf.predict(X), y))
    print_line("Confusion Matrix")
    plot_confusion_matrix(
        clf, 
        X, 
        y, 
        normalize='true'
    )

