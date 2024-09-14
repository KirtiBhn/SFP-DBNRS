import pandas as pd
from imblearn.under_sampling import TomekLinks
import numpy as np

def tom(dataset):
    X = dataset.iloc[:, 0:-1]
    Y = dataset.iloc[:, -1]
    

    
    undersample = TomekLinks()
    
    x_under_sampled, y_under_sampled = undersample.fit_resample(X, Y)

    # Build the resulting under sampled dataframe
    result = pd.DataFrame(x_under_sampled)

    # Restore the y values
    y_under_sampled = pd.Series(y_under_sampled)
    
    result[dataset.columns[-1]] = y_under_sampled    

    return result