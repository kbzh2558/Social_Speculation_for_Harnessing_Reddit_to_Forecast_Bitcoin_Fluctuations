# import package
import numpy as np
from sklearn.metrics import r2_score
import pandas as pd

class LinearRegression:
    def __init__(self, add_bias=True, fit_intercept=True, normalize=False):
        self.add_bias = add_bias
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        pass

    def fit(self, x, y):
        """
        Fit the model and find the weights matrix w that optimizes the loss function.
        """

        if isinstance(x, pd.Series):
            x = x.to_numpy()
        if isinstance(y, pd.Series):
            y = y.to_numpy()

        if x.ndim == 1:
            x = x[:, None]
        if y.ndim == 1:
            y = y[:, None]
        N = x.shape[0]
        if self.add_bias:
            x = np.column_stack([x,np.ones(N)])
        self.w = np.linalg.pinv(x).dot(y)

        return self

    def predict(self, x):
        if isinstance(x, pd.Series):
            x = x.to_numpy()
            
        N = x.shape[0]
        if self.add_bias:
            x = np.column_stack([x,np.ones(N)])
        yh = x@self.w
        return pd.Series(yh.flatten())

    def MSE(self, y, yh):
        MSE = np.square(np.subtract(y,yh)).mean()
        return MSE
    
    def SSE(self, y, yh):
        MSE = np.square(np.subtract(y,yh)).sum()
        return MSE

    def score(self, X, y):
        y_pred = self.predict(X)
        return r2_score(y, y_pred)
    
    @property
    def intercept_(self):
        """
        Extracts the intercept of the fitted model.
        """
        if hasattr(self, 'w'):
            # Convert coefficients to list and return
            return self.w.flatten().tolist()[-1]
        else:
            raise ValueError("Model has not been fitted yet. Please call the 'fit' method before extracting intercept.")

    @property    
    def coef_(self):
        """
        Extracts the coefficients of the fitted model into a list.
        """
        if hasattr(self, 'w'):
            # Convert coefficients to list and return
            return self.w.flatten().tolist()[:-1]
        else:
            raise ValueError("Model has not been fitted yet. Please call the 'fit' method before extracting coefficients.")
        
    def get_params(self, deep=True):
        return {"fit_intercept": self.fit_intercept, "normalize": self.normalize}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self