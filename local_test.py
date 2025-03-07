import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GridSearchCV
import joblib
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from scipy.signal import savgol_filter
from sklearn.model_selection import train_test_split
from scipy.ndimage import zoom
import matplotlib.pyplot as plt


#Get cycle 1 data
X_cyc1 = pd.read_excel('data.xlsx','cyc1_spectra')
X_cyc1 = X_cyc1.iloc[:,5:]
X_cyc1 = X_cyc1.iloc[:,np.logical_and(X_cyc1.columns<4000,X_cyc1.columns>600)]
X_cyc1 = X_cyc1.to_numpy()
X_cyc1_reshaped = np.empty((X_cyc1.shape[0],1701))

for i in range(X_cyc1.shape[0]):
    X_cyc1_reshaped[i,:] = zoom(X_cyc1[i,:], (1701/X_cyc1.shape[1]))

for i in range(X_cyc1_reshaped.shape[0]):
    for j in range(X_cyc1_reshaped.shape[1]):
        X_cyc1_reshaped[i,j] = (X_cyc1_reshaped[i,j]-np.min(X_cyc1_reshaped[i,:]))/(np.max(X_cyc1_reshaped[i,:])-np.min(X_cyc1_reshaped[i,:]))

X_cyc1_reshaped = np.fliplr(X_cyc1_reshaped)

y_cyc1 = pd.read_excel('data.xlsx','cyc1_bulk')
y_cyc1 = y_cyc1['oc_actual']

#Get cycle 2 data
X_cyc2 = pd.read_excel('data.xlsx','cyc2_spectra')
X_cyc2 = X_cyc2.iloc[:,5:]
X_cyc2 = X_cyc2.iloc[:,np.logical_and(X_cyc2.columns<4000,X_cyc2.columns>600)]
X_cyc2 = X_cyc2.to_numpy()

y_cyc2 = pd.read_excel('data.xlsx','cyc2_bulk')
y_cyc2 = y_cyc2['oc_actual']
y_cyc2 = y_cyc2[~np.isnan(X_cyc2[:,1])]

X_cyc2 = X_cyc2[~np.isnan(X_cyc2[:,1]),:]
X_cyc2_reshaped = np.empty((X_cyc2.shape[0],1701))

for i in range(X_cyc2.shape[0]):
    X_cyc2_reshaped[i,:] = zoom(X_cyc2[i,:], (1701/X_cyc2.shape[1]))

for i in range(X_cyc2_reshaped.shape[0]):
    for j in range(X_cyc2_reshaped.shape[1]):
        X_cyc2_reshaped[i,j] = (X_cyc2_reshaped[i,j]-np.min(X_cyc2_reshaped[i,:]))/(np.max(X_cyc2_reshaped[i,:])-np.min(X_cyc2_reshaped[i,:]))

X_cyc2_reshaped = np.fliplr(X_cyc2_reshaped)

#create the test set 
X_test = np.concatenate((X_cyc1_reshaped[~np.isnan(y_cyc1),:],X_cyc2_reshaped[~np.isnan(y_cyc2),:]))
Y_test =  np.concatenate((y_cyc1[~np.isnan(y_cyc1)].to_numpy(),y_cyc2[~np.isnan(y_cyc2)].to_numpy()))

#load the model
reg = joblib.load('glob_model.mdl')

#test the model
round(r2_score(y_true=Y_test,y_pred=reg.predict(X_test)),2) #0.98
round(np.sqrt(mean_squared_error(y_true=Y_test,y_pred=reg.predict(X_test))),2) #0.16
round(np.sqrt(mean_squared_error(y_true=Y_test,y_pred=reg.predict(X_test[Y_test>0.5])))/(np.max(Y_test)-np.min(Y_test)),2)*100 #5.0

#dump the predicted carbons
np.savetxt('c_pred_cyc1.csv',reg.predict(X_cyc1_reshaped),delimiter=',')
np.savetxt('c_pred_cyc2.csv',reg.predict(X_cyc2_reshaped),delimiter=',')

