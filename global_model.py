import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GridSearchCV
import joblib
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from scipy.ndimage import zoom
import matplotlib.pyplot as plt


###Full scale model
#read the global model data
X = pd.read_csv('X_glob_MIR_OC.csv',header=None).to_numpy()
Y = pd.read_csv('Y_glob_MIR_OC.csv',header=None).to_numpy()
Y = Y.reshape(-1)

#split train and test
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2)

#train the model on training data
reg = MLPRegressor(hidden_layer_sizes=(10,10),activation='relu',solver='adam',learning_rate_init=0.0001,random_state=1)
reg.fit(X_train,y_train)

#validate the model
r2_1 = r2_score(y_true=y_test,y_pred=reg.predict(X_test)) #0.98
rmse_1 = np.sqrt(mean_squared_error(y_true=y_test,y_pred=reg.predict(X_test))) #1.65
round(np.sqrt(mean_squared_error(y_true=y_test,y_pred=reg.predict(X_test)))/(np.max(y_test)-np.min(y_test)),2)*100 #3%
#dump the predicted carbons
np.savetxt('y_val_act.csv',y_test,delimiter=',')
np.savetxt('y_val_pred.csv',reg.predict(X_test),delimiter=',')


#retrain using the whole data and save the model
reg.fit(X,Y)
joblib.dump(reg,'glob_model.mdl')








