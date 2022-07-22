########################################################################
########### NARX-Neural Network for Spring Mass Damper System ##########
########################################################################
import numpy as np 
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout 
from tensorflow.keras.optimizers import RMSprop
from sklearn.metrics import, mean_absolute_error,r2_score
from matplotlib import pyplot as plt
plt.rcParams.update({'font.size': 18})

####################################################################
#           Load the Data
####################################################################
data = pd.read_csv('https://raw.githubusercontent.com/Mirirfan11/
                             System-identification/main/MSD.csv')
Time = data['Time'].values
########################### Training ##############################
Input1 = data['Input1'].values  # Excitation
output1 = data['output1'].values   # Displacement    
plots(Time,Input1,output1)
########################### Validation ############################
Input4 = data['Input4'].values  # Excitation
output4 = data['output4'].values   # Displacement
plots(Time,Input4,output4)
########################### Testing ###############################
Input3 = data['Input3'].values  # Excitation
output3 = data['output3'].values   # Displacement
plots(Time,Input3,output3)   

"""System Identification"""

##################################################################
#        This function formats the input and output data
##################################################################
def form_data(input_seq, output_seq,pastoutput,pastinput):
    data_len=np.max(input_seq.shape)
    X=np.zeros(shape=(data_len-pastinput,pastinput+pastoutput))
    Y=np.zeros(shape=(data_len-pastinput,))
    for i in range(0,data_len-(pastinput)):
        X[i,0:pastinput]=input_seq[i:i+pastinput,0]
        X[i,pastinput:]=output_seq[i:i+pastoutput,0]
        Y[i]=output_seq[i+pastoutput,0]
    return X,Y
#################### Direct Problem ###############################
pastinput=4;pastoutput=3;
###################################################################
#                  Create the training data
###################################################################
input_seq_train = Input1.reshape(Input1.shape[0],1)  
output_seq_train = output1.reshape(output1.shape[0],1)
X_train,Y_train = form_data(input_seq_train, output_seq_train,
                                         pastoutput,pastinput)
####################################################################
#                  Create the validation data
####################################################################
input_seq_validate = Input4.reshape(Input4.shape[0],1) 
output_seq_validate = output4.reshape(output4.shape[0],1)
X_validate,Y_validate =  form_data(input_seq_validate, 
                  output_seq_validate, pastoutput,pastinput)
#####################################################################
#                  Create the test data
#####################################################################
input_seq_test = Input1.reshape(Input1.shape[0],1)
output_seq_test = output1.reshape(output1.shape[0],1)
X_test,Y_test = form_data(input_seq_test, output_seq_test,
                                         pastoutput,pastinput)
####################################################################
#               Create the MLP network and train it
####################################################################
model = Sequential()
model.add(Dense(12,activation='linear',use_bias=False,
                            input_dim=(pastinput+pastoutput)))
model.add(Dense(8,activation='linear'))
model.add(Dense(1))
opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)
model.compile(optimizer=opt, loss='mse')
history=model.fit(X_train, Y_train, epochs=500, batch_size=20, 
          validation_data=(X_validate,Y_validate), verbose=2)
####################################################################
#                Plot training and validation curves
####################################################################
loss=history.history['loss']
val_loss=history.history['val_loss']
epochs=range(1,len(loss)+1)
plt.figure()
plt.plot(epochs, loss,'b', label='Training loss')
plt.plot(epochs, val_loss,'r', label='Validation loss')
plt.title('Training and validation losses')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.xscale('log')
plt.grid();plt.legend();
plt.show()
####################################################################
#   use the test data to investigate the prediction performance
####################################################################
network_prediction = model.predict(X_test)
Y_test = np.reshape(Y_test, (Y_test.shape[0],1)) 
plt.figure(figsize=(14,5))
plt.plot(Y_test, 'r', label='System')
plt.plot(network_prediction,'b', label='1 step ahead Prediction')
plt.xlabel('Discrete time steps')
plt.ylabel('Output');plt.legend();plt.grid();plt.show()
#################### Plot the Simulation ##########################
simulate(X_test,Y_test)
