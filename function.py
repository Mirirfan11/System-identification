####################################################################
# Function to determine goodness of fit
####################################################################
def fit(targets, predictions):
    from numpy import linalg as LA
    targets=targets.reshape(targets.shape[0],1)
    predictions = predictions.reshape(predictions.shape[0],1) 
    return (1-LA.norm(targets - predictions)/LA.norm
                        ....(targets-targets.mean()))*100
####################################################################
# Function to simulate and plot the results
####################################################################
def simulate(X_test,Y_test):
    predict_time=Y_test.shape[0]
    Y_predicted_offline=np.zeros(shape=(predict_time,1)) 
    X_predict_offline = X_test[0].reshape(1,pastoutput+pastinput);
    Y_predicted_offline[0] = Y_test[0]
    y_predict_tmp = Y_predicted_offline[0] 
    Y_past =X_test[0,pastinput:pastinput+pastoutput].reshape(1,pastoutput) 
    for i in range(0,predict_time-1):
        Y_past[:,0:pastoutput-1]=Y_past[:,1:] #Rolling window by 1
        Y_past[:,-1]=y_predict_tmp
        X_predict_offline[:,0:pastinput]=X_test[i+1,0:pastinput]
        X_predict_offline[:,pastinput:pastinput+pastoutput]=Y_past    
        y_predict_tmp= model.predict(X_predict_offline)
        Y_predicted_offline[i+1]=y_predict_tmp
    Y_test = Y_test.reshape(Y_test.shape[0],1)
    import matplotlib.pyplot as plt
    import math
    Fit = fit(Y_test[:predict_time],Y_predicted_offline[:predict_time])
    if Fit == -math.inf:
        return print("Bad Fit -Inf")
    else:
        plt.plot(Y_test[:predict_time],label='Real output')
        plt.plot(Y_predicted_offline[:predict_time], label='NARX_NN')
        plt.xlabel('Discrete time steps')
        plt.ylabel('Force')
        plt.title("Fit :" +str(f'{Fit:.2f}')+"%" )
        plt.legend() 
        plt.grid()   
    return plt.show()
####################################################################
# Function to Plot the given data
####################################################################
def plots(t,x,y):
    plt.figure(figsize=(14,6))
    plt.subplot(2,1,1)
    plt.plot(t,x,label=r'$Input Force$')
    plt.legend();plt.grid()
    plt.ylabel('Excitation')
    plt.subplot(2,1,2)
    plt.plot(t,y,label=r'$Displacement$')
    plt.legend()
    plt.ylabel('Displacement')
    plt.xlabel('Time')
    plt.grid()
    return plt.show()
