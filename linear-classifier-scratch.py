## TUGAS MACHINE LEARNING
## KURNIAWAN EKA NUGRAHA
## 15/383239/PA/16899


# Usage -> python linear-classifier-scratch
# Dependencies : python3, seaborn

def preprocess(input):

    output_array = []
    for item in input:
        output_array.append(float(item))

    return output_array

def split_train_test(x1,x2,x3,x4,label):

    x1_val = x1[40:50] + x1[90:100]
    x2_val = x2[40:50] + x2[90:100]
    x3_val = x3[40:50] + x3[90:100]
    x4_val = x4[40:50] + x4[90:100]
    label_val = label[40:50] + label[50:90]
    x1 = x1[0:40] + x1[50:90]
    x2 = x2[0:40] + x2[50:90]
    x3 = x3[0:40] + x3[50:90]
    x4 = x4[0:40] + x4[50:90]
    label = label[0:40] + label[50:90]

    return x1,x2,x3,x4,label,x1_val,x2_val,x3_val,x4_val,label_val


def calculate_h(x1,x2,x3,x4,theta1,theta2,theta3,theta4,bias,i):

    h = x1[i]*theta1+x2[i]*theta2+x3[i]*theta3+x4[i]*theta4+bias

    return h

def calculate_activation(h):
    
    e = 2.71828
    activation = 1/(1+e**(-h))

    return activation

def calculate_pred(s):

    if s<0.5:
        pred = 0
    else:
        pred = 1

    return pred

def calculate_delta_theta(activation,pred,label,x1,x2,x3,x4,i):

    dbias = 2*(activation-label[i])*(1-activation)*activation
    dtheta1 = x1[i]*dbias
    dtheta2 = x2[i]*dbias
    dtheta3 = x3[i]*dbias
    dtheta4 = x4[i]*dbias

    return dtheta1,dtheta2,dtheta3,dtheta4,dbias

def update_theta(dtheta1,dtheta2,dtheta3,dtheta4,dbias,theta1,theta2,theta3,theta4,bias,alpha):

    theta1 = theta1-dtheta1*alpha
    theta2 = theta2-dtheta2*alpha
    theta3 = theta3-dtheta3*alpha
    theta4 = theta4-dtheta4*alpha
    bias = bias-dbias*alpha

    return theta1,theta2,theta3,theta4,bias

def calculate_error(label,activation,i,error_array):

    error = (label[i]-activation)**2
    error_array[i]=error

    return error
 
def average_error_per_epoch(average_error_array,error_array):

    avg_error = sum(error_array) / float(len(error_array))
    average_error_array.append(avg_error)

    return avg_error

def predict(x1,x2,x3,x4,theta1,theta2,theta3,theta4,bias):

    h = x1*theta1+x2*theta2+x3*theta3+x4*theta4+bias
    s = calculate_activation(h)

    if s<0.5:
        pred = 0
        notes = "Iris - Setosa"
    else:
        pred = 1
        notes = "Iris - Versicolor"
    
    return pred, notes, s

def validation(x1_val,x2_val,x3_val,x4_val,label_val,theta1,theta2,theta3,theta4,bias,validation_array):

    activation_array = []
    tmp = []

    for i in range(len(x1_val)):
        _, _, s = predict(x1_val[i],x2_val[i],x3_val[i],x4_val[i],theta1,theta2,theta3,theta4,bias)
        activation_array.append(s)

    for i in range(len(activation_array)):
        error = (label_val[i]-activation_array[i])**2
        tmp.append(error)

    average_error_per_epoch(validation_array,tmp)

    return 0
    


import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot(validation_array,average_error_array):

    fig = plt.figure()
    fig.suptitle('MSE Graph')
    sns.set_style("darkgrid")

    plt.plot(list(range(1,61)),validation_array, label = "MSE validasi")
    plt.plot(list(range(1,61)),average_error_array, label = "MSE training")

    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()

    return plt.show()

def main():

    print("==================LINEAR CLASSIFIER IRIS==================")

    # Data input
    with open("iris.txt", "r") as data:
        data = data.read().replace('\n', ',')
        x1 = preprocess(data.split(",")[0::5])
        x2 = preprocess(data.split(",")[1::5])
        x3 = preprocess(data.split(",")[2::5])
        x4 = preprocess(data.split(",")[3::5])
        label = preprocess(data.split(",")[4::5])



    # Split train and validation
    x1,x2,x3,x4,label,x1_val,x2_val,x3_val,x4_val,label_val = split_train_test(x1,x2,x3,x4,label)

    # Global var declaration
    theta1 = 0.1
    theta2 = 0.1
    theta3 = 0.1
    theta4 = 0.1
    bias = 0.1
    alpha = float(input("Input learning rate: "))
    training_epoch = 60
    error_array_per_epoch=[None]*80 #train size
    average_error_array=[]
    validation_array=[]

    # For Convergence
    avg_error = 0.0
    prev = 0.0
    tmp = True

    print("=========Training Result with learning rate ",alpha," =========")
    # Training
    start = time.time()
    for x in range(training_epoch):

        # Do for Each Epoch
        for i in range(len(label)):
            h = calculate_h(x1,x2,x3,x4,theta1,theta2,theta3,theta4,bias,i)
            activation = calculate_activation(h)
            pred = calculate_pred(activation)
            error = calculate_error(label,activation,i,error_array_per_epoch)
            dtheta1,dtheta2,dtheta3,dtheta4,dbias = calculate_delta_theta(activation,pred,label,x1,x2,x3,x4,i)
            theta1,theta2,theta3,theta4,bias = update_theta(dtheta1,dtheta2,dtheta3,dtheta4,dbias,theta1,theta2,theta3,theta4,bias,alpha)

        
        # Calculate error for each epoch
        prev = avg_error
        avg_error = average_error_per_epoch(average_error_array,error_array_per_epoch)
        validation(x1_val,x2_val,x3_val,x4_val,label_val,theta1,theta2,theta3,theta4,bias,validation_array)

        # Check convergence, average error change very small (0.0...1)
        if abs(avg_error-prev)<0.0001 and (tmp==True and x>0):
            end = time.time()
            tmp = False
            print("Epoch needed to convergence :",x+1,"\nTime needed to convergence :",f'{end-start:.20f}')
    

    # Notes
    end_total = time.time()
    print("Total Running time to train 60 epoch :",f'{end_total-start:.20f}')
    print("theta 1 = ",theta1,"\ntheta 2 = ",theta2,"\ntheta 3 = ",theta3,"\ntheta 4 = ",theta4,"\nbias = ",bias)
    
    # Plot graph
    plot(validation_array,average_error_array)

    # Predict from values
    var = input("input `x1 x2 x3 x4` : ")
    var = list(map(float, var.split(' ')))
    pred, notes, _ = predict(var[0],var[1],var[2],var[3],theta1,theta2,theta3,theta4,bias)
    print("Prediction : ",notes," [",pred,"]")
    
main()
