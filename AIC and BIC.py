# %%
import numpy as np
import matplotlib.pyplot as plt

trng = np.loadtxt("C:\\Users\\wajpe\\OneDrive\\Desktop\\Lectures\\AESRP 597\\HW SET\\HW1\\flight_data_train.csv", delimiter=',',
                encoding='utf8')
test = np.loadtxt("C:\\Users\\wajpe\\OneDrive\\Desktop\\Lectures\\AESRP 597\\HW SET\\HW1\\flight_data_test.csv", delimiter=',',
                encoding='utf8')
dataset = np.concatenate((trng,test),axis=0)
t_train = dataset[:,6]
N = len(dataset)

phi_train = np.zeros([len(dataset),36])

for i in range(1,7):
  phi_train[:,6*(i-1):(6*i)]  = dataset[:,0:6]**i
phi_train_max = np.max(phi_train, axis=0, keepdims=True)
phi_train_min = np.min(phi_train, axis=0, keepdims=True)
phi_train = (phi_train - phi_train_min)/(phi_train_max - phi_train_min)
ones_array = np.ones([len(phi_train),1])
phi_train = np.c_[ones_array,phi_train]

def range_of_lambda(l):
    l = np.exp(l)
    w_train = np.linalg.inv(phi_train.T @ phi_train + l * np.eye(37)) @ phi_train.T @ t_train
    gammas = np.trace(np.linalg.inv(phi_train.T @ phi_train + l * np.eye(37)) @ phi_train.T @ phi_train)
    Ed = 0.5*np.sum((np.dot(phi_train,w_train)-t_train)**2)
    AIC = N*np.log(Ed/N) + gammas
    BIC = N*np.log(Ed/N) + gammas*np.log(N)
    return AIC, BIC


AIC = np.zeros(41)
BIC = np.zeros(41)
lambdas = list(range(-30, 11))
for i, l in enumerate(lambdas):
    AIC[i], BIC[i] = range_of_lambda(l)

plt.plot(list(range(-30,11)),AIC,marker='.',markersize=7.5)
plt.plot(list(range(-30,11)),BIC,marker='.',markersize=7.5)
plt.xlabel('$ln\lambda$')
plt.ylabel('Values of AIC and BIC over N')
plt.legend(['AIC','BIC'])

