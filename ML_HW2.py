#!/usr/bin/env python
# coding: utf-8

# In[135]:


import cvxpy as cp
import numpy as np


# In[198]:


## Load and split data
print('Q4:\nLoad and split data...')

data = np.genfromtxt('winequality-red.csv', delimiter = ';', skip_header=True)

training_X = data[:1400,:11]
training_Y = data[:1400,11]

testing_X = data[1400:,:11]
testing_Y = data[1400:,11]
m,n = training_X.shape

## Least Squares Loss
print('- Least Squares Loss:')

w = cp.Variable(n)
beta = cp.Variable(1)
cost = cp.sum_squares(training_X*w + beta - training_Y)
prob = cp.Problem(cp.Minimize(cost))
prob.solve()

b = beta.value.item()
prediction = [sum(i)+ b for i in training_X*w.value]
mae = sum([abs(j-i) for i,j in zip(prediction,training_Y)])/m
print('  Training MAE is ', mae)

prediction = [sum(i) + b for i in testing_X*w.value]
mae = sum([abs(j-i) for i,j in zip(prediction,testing_Y)])/len(testing_Y)
print('  Testing MAE is ', mae)

## Huber Loss
print('- Huber Loss:')

w = cp.Variable(n)
beta = cp.Variable(1)
cost = cp.sum(cp.huber(training_X*w + beta - training_Y, 1))
prob = cp.Problem(cp.Minimize(cost))
prob.solve()

b = beta.value.item()
prediction = [sum(i) + b for i in training_X*w.value]
mae = sum([abs(j-i) for i,j in zip(prediction,training_Y)])/m
print('  Training MAE is ', mae)

prediction = [sum(i)+b for i in testing_X*w.value]
mae = sum([abs(j-i) for i,j in zip(prediction,testing_Y)])/len(testing_Y)
print('  Testing MAE is ', mae)

## Hinge Loss
print('- Hinge Loss:')

def hinge(t):
    return cp.pos(cp.abs(t)-0.5)

w = cp.Variable(n)
beta = cp.Variable(1)
cost = cp.sum(hinge(training_X*w + beta - training_Y))
prob = cp.Problem(cp.Minimize(cost))
prob.solve(solver=cp.ECOS)

b = beta.value.item()
prediction = [sum(i) + b for i in training_X*w.value]
mae = sum([abs(j-i) for i,j in zip(prediction,training_Y)])/m
print('  Training MAE is ', mae)

prediction = [sum(i) + b for i in testing_X*w.value]
mae = sum([abs(j-i) for i,j in zip(prediction,testing_Y)])/len(testing_Y)
print('  Testing MAE is ', mae)


# In[199]:


## Load and split data
print('Q5:\nLoad and split data...')

data = np.genfromtxt('ionosphere.data', delimiter = ',')[:,:34]
labels = np.genfromtxt('ionosphere.data', delimiter = ',', dtype = 'str')[:,34]
labels = np.asarray([1 if i == 'g' else -1 for i in labels])

training_X = data[:300]
training_Y = labels[:300]

testing_X = data[300:]
testing_Y = labels[300:]
m,n = training_X.shape

## Least Squares Loss
print('- Least Squares Loss')

w = cp.Variable(n)
beta = cp.Variable(1)
cost = cp.sum_squares(training_X*w + beta - training_Y)
prob = cp.Problem(cp.Minimize(cost))
prob.solve()

b = beta.value.item()
prediction = [1 if sum(i)+ b > 0 else -1 for i in training_X*w.value]
accuracy = sum([1 if i == j else 0 for i,j in zip(prediction,training_Y)])/m
print('  Training accuracy is ', accuracy)

prediction = [1 if sum(i)+ b > 0 else -1 for i in testing_X*w.value]
accuracy = sum([1 if i == j else 0 for i,j in zip(prediction,testing_Y)])/len(testing_Y)
print('  Testing accuracy is ', accuracy)


## Logistic Loss
print('- Logistic Loss:')

w = cp.Variable(n)
beta = cp.Variable(1)
cost = cp.sum(cp.logistic(-cp.multiply(training_X*w + beta, training_Y)))
prob = cp.Problem(cp.Minimize(cost))
prob.solve()

b = beta.value.item()
prediction = [1 if sum(i)+ b > 0 else -1 for i in training_X*w.value]
accuracy = sum([1 if i == j else 0 for i,j in zip(prediction,training_Y)])/m
print('  Training accuracy is ', accuracy)

prediction = [1 if sum(i)+ b > 0 else -1 for i in testing_X*w.value]
accuracy = sum([1 if i == j else 0 for i,j in zip(prediction,testing_Y)])/len(testing_Y)
print('  Testing accuracy is ', accuracy)

## Hinge Loss
print('- Hinge Loss:')

w = cp.Variable(n)
beta = cp.Variable(1)
cost = cp.sum(cp.pos(1-cp.multiply(training_X*w + beta, training_Y)))
prob = cp.Problem(cp.Minimize(cost))
prob.solve(solver = cp.ECOS)

b = beta.value.item()
prediction = [1 if sum(i)+ b > 0 else -1 for i in training_X*w.value]
accuracy = sum([1 if i == j else 0 for i,j in zip(prediction,training_Y)])/m
print('  Training accuracy is ', accuracy)

prediction = [1 if sum(i)+ b > 0 else -1 for i in testing_X*w.value]
accuracy = sum([1 if i == j else 0 for i,j in zip(prediction,testing_Y)])/len(testing_Y)
print('  Testing accuracy is ', accuracy)


# In[ ]:




