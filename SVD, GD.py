import numpy as np
import matplotlib.pyplot as plt

# %% Normal Equation method
np.random.seed(42)  # to reproduce every time the same results.
X = 3 * np.random.rand(200, 1)
y = 3 + 5 * X + np.random.randn(200, 1)
# %%
plt.plot(X, y, 'b.')
plt.xlabel('$X_1$', fontsize=18)
plt.axis([0, 2, 0, 15])
plt.show()
# %%
X_b = np.c_[np.ones((200, 1)), X]
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
theta_best
# %% Evaluate theta_0 and theta_1 with LinearRegression model
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X, y)
lin_reg.intercept_, lin_reg.coef_

# It exactly ame as theta which evaluated with normal equation method
# what does happen behind the linear regression method ?
# %% SVD ( Singular Value Decomposition)
theta_best_svd, residuals, rank, s = np.linalg.lstsq(X_b, y, rcond=1e-6)
s
residuals
rank
theta_best_svd
# again it is same as latter result, it uses pseudoinverse method
# theta = X^+ y
# %%
np.linalg.pinv(X_b).dot(y)
# %% 2nd model of calculation, Gradiant Descend (GD)
# Batch Gradiant Descend
# This method, uses all instances in out dataset. and causes that the model takes more time than the others.
# We should start with arbitrary theta, specify iteration numbers and the learning step (eta).
eta = 0.2
n_iterations = 1000
m = 300
theta = np.random.rand(2, 1)  # initial arbitrary theta_0 and theta_1
for iteration in range(n_iterations):
    gradiant = 2 / m * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - eta * gradiant

theta
# %%
X_new = np.array([[0], [3]])
X_b_new = np.c_[np.ones((2, 1)), X_new]
y_best = X_b_new.dot(theta)
plt.plot(X_new, y_best, 'r-')
plt.plot(X, y, 'b.')
plt.axis([0, 2, 2, 15])
plt.show()
# %% Visualizing effect of learning rate hyperparameter (step size)
# %% Too large steps
theta = np.random.rand(2, 1)
eta = 0.4
n_iterations = 10
m = 300
X_new = np.array([[0], [3]])
X_b_new = np.c_[np.ones((2, 1)), X_new]
for iteration in range(n_iterations):
    gradiant = 2 / m * X_b.T.dot(X_b.dot(theta) - y)
    y_predict = X_b_new.dot(theta)
    plt.plot(X_new, y_predict, 'b-')
    theta = theta - eta * gradiant
plt.axis([0, 3, 0, 15])
plt.plot(X, y, 'r.')
plt.show()
# %% best step rate
theta = np.random.rand(2, 1)
eta = 0.2
n_iterations = 10
m = 300
X_new = np.array([[0], [3]])
X_b_new = np.c_[np.ones((2, 1)), X_new]
for iteration in range(n_iterations):
    gradiant = 2 / m * X_b.T.dot(X_b.dot(theta) - y)
    y_predict = X_b_new.dot(theta)
    plt.plot(X_new, y_predict, 'b-')
    theta = theta - eta * gradiant
# plt.axis([0, 3, 0, 15])
plt.plot(X, y, 'r.')
plt.show()
# %% Very tiny step
theta = np.random.rand(2, 1)
eta = 0.02
n_iterations = 10
m = 300
X_new = np.array([[0], [3]])
X_b_new = np.c_[np.ones((2, 1)), X_new]
for iteration in range(n_iterations):
    gradiant = 2 / m * X_b.T.dot(X_b.dot(theta) - y)
    y_predict = X_b_new.dot(theta)
    plt.plot(X_new, y_predict, 'b-')
    theta = theta - eta * gradiant
plt.axis([0, 3, 0, 15])
plt.plot(X, y, 'r.')
plt.show()

# %% Stochastic Gradiant Descend (SGD)
# This DG method chooses an instance randomly instead of all instances. and computes it in number of epoch.
m = len(X_b)
np.random.seed(42)
# %%
n_epoch = 50
t0, t1 = 5, 50


def learning_schedule(t):
    return t0 / (t + t1)


theta = np.random.rand(2, 1)

for epoch in range(n_epoch):
    for i in range(m):
        random_index = np.random.randint(m)
        xi = X_b[random_index:random_index + 1]
        yi = y[random_index:random_index + 1]
        gradiant = 2 * xi.T.dot(xi.dot(theta) - yi)
        eta = learning_schedule(n_epoch * m + i)
        theta = theta - eta * gradiant
theta
#%% plotting
m = 25
np.random.seed(42)
n_epoch = 1
t0, t1 = 5, 15


def learning_schedule(t):
    return t0 / (t + t1)


theta = np.random.rand(2, 1)
X_new = np.array([[0], [3]])
X_b_new = np.c_[np.ones((2, 1)), X_new]

for epoch in range(n_epoch):
    for i in range(m):
        random_index = np.random.randint(m)
        xi = X_b[random_index:random_index + 1]
        yi = y[random_index:random_index + 1]
        gradiant = 2 * xi.T.dot(xi.dot(theta) - yi)
        y_predict = X_b_new.dot(theta)
        plt.plot(y_predict, X_new, 'r-')
        eta = learning_schedule(n_epoch * m + i)
        theta = theta - eta * gradiant

plt.plot(y, X, 'b.')
plt.axis([0, 15, 0, 2])
plt.show()

#%% INSTEAD OF THESE, WE CAN USE SGD in sklearn
from sklearn.linear_model import SGDRegressor

sgd_reg = SGDRegressor(max_iter=1000, tol=1e-3, penalty=None, eta0=0.1)
sgd_reg.fit(X, y.ravel())
#%%
sgd_reg.intercept_, sgd_reg.coef_
