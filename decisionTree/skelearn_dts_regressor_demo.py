#-*- coding:utf-8 -*-
#########################################################################
# File Name: skelearn_dts_regressor_demo.py
# Author: Nichol.Shen
# mail: nichol_shen@yahoo.com
#########################################################################
#!/usr/bin/python3

import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

rng = np.random.RandomState(1)


X = np.sort(5 * rng.rand(80, 1), axis=0)

y = np.sin(X).ravel()

y[::5] += 3 * (0.5 - rng.rand(16))


regr_2 = DecisionTreeRegressor(max_depth=5)
regr_2 = DecisionTreeRegressor(min_samples_leaf=6)

regr_2.fit(X, y)


#predict
X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
y_2 = regr_2.predict(X_test)


#Draw pylot
plt.figure()
plt.scatter(X, y, c='darkorange', label='data')
plt.plot(X_test, y_2, color = 'yellowgreen', label = 'max_depth=5', linewidth = 2)

plt.xlabel('data')
plt.ylabel('target')
plt.title('Decision Tree Regression')
plt.legend()
plt.show()
