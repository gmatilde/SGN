import json
import matplotlib.pyplot as plt


with open('sinusoidal_data_100.0_0.01.json', 'r') as f:
    data_1 = json.load(f)

f.close()

with open('../sinusoidal_data_100.0_0.01.json', 'r') as f:
    data_2 = json.load(f)
f.close()
import pdb; pdb.set_trace()
plt.figure()
plt.scatter(data_1['X_train'], data_1['y_train'])
plt.scatter(data_2['X_train'], data_2['y_train'])
plt.show()

