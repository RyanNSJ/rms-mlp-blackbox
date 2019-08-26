import blackbox as bb

import numpy as np

import pandas as pd

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split

# Generate 1000 Data Points

NUM_DATA = 100000
NUM_PRODUCTS = 5


prices_dict = {}
prices_mean = [2,3,10,13,25]
prices_stdev = [1,1,3,2,4]


# Prices
for i in range(NUM_PRODUCTS):
    rng_mean = prices_mean[i]
    rng_stdev = prices_stdev[i]
    prices_dict[i] = np.random.normal(loc=rng_mean, scale=rng_stdev, size=NUM_DATA)
    
prices = np.asarray([prices_dict[i] for i in range(NUM_PRODUCTS)], dtype=np.float32)
prices = prices.transpose()

price_elasticities = [[25,5,15,15,15],
                      [5,25,15,15,15],
                      [15,15,15,5,15],
                      [15,15,5,15,15],
                      [15,15,15,15,10],
                     ]

base_demand = [100,120,40,50,20]

demands = {}
for k in range(NUM_DATA):
    demands[k] = np.ndarray(NUM_PRODUCTS)
    demands_list = []
    for i in range(NUM_PRODUCTS):
        demand_i = base_demand[i]
        price_data = prices[k,:]
        for j in range(NUM_PRODUCTS):
            drop_in_demand = price_elasticities[i][j]*(price_data[j]-prices_mean[j])
            demand_i -= drop_in_demand
        noise = np.random.normal(loc=0, scale=base_demand[i]/20, size=1)
        demand_i = demand_i + noise
        demands_list.append(demand_i)
    demands[k] = np.asarray(demands_list)
    
demands_array = np.asarray([demands[k] for k in range(NUM_DATA)])
demands_array = demands_array[:,:,0]

all_data = np.hstack([prices, demands_array])

X = all_data[:,:5]
Y = all_data[:,5:]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

from sklearn.neural_network import MLPRegressor

reg = MLPRegressor(hidden_layer_sizes=(16,16,16), max_iter=10000)
reg.fit(X=X_train, y=Y_train)
mlp_pred_yss = reg.predict(X=X_test)

print("mlp rmse:",np.sqrt(mean_squared_error(y_pred=mlp_pred_yss,y_true=Y_test)))

def predicted_revenue(par):
    predicted_demand = reg.predict(np.asarray(par).reshape(1,-1))[0]
    predicted_price = np.asarray(par).reshape(1,-1)
    print(predicted_demand)
    print(par)
    revenue = np.dot(predicted_price, predicted_demand)
    return revenue[0]

def main():
    bb.search(f=predicted_revenue,  # given function
              box=[[0., 10.], [0., 12.],[0.,40.],[0.,50.],[0.,100.]],  # range of values for each parameter (2D case)
              n=100,  # number of function calls on initial stage (global search)
              m=100,  # number of function calls on subsequent stage (local search)
              batch=16,  # number of calls that will be evaluated in parallel
              resfile='revenue_output.csv')  # text file where results will be saved


if __name__ == '__main__':
    main()