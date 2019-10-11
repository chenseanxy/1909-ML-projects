#%%
import pandas as pd
from math import exp, log

#%% Reading and processing dataset
dataset_path = "watermelon3.0alpha.csv"
raw_dataframe = pd.read_csv(dataset_path)
x1 = raw_dataframe["density"]
x2 = raw_dataframe["Sugar_content"]
y = raw_dataframe["label"]
m = len(x1)

#%% Defining calc_() functions
def calc_z(w1, w2, b, x1, x2):
    return w1*x1 + w2*x2 + b

def calc_a(z): #Logistic
    return 1/(1+exp(-z))

def calc_loss(yhat, y):
    return -(y*log(yhat) + (1-y)*log(1-yhat))

def calc_dw(x, y, yhat):
    return x * (yhat - y)

def calc_dws(x, y, yhat):
    return x*yhat*(1-yhat)

def calc_db(yhat, y):
    return yhat-y

def calc_dbs(yhat, y):
    return yhat*(1-yhat)

#%% Training with Newton
w1, w2, b = 0, 0, 0
last_cost = m   # Max cost

min_step = 1E-10
step_counter = 0

while True:
    dw1, dw1s, dw2, dw2s, db, dbs = 0, 0, 0, 0, 0, 0
    cost = 0
    
    for i in range(m):
        z = calc_z(w1, w2, b, x1[i], x2[i])
        yhat = calc_a(z)
        loss = calc_loss(yhat, y[i])
        cost += loss

        dw1 += calc_dw(x1[i], y[i], yhat)
        dw1s += calc_dws(x1[i], y[i], yhat)
        dw2 += calc_dw(x2[i], y[i], yhat)
        dw2s += calc_dws(x2[i], y[i], yhat)
        db += calc_db(yhat, y[i])
        dbs += calc_dbs(yhat, y[i])
    
    w1 = w1 - dw1 / dw1s
    w2 = w2 - dw2 / dw2s
    b = b - db / dbs
    step_counter += 1

    print(f"{step_counter} - cost={cost}, r={last_cost-cost}")

    
    if (last_cost - cost) < min_step:
        break
    last_cost = cost

print("---------------------------------------------")
print("Training complete using Newton")
print(f"Iterations: {step_counter}")
print(f"Cost: {cost}")
print(f"Params: w1={w1}, w2={w2}, b={b}")

#%% Training with Gradient Decent
w1, w2, b = 0, 0, 0
last_cost = m   # Max cost

min_step = 1E-10
step_counter = 0
learning_rate = 0.1

while True:
    dw1, dw1s, dw2, dw2s, db, dbs = 0, 0, 0, 0, 0, 0
    cost = 0
    
    for i in range(m):
        z = calc_z(w1, w2, b, x1[i], x2[i])
        yhat = calc_a(z)
        loss = calc_loss(yhat, y[i])
        cost += loss

        dw1 += calc_dw(x1[i], y[i], yhat)
        dw2 += calc_dw(x2[i], y[i], yhat)
        db += calc_db(yhat, y[i])
    
    w1 = w1 - dw1 * learning_rate
    w2 = w2 - dw2 * learning_rate
    b = b - db * learning_rate
    step_counter += 1

    print(f"{step_counter} - cost={cost}, r={last_cost-cost}")

    if (last_cost - cost) < min_step:
        break
    last_cost = cost

print("---------------------------------------------")
print("Training complete using Gradient Decent")
print(f"Iterations: {step_counter}")
print(f"Cost: {cost}")
print(f"Params: w1={w1}, w2={w2}, b={b}")

#%% Prediction wrapper
def predict(x1, x2):
    global w1, w2, b    #Use the last trained params
    return calc_a(calc_z(w1, w2, b, x1, x2))

#%% Sanity check: does the model perform properly on trained sets?
gr_err = 0

for i in range(m):
    prediction = predict(x1[i], x2[i])
    gr_err += abs(prediction - y[i])
    print(f"{i} - predicted {prediction}, label {y[i]}")

print(f"Avg err: {gr_err/m}")
