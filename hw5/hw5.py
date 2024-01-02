import sys
import matplotlib.pyplot as plt
import csv
import numpy as np

def load_data():
    years = []
    days = []
    with open(sys.argv[1], 'r') as csvfile:
        next(csvfile)
        lines = csv.reader(csvfile, delimiter=',')
        for row in lines:
            years.append(int(row[0]))
            days.append(int(row[1]))
    return years, days

def Q2(years, days):
    plt.plot(years, days)
    plt.xlabel('Year')
    plt.ylabel('Number of frozen days')
    plt.savefig("plot.jpg")
    return years, days

def Q3(x_list, y_list):
    n_x = len(x_list)
    X = np.zeros((n_x, 2))
    Y = np.array(y_list)
    for i in range(n_x):
        X[i] = np.array([1, x_list[i]])
    X = X.astype(int)
    print('Q3a:')
    print(X)
    print('Q3b:')
    print(Y)

    Z = np.dot(np.transpose(X), X)
    print("Q3c:")
    print(Z) 

    I = np.linalg.inv(Z)
    print('Q3d:')
    print(I)

    PI = np.dot(I, np.transpose(X))
    print('Q3e:')
    print(PI)
    

    hat_beta = np.dot(PI, Y)

    print('Q3f')
    print(hat_beta)
    return hat_beta

def main():
    x, y = load_data()
    Q2(x, y)
    b_arr = Q3(x, y)
    y_test = b_arr[0] + b_arr[1] * 2022
    print("Q4: " + str(y_test))
    if(b_arr[1] < 0):
        print('Q5a: <')
    elif(b_arr[1] > 0):
        print('Q5a: >')
    else:
        print('Q5a: =')
    print('Q5b: The sign means the number of frozen days of Lake Mendota is decreasing every year.')

    x_star = (-1) * b_arr[0] / b_arr[1]
    print('Q6a: ' + str(x_star))
    print('Q6b: This data makes sense, beacause the frozen days are decreasing year by year')




if __name__ == "__main__":
    main()