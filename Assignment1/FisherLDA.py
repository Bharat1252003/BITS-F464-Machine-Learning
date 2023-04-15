import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

seeds = [np.random.randint(0,2**32-1, dtype=np.int64) for _ in range(10)]
print(seeds)
print("")

def get_data(filename):
    # read data from csv file
    data = pd.read_csv(filename).to_numpy()

    # replace character for class with an integer 0 or 1
    for i in range(data.shape[0]):
        if data[i,1] == 'M':                # M == 1
            data[i,1] = 1.0
        else:
            data[i,1] = 0.0

    features = data

    # test train split
    np.random.shuffle(features)
    l = len(features)
    features_train = features[:int(0.67*l),:]
    features_test = features[int(0.67*l):,:]

    return features_train, features_test

def feature_eng(features):
    # removes nans
    col_mean = np.nanmean(features, axis = 0)
    inds = np.where(pd.isna(features))
    features[inds] = np.take(col_mean, inds[1])

    # normalize data
    means = np.mean(features, axis=0)
    vars = features.var(axis=0, dtype=np.float32)
    features_normed = (features-means) / np.sqrt(vars)

    # split the 2 classes
    features1 = features_normed[features[:,1]==1.0]
    features2 = features_normed[features[:,1]==0.0]

    return features1, features2, features_normed, (means, vars)

def lda(features1, features2, features_normed):
    # projection to 1D
    pca_all = PCA(n_components=1)
    all_1 = pca_all.fit(features_normed)
    s_b = pca_all.get_covariance()

    pca_class1 = PCA(n_components=1)
    class1_1 = pca_class1.fit(features1)
    s_w_1 = pca_class1.get_covariance()

    pca_class2 = PCA(n_components=1)
    class2_1 = pca_class2.fit(features2)
    s_w_2 = pca_class2.get_covariance()

    s_w = s_w_1 + s_w_2

    # finding the boundary
    T = np.matmul(np.linalg.inv(s_w), s_b)
    eigenValues, eigenVectors = np.linalg.eig(T)

    idx = np.argmax(eigenValues)

    # means and variances of each class
    c1_mean = np.real(np.mean(features1.dot(eigenVectors[idx])))
    c1_var = np.real(np.var(features1.dot(eigenVectors[idx])))

    c2_mean = np.real(np.mean(features2.dot(eigenVectors[idx])))
    c2_var = np.real(np.var(features2.dot(eigenVectors[idx])))

    #print(f"Class1 Mean: {c1_mean} Class1 Variance: {c1_var}")
    #print(f"Class2 Mean: {c2_mean} Class2 Variance: {c2_var}")

    return (c1_mean, c1_var), (c2_mean, c2_var), eigenVectors[idx] # weight vector for the boundary

def normalize_test_data(features, params):
    # removing nans
    col_mean = np.nanmean(features, axis = 0)
    inds = np.where(pd.isna(features))
    features[inds] = np.take(col_mean, inds[1])

    # normalize data using same parameters as the training data
    (means, vars) = params 
    features_normed = (features-means) / np.sqrt(vars)

    return features_normed

def main_1(seed):
    np.random.seed(seed)
    #648731 - best result in the few seeds that were explored
    features_train, features_test = get_data(r"C:\Bharat\College\Sem6\Machine_Learning\Dsata Set for Assignment 1.csv")
    features1, features2, features_normed, (means, vars) = feature_eng(features_train)
    features_test = normalize_test_data(features_test, (means, vars))

    (c1_mean, c1_var), (c2_mean, c2_var), w = lda(features1, features2, features_normed)

    x = np.arange(-10,10,.001)
    f1 = np.exp(-np.square(x-c1_mean)/2*c1_var)/(np.sqrt(2*np.pi*c1_var))
    f2 = np.exp(-np.square(x-c2_mean)/2*c1_var)/(np.sqrt(2*np.pi*c2_var))

    plt.plot(x,f1)
    plt.plot(x,f2)
    #plt.show()

    true_class = (features_test[:,1]>0).astype(int)
    p1 = sum(true_class)/len(true_class)
    p2 = 1-p1
    x_test = np.dot(features_test,w).reshape(-1,1)

    for i in range(len(x_test)):
        x_test[i] = np.array(np.real(x_test[i][0]))

    x_test = np.array(x_test, dtype=np.float64)

    f1 = np.exp(-np.square(x_test-c1_mean)/2*c1_var)/(np.sqrt(2*np.pi*c1_var))
    f2 = np.exp(-np.square(x_test-c2_mean)/2*c1_var)/(np.sqrt(2*np.pi*c2_var))

    output = (p1*f1>p2*f2).astype(int)
    acc= sum(output.ravel()==true_class.ravel())/len(output)
    
    return acc

# 10 test train splits
lst_1 = []
for i in range(len(seeds)):
    acc = main_1(seeds[i])
    print("Accuracy for seed {:d}: {:.3f}".format(i, acc))
    lst_1.append(acc)

print("Task1 Average Accuracy: {:.3f} Variance in accuracy: {:.5f}".format(np.average(np.array(lst_1)), np.var(np.array(lst_1))))

print("")

def permutate_cols(features):
    save_col = features[:,1]
    features = np.delete(features, 1, axis=1)
    features = features[:, np.random.permutation(features.shape[1])]
    features = np.insert(features, 1, save_col, axis=1)
    return features

def main_2(seed):
    np.random.seed(seed)
    
    features_train, features_test = get_data(r"C:\Bharat\College\Sem6\Machine_Learning\Dsata Set for Assignment 1.csv")
    
    features_train = permutate_cols(features_train)
    features_test = permutate_cols(features_test)

    features1, features2, features_normed, (means, vars) = feature_eng(features_train)
    features_test = normalize_test_data(features_test, (means, vars))

    (c1_mean, c1_var), (c2_mean, c2_var), w = lda(features1, features2, features_normed)


    x = np.arange(-10,10,.001)
    f1 = np.exp(-np.square(x-c1_mean)/2*c1_var)/(np.sqrt(2*np.pi*c1_var))
    f2 = np.exp(-np.square(x-c2_mean)/2*c1_var)/(np.sqrt(2*np.pi*c2_var))

    plt.plot(x,f1)
    plt.plot(x,f2)
    #plt.show()

    true_class = (features_test[:,1]>0).astype(int)
    p1 = sum(true_class)/len(true_class)
    p2 = 1-p1
    x_test = np.dot(features_test,w).reshape(-1,1)

    for i in range(len(x_test)):
        x_test[i] = np.array(np.real(x_test[i][0]))

    x_test = np.array(x_test, dtype=np.float64)

    f1 = np.exp(-np.square(x_test-c1_mean)/2*c1_var)/(np.sqrt(2*np.pi*c1_var))
    f2 = np.exp(-np.square(x_test-c2_mean)/2*c1_var)/(np.sqrt(2*np.pi*c2_var))

    output = (p1*f1>p2*f2).astype(int)
    acc= sum(output.ravel()==true_class.ravel())/len(output)
    
    return acc

# 10 test train splits
lst_2 = []
for i in range(len(seeds)):
    acc = main_2(seeds[i])
    print("Accuracy for seed {:d}: {:.3f}".format(i, acc))
    lst_2.append(acc)

print("Task2 Average Accuracy: {:.3f} Variance in accuracy: {:.5f}".format(np.average(np.array(lst_2)), np.var(np.array(lst_2))))