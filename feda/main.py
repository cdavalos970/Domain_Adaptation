import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import random
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn import preprocessing
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

pd.set_option('display.max_columns', None)

PATH = '/Volumes/Transcend/Semester 3/Statistical Machine Learning/Assignments/Assignment 2/Data-Project2/'
ALGORITHMS = ['ALL']
MODELS = ['female', 'male', 'mixed']

PER_DEV = 0.3
PER_TEST = 0.333
SAMPLES = np.arange(0.01, 0.52, 0.02)
#SAMPLES = np.arange(50, 1050, 50)
SAMPLES_DEV = 100
FOLDS = 20
ITER = 10

enc = OneHotEncoder(handle_unknown='ignore')

def read_file(PATH):
    female = pd.read_csv(PATH + "FEMALE.csv")
    male = pd.read_csv(PATH + "MALE.csv")
    mixed = pd.read_csv(PATH + "MIXED.csv")

    female['type'] = 'female'
    male['type'] = 'male'
    mixed['type'] = 'mixed'

    data = pd.concat([female,male])
    data = pd.concat([data,mixed])
    return data


def encode_data(data):
    groups = data.type.unique()
    type = data[['type']]
    categorical_data = data[['Year', 'VR Band of Student', 'Ethnic group of student', 'School denomination']]
    categorical_data_encode = pd.get_dummies(categorical_data, columns=['Year', 'VR Band of Student', 'Ethnic group of student', 'School denomination'], drop_first=True)

    numerical_data = data[['FSM', 'VR1 Band']]
    numerical_data_scaled = preprocessing.scale(numerical_data)
    numerical_data_scaled = pd.DataFrame(numerical_data_scaled)
    numerical_data_scaled.columns = numerical_data.columns
    numerical_data_scaled.index = numerical_data.index

    base = categorical_data_encode.copy()
    base = pd.concat([base, type], axis=1)

    final = pd.DataFrame()
    for i in groups:
        base_group = base[base['type'] == i]
        base_group = base_group.drop(columns=['type'])
        general_group = base_group.copy().add_prefix('general_')
        for j in groups:
            if i !=  j:
                other = pd.DataFrame(np.zeros(base_group.shape))
                names = list(base_group.columns)
                var = str(j) + '_'
                names = [var + s for s in names]
                other.columns = names
                general_group = pd.concat([general_group, other], axis=1)
            else:
                target = base_group.copy()
                names = list(target.columns)
                var = str(j) + '_'
                names = [var + s for s in names]
                target.columns = names
                general_group = pd.concat([general_group, target], axis=1)
        final = final.append(general_group)

    type = data[['type']]
    response = data[['Exam Score']]
    final = pd.concat([final, numerical_data_scaled, response, type], axis=1)
    return final


def k_fold_plit(data, folds, size):
    fold = list()
    for i in range(0, folds):
        sample = data.sample(n = size)
        fold.append(sample)
    return fold


def partition_data(data, percentage_dev, percentage_test):
    X = data.drop(columns=['Exam Score'])
    y = data[['Exam Score']]
    X_train, X_other, y_train, y_other = train_test_split(X, y, test_size=percentage_dev)
    X_test, X_dev, y_test, y_dev = train_test_split(X_other, y_other, test_size=percentage_test)

    train = pd.concat([X_train, y_train], axis=1)
    test = pd.concat([X_test, y_test], axis=1)
    dev = pd.concat([X_dev, y_dev], axis=1)

    return train, dev, test


def cross_validation_set(replica_number, input, samples, percentage_dev = PER_DEV, percentage_test = PER_TEST, folds = FOLDS):
    female = input[input['type'] == 'female']
    male = input[input['type'] == 'male']
    mixed = input[input['type'] == 'mixed']

    if replica_number == 'female':
        train_female, dev_female, test_female = partition_data(female, percentage_dev, percentage_test)
        train_domain = pd.concat([male,mixed])

        sample_number = int(samples*train_domain.shape[0]/(1-samples))
        if sample_number <= train_female.shape[0]:
            train_target = train_female.sample(n = sample_number)
        else:
            remaining = sample_number - train_female.shape[0]
            remaining_target = train_female.sample(n = remaining, replace = True)
            train_target = pd.concat([train_female,remaining_target])

        dev = k_fold_plit(dev_female, folds, SAMPLES_DEV)
        test = test_female
    elif replica_number == 'male':
        train_male, dev_male, test_male = partition_data(male, percentage_dev, percentage_test)
        train_domain = pd.concat([female,mixed])

        sample_number = int(samples*train_domain.shape[0]/(1-samples))
        if sample_number <= train_male.shape[0]:
            train_target = train_male.sample(n = sample_number)
        else:
            remaining = sample_number - train_male.shape[0]
            remaining_target = train_male.sample(n = remaining, replace = True)
            train_target = pd.concat([train_male,remaining_target])

        dev = k_fold_plit(dev_male, folds, SAMPLES_DEV)
        test = test_male
    else:
        train_mixed, dev_mixed, test_mixed = partition_data(mixed, percentage_dev, percentage_test)
        train_domain = pd.concat([female,male])
        sample_number = int(samples*train_domain.shape[0]/(1-samples))
        if sample_number <= train_mixed.shape[0]:
            train_target = train_mixed.sample(n = sample_number)
        else:
            remaining = sample_number - train_mixed.shape[0]
            remaining_target = train_mixed.sample(n = remaining, replace = True)
            train_target = pd.concat([train_mixed,remaining_target])

        dev = k_fold_plit(dev_mixed, folds, SAMPLES_DEV)
        test = test_mixed

    print(train_target.shape[0]/(train_target.shape[0] + train_domain.shape[0]))
    return train_domain, train_target, dev, test


def all(train_domain, train_target, dev, test):
    summary_final = pd.DataFrame()
    all_data = pd.DataFrame()
    train = pd.concat([train_target,train_domain])

    mse, summary = linear_regression(train, dev, test)
    summary_final = summary_final.append(mse)
    all_data = all_data.append(summary)

    mse, summary = neural_network(train, dev, test)
    summary_final = summary_final.append(mse)
    all_data = all_data.append(summary)

    mse, summary = support_vector(train, dev, test)
    summary_final = summary_final.append(mse)
    all_data = all_data.append(summary)

    return all_data, summary_final

################################################################################
## Linear Regression
################################################################################

def linear_regression(train, dev, test, iter = ITER):
    summary = pd.DataFrame()
    X_train = train.drop(columns=['type','Exam Score'])
    y_train = train[['Exam Score']]
    X_test = test.drop(columns=['type','Exam Score'])
    y_test = test[['Exam Score']]
    for i in range(1,iter + 1):
        alpha = random.uniform(0.0001, 0.9)
        clf = linear_model.Lasso(random_state=0, alpha = alpha).fit(X_train, y_train)
        for j in range(0, len(dev)):
            dev_df = dev[j]
            X_dev = dev_df.drop(columns=['type','Exam Score'])
            y_dev = dev_df[['Exam Score']]
            y_pred_cv = clf.predict(X_dev).round(decimals=0)
            mse = mean_squared_error(y_dev, y_pred_cv)
            out = ('Linear Regression', alpha, mse, i, j)
            output = pd.DataFrame([out], columns=('model', 'alpha', 'mse', 'iter', 'fold'))
            summary = summary.append(output)
    summary_final = pd.DataFrame(summary.groupby(['model', 'alpha', 'iter']).mean().sort_values(by=['mse']).reset_index())
    best_alpha = summary_final['alpha'].iloc[0]
    clf = linear_model.Lasso(random_state=0, alpha = best_alpha).fit(X_train, y_train)
    y_pred = clf.predict(X_test).round(decimals=0)
    mse = mean_squared_error(y_test, y_pred)
    out = ('Linear Regression',{'alpha':round(best_alpha,3)}, mse)
    mse = pd.DataFrame([out], columns=('model','parameters','mse'))
    return mse, summary



################################################################################
## Neural Network
################################################################################

def neural_network(train, dev, test, iter = ITER):
    summary = pd.DataFrame()
    X_train = train.drop(columns=['type','Exam Score'])
    y_train = train[['Exam Score']]
    X_test = test.drop(columns=['type','Exam Score'])
    y_test = test[['Exam Score']]
    for i in range(1,iter + 1):
        hidden_layer_sizes = random.choice([(random.randint(1, 100), random.randint(1, 100), random.randint(1, 100),), (random.randint(1, 100), random.randint(1, 100),), (random.randint(1, 100),)])
        alpha = random.uniform(0.0001, 0.9)
        mlp = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, activation='relu', alpha = alpha, max_iter=2000)
        mlp.fit(X_train, y_train.values.ravel())
        for j in range(0, len(dev)):
            dev_df = dev[j]
            X_dev = dev_df.drop(columns=['type','Exam Score'])
            y_dev = dev_df[['Exam Score']]
            y_pred_cv = mlp.predict(X_dev).round(decimals=0)
            mse = mean_squared_error(y_dev, y_pred_cv)
            out = ('Neural Network', hidden_layer_sizes, alpha , mse, i, j)
            output = pd.DataFrame([out], columns=('model', 'hidden_layers', 'alpha', 'mse', 'iter', 'fold'))
            summary = summary.append(output)
    summary_final = pd.DataFrame(summary.groupby(['model', 'hidden_layers', 'alpha', 'iter']).mean().sort_values(by=['mse']).reset_index())
    best_hidden = summary_final['hidden_layers'].iloc[0]
    best_alpha = summary_final['alpha'].iloc[0]
    mlp = MLPRegressor(hidden_layer_sizes=best_hidden, activation='relu', alpha = best_alpha, max_iter=2000)
    mlp.fit(X_train, y_train.values.ravel())
    y_pred = mlp.predict(X_test).round(decimals=0)
    mse = mean_squared_error(y_test, y_pred)
    out = ('Neural Network', {'hidden_layer_sizes':best_hidden, 'alpha': round(best_alpha,3) }, mse)
    mse = pd.DataFrame([out], columns=('model','parameters','mse'))
    return mse, summary


################################################################################
## Support Vector Machine
################################################################################

def support_vector(train, dev, test, iter = ITER):
    summary = pd.DataFrame()
    X_train = train.drop(columns=['type','Exam Score'])
    y_train = train[['Exam Score']]
    X_test = test.drop(columns=['type','Exam Score'])
    y_test = test[['Exam Score']]
    for i in range(1,iter + 1):
        epsilon = random.uniform(0.0001, 0.9)
        C = random.uniform(0.001, 10)
        kernel = random.choice(['linear', 'rbf'])
        clf = SVR(C=C, epsilon=epsilon, kernel=kernel).fit(X_train, y_train.values.ravel())
        for j in range(0, len(dev)):
            dev_df = dev[j]
            X_dev = dev_df.drop(columns=['type','Exam Score'])
            y_dev = dev_df[['Exam Score']]
            y_pred_cv = clf.predict(X_dev).round(decimals=0)
            mse = mean_squared_error(y_dev, y_pred_cv)
            out = ('SVM', epsilon, C, kernel, mse, i, j)
            output = pd.DataFrame([out], columns=('model', 'epsilon', 'C', 'kernel', 'mse', 'iter', 'fold'))
            summary = summary.append(output)
    summary_final = pd.DataFrame(summary.groupby(['model', 'epsilon', 'C', 'kernel', 'iter']).mean().sort_values(by=['mse']).reset_index())
    best_epsilon = summary_final['epsilon'].iloc[0]
    best_C = summary_final['C'].iloc[0]
    best_kernel = summary_final['kernel'].iloc[0]
    clf = SVR(C=best_C, epsilon=best_epsilon, kernel=best_kernel).fit(X_train, y_train.values.ravel())
    y_pred = clf.predict(X_test).round(decimals=0)
    mse = mean_squared_error(y_test, y_pred)
    out = ('SVM',{'epsilon':round(best_epsilon,3), 'C':round(best_C,3), 'kernel': best_kernel}, mse)
    mse = pd.DataFrame([out], columns=('model','parameters','mse'))
    return mse, summary


def main():
    input = read_file(PATH)
    input = encode_data(input)

    summary_final = pd.DataFrame()
    all_data_final = pd.DataFrame()
    for model in MODELS:
        summary = pd.DataFrame()
        all_data_summary = pd.DataFrame()
        for samples in SAMPLES:
            train_domain, train_target, dev, test = cross_validation_set(model, input, samples)
            all_data, mse = all(train_domain, train_target, dev, test)

            mse['samples'] = samples
            all_data['samples'] = samples

            summary = summary.append(mse)
            all_data_summary = all_data_summary.append(all_data)
        summary['target'] = model
        all_data_summary['target'] = model
        summary_final = summary_final.append(summary)
        all_data_final = all_data_final.append(all_data_summary)
    summary_final.to_csv('summary_FEDA_test_ratio_.csv', index=False, sep = ',')
    all_data_final.to_csv('FEDA_test_ratio_.csv', index=False, sep = ',')

if __name__ == '__main__':
    main()
