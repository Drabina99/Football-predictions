import pandas as pd
from sklearn import linear_model
from sklearn.neural_network import MLPClassifier
from pyGRNN import GRNN
from sklearn.utils import shuffle
from termcolor import colored
from tabulate import tabulate
from rbfnn.rbfnn import RBF


def pred_to_int(pred):
    """Function used for linear regression and GRNN
    to approximate float values to integers representing the events"""
    for i in range(len(pred)):
        if -1 <= pred[i] <= -0.33:
            pred[i] = -1
        elif -0.33 < pred[i] <= 0.33:
            pred[i] = 0
        else:
            pred[i] = 1
    return pred.astype(int)


def rbf_correct(rbf):
    """RBFNN for the win of the opponent returns 0, and 2 for the draw,
    so it has to be corrected to appropriate values"""
    for i in range(len(rbf)):
        if rbf[i] == 0:
            rbf[i] = -1
        elif rbf[i] == 2:
            rbf[i] = 0
    return rbf


def int_to_event(pred_k):
    """'Translates' an event from a number to a word"""
    if pred_k == -1:
        pred_ev = 'loss'
    elif pred_k == 0:
        pred_ev = 'draw'
    else:
        pred_ev = 'win'
    return pred_ev


correct_lr = 0
correct_grnn = 0
correct_mlp = 0
correct_rbf = 0

loops = 10

for k in range(loops):
    df = pd.read_csv('db/results.csv', sep=';')
    df = df[5:]
    df = shuffle(df)

    X = df[['RoundNumber', 'IsHome', 'Last5Form', 'Last5GDiff', 'LSRP', 'RL5F']]
    X_info = df[['Team', 'Scored', 'Conceded']]
    X_train = X[:117]
    X_test = X[117:]
    X_test_info = X_info[117:]
    y = df['Type']
    y_train = y[:117]
    y_test = y[117:]

    regr = linear_model.LinearRegression()
    regr.fit(X_train, y_train)

    grnn = GRNN(sigma=3.989999999999998)
    grnn.fit(X_train, y_train)

    mlp = MLPClassifier(hidden_layer_sizes=5, random_state=0)
    mlp.fit(X_train, y_train)

    rbf = RBF(X_train.to_numpy(), y_train.to_numpy(),
                X_test.to_numpy(), y_test.to_numpy(), num_of_classes=3, k=10,
                std_from_clusters=False)
    rbf.fit()

    y_pred_lr = regr.predict(X_test)
    y_pred_grnn = grnn.predict(X_test)
    y_pred_mlp = mlp.predict(X_test)
    y_pred_rbf = rbf_correct(rbf.pred_ty)

    y_pred_lr_real = y_pred_lr.copy()
    y_pred_grnn_real = y_pred_grnn.copy()
    y_pred_rbf_real = y_pred_rbf.copy()

    y_pred_lr = pred_to_int(y_pred_lr)
    y_pred_grnn = pred_to_int(y_pred_grnn)

    table = []

    for i in range(len(y_pred_lr)):
        if y_test.to_numpy()[i] == -1:
            real = 'loss (-1)'
        elif y_test.to_numpy()[i] == 0:
            real = 'draw (0)'
        else:
            real = 'win (1)'

        pred_lr = int_to_event(y_pred_lr[i])
        pred_grnn = int_to_event(y_pred_grnn[i])
        pred_mlp = int_to_event(y_pred_mlp[i])
        pred_rbf = int_to_event(y_pred_rbf[i])

        if X_test.to_numpy()[i][1] == 1:
            home = "Man Utd"
            away = X_test_info.to_numpy()[i][0]
            home_goals = X_test_info.to_numpy()[i][1]
            guests_goals = X_test_info.to_numpy()[i][2]
        else:
            home = X_test_info.to_numpy()[i][0]
            away = "Man Utd"
            guests_goals = X_test_info.to_numpy()[i][1]
            home_goals = X_test_info.to_numpy()[i][2]

        game = "%s %d:%d %s" % (home, home_goals, guests_goals, away)

        if y_pred_lr[i] == y_test.to_numpy()[i]:
            pred_lr = colored(pred_lr, "green")
            correct_lr += 1
        else:
            pred_lr = colored(pred_lr, "red")

        if y_pred_grnn[i] == y_test.to_numpy()[i]:
            pred_grnn = colored(pred_grnn, "green")
            correct_grnn += 1
        else:
            pred_grnn = colored(pred_grnn, "red")

        if y_pred_mlp[i] == y_test.to_numpy()[i]:
            pred_mlp = colored(pred_mlp, "green")
            correct_mlp += 1
        else:
            pred_mlp = colored(pred_mlp, "red")

        if y_pred_rbf[i] == y_test.to_numpy()[i]:
            pred_rbf = colored(pred_rbf, "green")
            correct_rbf += 1
        else:
            pred_rbf = colored(pred_rbf, "red")

        table.append([game, pred_lr, pred_grnn, pred_mlp, pred_rbf, real, y_pred_lr_real[i],
                        y_pred_grnn_real[i], y_pred_mlp[i], y_pred_rbf_real[i]])

    print(colored("\n\nRESULTS:\n", "magenta"))
    print(tabulate(table, headers=['GAME', 'PRED_REGR', 'PRED_GRNN', 'PRED_MLP', 'PRED_RBF', 'REAL',
                                       'LR', 'GRNN', 'MLP', 'RBF'], tablefmt='orgtbl'))

div = loops * len(y_test)
summary = [[colored("LINEAR REGRESSION", "magenta"), correct_lr, 100 * correct_lr / div],
           [colored("GRNN", "magenta"), correct_grnn, 100 * correct_grnn / div],
           [colored("MLP", "magenta"), correct_mlp, 100 * correct_mlp / div],
           [colored("RBFNN", "magenta"), correct_rbf, 100 * correct_rbf / div]]
test_size_str = 'CORRECT (/%d)' % len(y_test)
print(colored("\n\nSUMMARY:\n", "magenta"))
print(tabulate(summary, headers=['MODEL', test_size_str, 'ACCURACY (%)'], tablefmt='orgtbl'))
