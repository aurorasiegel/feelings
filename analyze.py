#!/usr/bin/env python3
"""
Created on Thu Jul  5 16:12:31 2018
@author: Aurora Siegel
"""
import random
import warnings
import numpy as np
import pandas as pd
from biokit import corrplot
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import statsmodels.api as sm

def invert_values(var):
    """ Invert values in a numpy array """
    return (var.min() + var.max()) - var

def is_issue(name):
    """ Return true if variable name is of an issue variable """
    return 'issue_' in name

def is_demo(name):
    """ Return true if variable name is of an demographic variable """
    return 'demo_' in name

def is_feeling(name):
    """ Return true if variable name is of an feeling variable """
    return 'ft_' in name

def is_party(name):
    """ Return true if variable name is of an party variable """
    return 'party_' in name


def plot_scores(experiments, scores, title):
    """ Plot models in an experiment by their scores """
    ordered_names, ordered_scores, means = [], [], []
    for index, name in enumerate(sorted(experiments)):
        means.append([np.mean(scores[index]), scores[index], name])
    for _, exp_scores, name in sorted(means, key=lambda l: l[0]):
        ordered_names.append(name)
        ordered_scores.append(exp_scores)
    plt.figure(figsize=(8, 8))
    plt.suptitle("Predicting Presidential Candidate Vote based on Feeling Responses")
    plt.title("Each model trained with a 10-fold crossvalidation")
    plt.ylabel('Accuracy')
    plt.ylim(0.75, 1)
    plt.grid()
    plt.boxplot(ordered_scores, whis=[10, 90])
    axes = plt.gca()
    axes.set_yticklabels(['{:,.2%}'.format(x) for x in axes.get_yticks()])
    plt.xticks(np.arange(1, 1 + len(ordered_names)), ordered_names, rotation=90)
    plt.savefig(title + '.png', bbox_inches='tight', dpi=400)
    plt.close()
    print("Saved", title)


def plot_variables(dataframe):
    """ Boxplot all values in a given dataframe """
    names, scores = [], []
    for name in sorted(dataframe):
        names.append(name)
        scores.append(dataframe[name])
    plt.figure(figsize=(12, 6))
    plt.title("Variables")
    plt.boxplot(scores)
    plt.xticks(np.arange(1, 1 + len(names)), names, rotation=90)
    plt.savefig('variables.png', bbox_inches='tight', dpi=400)
    plt.close()


def plot_correlation(dataframe, name):
    """ Create a  biokit-based correlation plot for all variables in the given dataframe """
    clean_df = pd.DataFrame()
    for field_name in dataframe:
        if '_' in field_name:
            subname = field_name.split('_')[1]
        else:
            subname = 'voted_clinton'
        clean_df[subname] = dataframe[field_name]
    corrplot.Corrplot(clean_df).plot()
    plt.savefig('correlation_%s.png' % name, bbox_inches='tight', dpi=400)
    plt.close()


def crossvalidate(Xs, y, k_folds):
    """ Do k-fold crossvalidation on our experiments (Xs) """
    scores = [[] for _ in range(len(Xs))]
    kfold = KFold(n_splits=k_folds)
    for train_index, test_index in kfold.split(y):
        for x_index, X in enumerate(Xs):
            X_train = X.iloc[train_index]
            X_test = X.iloc[test_index]
            y_train = y.iloc[train_index]
            y_test = y.iloc[test_index]
            model = LogisticRegression(solver='lbfgs')
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            scores[x_index].append(score)
    return scores


def run_bivariate(dataframe, independent_name, dependent_name):
    """
    Run a bivariate analysis on two given variables, printing the p-value of the chance that
    the independent variable's values are different when the matching dependent variable's
    value is either 0 or 1 (Clinton or Trump in our study)
    """
    t_0 = dataframe[dataframe[dependent_name] == 0][independent_name]
    t_1 = dataframe[dataframe[dependent_name] == 1][independent_name]
    _, pval = stats.ttest_ind(t_0, t_1, equal_var=False)
    print('P value for %-20s in %s: %.6f' % (independent_name, dependent_name, pval))


def prepare_data(path='data/VOTER_Survey_December16_Release1.csv'):
    """ Prepare a dataframe from our voting csv data and normalize all values """
    voter = pd.read_csv(path)

    # Keep only voters that voted for Trump or Clinton
    votedfor = voter['presvote16post_2016'].copy()
    votedfor[votedfor >= 3.0] = np.NaN

    # Prepare data frame with all variables we care about
    dataframe = pd.DataFrame({
        'issue_crime': (voter["imiss_a_2016"] - 1) / 3,
        'issue_economy': (voter["imiss_b_2016"] - 1) / 3,
        'issue_immigration': (voter["imiss_c_2016"] - 1) / 3,
        'issue_religiousLiberty': (voter["imiss_e_2016"] - 1) / 3,
        'issue_terrorism': (voter["imiss_f_2016"] - 1) / 3,
        'issue_gayRights': (voter["imiss_g_2016"] - 1) / 3,
        'issue_moneyinPolitics': (voter["imiss_k_2016"] - 1) / 3,
        'issue_jobs': (voter["imiss_o_2016"] - 1) / 3,
        'issue_taxes': (voter["imiss_r_2016"] - 1) / 3,
        'issue_abortion': (voter["imiss_t_2016"] - 1) / 3,
        'issue_racialEquality': (voter["imiss_x_2016"] - 1) / 3,
        'issue_genderEquality': (voter["imiss_y_2016"] - 1) / 3,
        'demo_education': (voter["educ_baseline"] - 1) / 5,
        'demo_male': voter["gender_baseline"].replace(2, 0),
        'demo_income': (voter["faminc_baseline"].replace(97, np.NaN).replace(31, 12) - 1) / 11,
        'demo_white': voter["race_baseline"].copy(),
        'demo_black': voter["race_baseline"].copy(),
        'demo_asian': voter["race_baseline"].copy(),
        'demo_hispanic': voter["race_baseline"].copy(),
        'party_democrat' : voter['pid3_baseline'].copy(),
        'party_republican' : voter['pid3_baseline'].copy(),
        'ft_black': voter["ft_black_2016"].replace(997, np.NaN) / 100,
        'ft_white': voter['ft_white_2016'].replace(997, np.NAN) / 100,
        'ft_hispanic': voter['ft_hisp_2016'].replace(997, np.NAN) / 100,
        'ft_asian': voter['ft_asian_2016'].replace(997, np.NAN) / 100,
        'ft_christ' : voter['ft_christ_2016'].replace(997, np.NAN) / 100,
        'ft_muslim' : voter['ft_muslim_2016'].replace(997, np.NAN) / 100,
        'ft_jew' : voter['ft_jew_2016'].replace(997, np.NAN) / 100,
        'ft_feminist' : voter['ft_fem_2016'].replace(997, np.NAN) / 100,
        'ft_immigrant' : voter['ft_immig_2016'].replace(997, np.NAN) / 100,
        'ft_blacklivesmatter' : voter['ft_blm_2016'].replace(997, np.NaN) / 100,
        'ft_wallst' : voter['ft_wallst_2016'].replace(997, np.NAN) / 100,
        'ft_gays' : voter['ft_gays_2016'].replace(997, np.NAN) / 100,
        'ft_unions' : voter['ft_unions_2016'].replace(997, np.NAN) / 100,
        'ft_police' : voter['ft_police_2016'] .replace(997, np.NAN) / 100,
        'ft_altright' : voter['ft_altright_2016'].replace(997, np.NAN) / 100,
        'clinton': votedfor.replace(2, 0),})

    dataframe['demo_white'].values[dataframe['demo_white'] != 1] = 0
    dataframe['demo_black'].values[dataframe['demo_black'] != 2] = 0
    dataframe['demo_black'].replace(2, 1, inplace=True)
    dataframe['demo_hispanic'].values[dataframe['demo_hispanic'] != 3] = 0
    dataframe['demo_hispanic'].replace(3, 1, inplace=True)
    dataframe['demo_asian'].values[dataframe['demo_asian'] != 4] = 0
    dataframe['demo_asian'].replace(4, 1, inplace=True)
    dataframe['party_democrat'].values[dataframe['party_democrat'] != 1] = 0
    dataframe['party_republican'].values[dataframe['party_republican'] != 2] = 0
    dataframe['party_republican'].replace(2, 1, inplace=True)

    for name in dataframe:
        # Make sure higher value means very important
        if is_issue(name):
            dataframe[name] = invert_values(dataframe[name])

        # Feeling variables NANs replace with mean of that feeling thermometer
        if is_feeling(name):
            mean = dataframe[name].mean()
            dataframe[name].fillna(mean, inplace=True)

        # Verify min and max of each variable is normalized to 0, 1
        assert dataframe[name].min() == 0, name
        assert dataframe[name].max() == 1, name

    # Create a friendly dataframe that doesn't have NaNs
    nonan_dataframe = dataframe.dropna(axis=0, how='any')
    return nonan_dataframe


def main():
    """ Main function that runs the full analysis """

    # Initialize defaults for program
    sns.set()
    warnings.filterwarnings("ignore")
    random.seed(2)
    np.random.seed(2)

    data = prepare_data()

    # Prepare lists of variable names for easy use later on
    feeling_names = [name for name in list(data) if is_feeling(name)]
    issue_names = [name for name in list(data) if is_issue(name)]
    control_names = [name for name in list(data) if is_demo(name) or is_party(name)]
    racereligion_names = ['ft_black', 'ft_white', 'ft_asian', 'ft_hispanic',
                          'ft_christ', 'ft_muslim', 'ft_jew']
    hotbutton_names = ['ft_wallst', 'ft_unions', 'ft_immigrant', 'ft_police',
                       'ft_gays', 'ft_blacklivesmatter', 'ft_altright', 'ft_feminist']

    # Analyze our variables individually
    plot_variables(data)
    plot_correlation(data[[x for x in data if is_feeling(x) or x == 'clinton']], 'all')
    for name in feeling_names:
        run_bivariate(data, name, 'clinton')

    # Print out logistic regression coefficients for all variables
    y = data['clinton']
    x = sm.add_constant(data[feeling_names + control_names + issue_names])
    logit = sm.Logit(y, x).fit()
    print(logit.summary())

    # Train models for all our experiments
    y = data['clinton']
    experiments = {'control': control_names,
                   'control+issue': control_names + issue_names,
                   'racerel': control_names + racereligion_names,
                   'hotbutton': control_names + hotbutton_names,
                   'racerel+issue': control_names + issue_names + racereligion_names,
                   'hottbuton+issue': control_names + issue_names + hotbutton_names,
                   'blm': control_names + ['ft_blacklivesmatter'],
                   'black+police': control_names + ['ft_black', 'ft_police']}
    Xs = [sm.add_constant(data[experiments[name]]) for name in sorted(experiments)]
    scores = crossvalidate(Xs, y, 10)

    print("Scores")
    for index, name  in enumerate(sorted(experiments)):
        print("%-16s %5.3f%%" % (name, 100 * np.mean(scores[index])))

    print("T-test P-values")
    for index_1, name_1 in enumerate(sorted(experiments)):
        for index_2, name_2 in enumerate(sorted(experiments)):
            if index_2 <= index_1:
                continue
            _, pval = stats.ttest_ind(scores[index_1], scores[index_2], equal_var=False)
            print("%-16s %-15s %.4f" % (name_1, name_2, pval))
    plot_scores(experiments, scores, "experiments")

if __name__ == "__main__":
    main()
