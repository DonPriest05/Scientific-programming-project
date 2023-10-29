import seaborn as sns
import numpy as np
from numpy.matlib import repmat
from scipy.stats import ks_2samp, gaussian_kde, ttest_ind
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import impute
from sklearn.decomposition import PCA
from imblearn.under_sampling import CondensedNearestNeighbour, NearMiss, TomekLinks, RandomUnderSampler
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.metrics import roc_curve, auc
from keras.models import Model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Input
from keras.layers import Concatenate
from keras.layers import BatchNormalization
import tensorflow as tf
import tensorflow.keras.backend as K
import random as rn
from sklearn.metrics import accuracy_score, recall_score, precision_score

pd.options.mode.chained_assignment = None  # default='warn'


def set_seed_val(seed_val):
    """
    Set the random seed
    ----------

    seed_val: int
        The seed value
    
        
    Returns
    -------
    void
        
    """
    np.random.seed(seed_val)
    rn.seed(seed_val)
    tf.keras.utils.set_random_seed(seed_val)
    tf.random.set_seed(seed_val)


def open_data(directory):
    """
    Gets the file directory and reads the data into a dataframe
    ----------

    directory: string, optional
        Set the directory of the file
    
        
    Returns
    -------
    
    big_data_df: pandas DataFrame
        Data in the file formated as dataframe
        
    """
    print(directory)
    global df_pheno
    try:
        big_data_df = pd.read_excel(directory)
    except:
        try:
            big_data_df = pd.read_csv(directory)
        except:
            print('format not accepted')

    return big_data_df


def scale_for_model(cat_train_df=pd.DataFrame(), num_train_df=pd.DataFrame(), cat_test_df=pd.DataFrame(),
                    num_test_df=pd.DataFrame()):
    """
    Gets the file directory and reads the data into a dataframe
    ----------

    cat_train_df: dataframe, optional
        Dataframe that contains the categorial data of the train set
    
    num_train_df: dataframe, optional
        Dataframe that contains the numerical data of the test set
        
    cat_test_df: dataframe, optional
        Dataframe that contains the categorial data of the test set
        
    num_test_df: dataframe, optional
        Dataframe that contains the numerical data of the test set
        
    Returns
    -------
    
    cat_train_df: dataframe, optional
        Train set categorical data normalized and standard scaled
    
    num_train_df: dataframe, optional
        Train set numerical data normalized and standard scaled
        
    cat_test_df: dataframe, optional
        Test set categorical data normalized and standard scaled
        
    num_test_df: dataframe, optional
        Test set numerical data normalized and standard scaled
        
    """

    cat_columns = cat_train_df.columns.tolist()
    num_columns = num_train_df.columns.tolist()

    if not num_train_df.empty:
        # Fit and transform numerical data using a StandardScaler and Normalizer
        scaler_num = StandardScaler()
        normalizer_num = Normalizer()

        num_train_df = pd.DataFrame(scaler_num.fit_transform(num_train_df), columns=num_columns,
                                    index=num_train_df.index)
        num_train_df = pd.DataFrame(normalizer_num.fit_transform(num_train_df), columns=num_columns,
                                    index=num_train_df.index)
        num_test_df = pd.DataFrame(scaler_num.transform(num_test_df), columns=num_columns, index=num_test_df.index)
        num_test_df = pd.DataFrame(normalizer_num.transform(num_test_df), columns=num_columns, index=num_test_df.index)

    if not cat_train_df.empty:
        # Fit and transform categorical data using a StandardScaler and Normalizer
        scaler_cat = StandardScaler()
        normalizer_cat = Normalizer()

        cat_train_df = pd.DataFrame(scaler_cat.fit_transform(cat_train_df), columns=cat_columns,
                                    index=cat_train_df.index)
        cat_train_df = pd.DataFrame(normalizer_cat.fit_transform(cat_train_df), columns=cat_columns,
                                    index=cat_train_df.index)
        cat_test_df = pd.DataFrame(scaler_cat.transform(cat_test_df), columns=cat_columns, index=cat_test_df.index)
        cat_test_df = pd.DataFrame(normalizer_cat.transform(cat_test_df), columns=cat_columns, index=cat_test_df.index)

    return cat_train_df, num_train_df, cat_test_df, num_test_df


def split_data(big_data_df, strat='mean', test=False, imputer=None, imputer_cat=None):
    """
    Splits the data into a dataframe with categorial variables and a 
    dataframe with numerical values. Where missing, imputes data of the
    numercial variables
    ----------

    big_data_df: pandas DataFrame
        Dataframe that contains the data
    
    strat: string, optional
        Strategy to use when imputing.
        See sklearn.impute for more information
        
    imputer: sklearn.impute object
        Imputer object used to impute data
        
    Returns
    -------
    
    df_categorial: pandas DataFrame
        Dataframe containing the categorial data
        
    df_numeric: pandas DataFrame
        Dataframe containing the numeric data
        
    imputed_data_df: pandas DataFrame
        Dataframe containing both categorial and numerical data with the 
        numerical data imputed
        
    imputer: sklearn.impute object
        Fitted impute object to fit future data with
        
    """

    # Numerical imputer initialization
    if imputer is None:
        imputer = impute.SimpleImputer(missing_values=np.nan, strategy=strat)

    # Categorical imputer initialization
    if imputer_cat is None:
        # Constant is set as default
        if strat == 'mean' or strat == 'median':
            strat = 'constant'
        if strat == 'constant':
            imputer_cat = impute.SimpleImputer(missing_values=np.nan, strategy=strat, fill_value='NA')
        else:
            imputer_cat = impute.SimpleImputer(missing_values=np.nan, strategy=strat)

    # if there are categorical columns, extract them, otherwise return an empty dataframe
    if 'object' in big_data_df.dtypes.values:
        df_categorial = big_data_df.select_dtypes(include=['object']).copy()
        # df_categorial.fillna('not applicable', inplace=True)
        if test == False:
            imputer_cat.fit(df_categorial)
        df_categorial[:] = imputer_cat.transform(df_categorial)
    else:
        df_categorial = pd.DataFrame()

    # if there are numeric columns, extract and impute them, otherwise return an empty dataframe
    if 'object' not in big_data_df.dtypes.values or len(big_data_df.select_dtypes(exclude=['object']).columns) > 0:
        df_numeric = big_data_df.select_dtypes(exclude=['object']).copy()
        if not df_numeric.empty:
            if test is False:
                imputer.fit(df_numeric)
            df_numeric[:] = imputer.transform(df_numeric)
    else:
        df_numeric = pd.DataFrame()

    # Combine dataframes
    imputed_data_df = pd.concat([df_categorial, round(df_numeric)], axis=1)
    imputed_data_df = imputed_data_df.reindex(big_data_df.columns, axis=1)

    return df_categorial, df_numeric, imputed_data_df, imputer, imputer_cat


def plot_explained_variance(pca):
    """
    Plots how much each variable contributes to each principal component
    ----------
    pca: sklearn.decomposition.PCA object
        Contains the pca object
        
    Returns
    ----------
    void
    
    """

    # Plot explained variance
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)  # Calculate cumulative explained variance

    # Find the index where the cumulative variance first exceeds 0.99
    index_99 = np.argmax(cumulative_variance >= 0.99)

    # Create a new figure and axis
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Bar plot for individual explained variance (left y-axis) up to index_99
    ax1.bar(range(index_99 + 1), explained_variance[:index_99 + 1], alpha=0.7, align='center',
            label='Individual explained variance')
    ax1.set_ylabel('Explained variance ratio')
    ax1.set_xlabel('Principal components')
    ax1.set_xticks(range(index_99 + 1))
    ax1.set_title('Explained Variance Ratios and Cumulative Probability')
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5, axis='y')

    # Create the second y-axis for the cumulative explained variance (right y-axis)
    ax2 = ax1.twinx()
    ax2.plot(range(index_99 + 1), cumulative_variance[:index_99 + 1], color='r', marker='o',
             label='Cumulative explained variance')
    ax2.set_ylim(0, 1)
    ax2.set_ylabel('Cumulative Probability')

    # set legend for both plots
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc=0)

    plt.tight_layout()

    plt.show()


def plot_pca_loadings(pca, component_num, feature_names):
    """
    Plots the loadings for a specified principal component.
    ----------
    pca: sklearn.decomposition.PCA
        The fitted PCA object
        
    component_num: int
        The component number to visualize (starts at 0)
        
    feature_names: list
        List of feature names
        
    Returns
    ----------
    void
    
    """
    loadings = pca.components_[component_num]

    # Sort the loadings and corresponding feature names
    sorted_indices = np.argsort(np.abs(loadings))[::-1]  # sort by absolute value but keep original sign
    sorted_loadings = loadings[sorted_indices]
    sorted_features = np.array(feature_names)[sorted_indices]

    plt.figure(figsize=(10, 6))
    plt.barh(sorted_features, sorted_loadings)
    plt.xlabel('Loading Value')
    plt.ylabel('Feature')
    plt.title('loadings for Principal Component {}'.format(component_num + 1))
    plt.gca().invert_yaxis()  # Show the feature with the highest absolute loading at the top
    plt.tight_layout()
    plt.tick_params(axis='y', labelsize=5)
    plt.show()


def pca_analysis(cat_data_train, num_data_train, cat_data_test, num_data_test, \
                 train_pheno, test_pheno, x_component=0, y_component=1):
    """
    Performs PCA analysis on the numerical columns of the data
    ----------

    num_data_train: pandas DataFrame
        Dataframe that contains the numerical data of the training set
        
    cat_data_train: pandas DataFrame
        Dataframe that contains the categorial dataof the training set
        
    num_data_test: pandas DataFrame
        Dataframe that contains the numerical data of the testing set
        
    cat_data_test: pandas DataFrame
        Dataframe that contains the categorial dataof the testing set
        
    
    train_pheno: pandas DataFrame
        Contains the phenotypes of interest for the training set
        
    test_pheno: pandas DataFrame
        Contains the phenotypes of interest for the test set
            
    x_component: integer
        The PCA component to plot on the x-axis
        
    y_component: integer
        The PCA component to plot on the y-axis
        
    Returns
    -------
    
    void
        
    """

    feature_names = []
    cat_data_train, num_data_train, cat_data_test, num_data_test = \
        scale_for_model(cat_data_train, num_data_train, cat_data_test, num_data_test)

    if not num_data_train.empty:
        feature_names.extend(list(num_data_train.columns))

    # Normalize and One-Hot Encode cat_data
    if not cat_data_train.empty:
        feature_names.extend(list(cat_data_train.columns))

    # combine the categorial and numerical data
    data_combined_train = pd.concat([cat_data_train, num_data_train], axis=1)
    data_combined_test = pd.concat([cat_data_test, num_data_test], axis=1)

    pca = PCA()
    princ_comp_train = pca.fit_transform(data_combined_train)

    # Scatter plot using the first two PCs
    plt.figure()
    plt.scatter(princ_comp_train[:, x_component], princ_comp_train[:, y_component], label='None')

    for pheno in list(train_pheno):
        princ_comp_train_temp = np.multiply \
            (princ_comp_train, np.tile(np.array(train_pheno[pheno]), (princ_comp_train.shape[1], 1)).T)

        princ_comp_train_temp = princ_comp_train_temp[~np.all(princ_comp_train_temp == 0, axis=1)]
        plt.scatter(princ_comp_train_temp[:, x_component], princ_comp_train_temp[:, y_component], label=pheno)

    plt.legend()
    plt.xlabel(f'Principal Component {x_component}')
    plt.ylabel(f'Principal Component {y_component}')

    plt.title("PCA train set")
    plt.show()

    plot_pca_loadings(pca, x_component, feature_names)
    plot_pca_loadings(pca, y_component, feature_names)
    plot_explained_variance(pca)

    princ_comp_test = pca.transform(data_combined_test)

    # Scatter plot using the chosen PCs
    plt.figure()
    plt.scatter(princ_comp_test[:, x_component], princ_comp_test[:, y_component], label='None')

    for pheno in list(test_pheno):
        princ_comp_test_temp = np.multiply \
            (princ_comp_test, np.tile(np.array(test_pheno[pheno]), (princ_comp_test.shape[1], 1)).T)

        princ_comp_test_temp = princ_comp_test_temp[~np.all(princ_comp_test_temp == 0, axis=1)]
        plt.scatter(princ_comp_test_temp[:, x_component], princ_comp_test_temp[:, y_component], label=pheno)

    plt.legend()
    plt.xlabel(f'Principal Component {x_component}')
    plt.ylabel(f'Principal Component {y_component}')

    plt.title("PCA test set")
    plt.show()


def visualize_data(cat_data, selection, num_data, df_pheno):
    """
    Splits the data into a dataframe with categorial variables and a 
    dataframe with numerical values. Where missing, imputes data of the
    numercial variables
    ----------

    cat_data: pandas DataFrame
        Contains the categorial data
        
    selection: string
        The column name of which the results need to be visualized
        
    num_data: pandas DataFrame
        Contains the numerical data
        
    df_pheno: pandas DataFrame
        Contains the phenotypes of interest
        
    Returns
    -------
    
    fig: figure object
        Boxplot figure that visualizes the phenotype distribution depending
        on different categories
        
    """
    # if distribution of phenotypes depending on a category is of interest
    if selection != 'Total numbers':
        # get the counts for each phenotype in the category
        levels = set(list(cat_data[selection]))
        height_pheno = np.zeros([len(list(levels)), len(list(df_pheno))])
        height = np.zeros([len(list(levels))])
        bars = np.array(list(levels))
        column_names = df_pheno.columns

        for i, level in enumerate(list(levels)):
            height[i] = sum(cat_data[selection] == level)

            for j, pheno in enumerate(list(df_pheno)):
                height_pheno[i, j] = sum(np.multiply(np.array(cat_data[selection] \
                                                              == level), np.array(df_pheno[pheno])))

        height_none = np.subtract(height, np.sum(height_pheno, axis=1))

        # create the barplot
        fig, ax = plt.subplots()
        x = np.arange(len(bars))
        width = 0.1
        num_phenotypes = np.shape(height_pheno)[1] + 1

        for i in range(num_phenotypes):
            if i == 0:
                rects = ax.bar(x - i * width, height_none, width, label='None', log=True)
                ax.bar_label(rects, padding=3)
            else:
                if i % 2 == 0:
                    rects = ax.bar(x - i / 2 * width, height_pheno[:, i - 1], width, label=column_names[i - 1],
                                   log=True)
                    ax.bar_label(rects, padding=3)
                else:
                    rects = ax.bar(x + i / 2 * width, height_pheno[:, i - 1], width, label=column_names[i - 1],
                                   log=True)
                    ax.bar_label(rects, padding=3)

        ax.set_xlabel(selection)
        ax.set_ylabel('number')
        ax.set_xticks(x, bars)
        ax.legend()
        fig.tight_layout()

    # if no category needs to be selected but the total distribution is
    # of interest
    else:
        fig, ax = plt.subplots()
        sum_cols = df_pheno.sum(axis=1)
        none_pheno = []
        [none_pheno.append(1) if row == 0 else none_pheno.append(0) for row in sum_cols]
        all_phenos = pd.concat([df_pheno, pd.DataFrame(none_pheno, columns=['None'])], axis=1)
        counts_phenos = all_phenos.sum(axis=0)
        counts_phenos.plot(kind='bar', ax=ax, ylabel='number')
    return fig


def macro_precision(y_true, y_pred):
    """
    Computes the macro precision when given the ground truth and predicted values
    ----------

    y_true: pandas dataframe
        Contains the true ground truth classes
        
    y_pred: pandas dataframe
        Contains the predicted classes
        
    Returns
    -------
    
    macro precision: float
        Macro precision value   
    """
    # Convert predictions to one-hot vectors
    y_pred = K.round(K.clip(y_pred, 0, 1))
    
    # Calculate precision for each class
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)), axis=0)
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)), axis=0)
    precision = true_positives / (predicted_positives + K.epsilon())
    
    # Average precision across all classes
    macro_precision = K.mean(precision)
    
    return macro_precision

def macro_recall(y_true, y_pred):
    """
    Computes the macro recall when given the ground truth and predicted values
    ----------

    y_true: pandas dataframe
        Contains the true ground truth classes
        
    y_pred: pandas dataframe
        Contains the predicted classes
        
    Returns
    -------
    
    macro recall: float
        Macro recall value   
    """
    # Convert predictions to one-hot vectors
    y_pred = K.round(K.clip(y_pred, 0, 1))
    
    # Calculate recall for each class
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)), axis=0)
    actual_positives = K.sum(K.round(K.clip(y_true, 0, 1)), axis=0)
    recall = true_positives / (actual_positives + K.epsilon())
    
    # Average recall across all classes
    macro_recall = K.mean(recall)
    
    return macro_recall

def random_pred(y_true):
    """
    A random classifier, the classes it predicts are random
    ----------

    y_true: pandas dataframe
        Contains the true ground truth classes
        
        
    Returns
    -------
    
    np.mean(accuracies): float
        Mean of the accuracies computed over every iteration
        
    np.mean(precisions): float
        Mean of the precisions computed over every iteration
        
    np.mean(recalls): float
        Mean of the recalls computed over every iteration   

    """
    accuracies = []
    recalls = []
    precisions = []
    
    y_random_range = np.random.choice(y_true.columns, size=len(y_true)*100)
    y_random_splits = np.split(y_random_range, 100)
    
    # Runs 100 times and takes the average
    for y_random in y_random_splits:
        y_random_dummy = pd.get_dummies(y_random).reindex(columns=y_true.columns, fill_value=0)

        accuracies.append(accuracy_score(y_true.values.argmax(axis=1), y_random_dummy.values.argmax(axis=1)))
        recalls.append(recall_score(y_true, y_random_dummy, average='macro'))
        precisions.append(precision_score(y_true, y_random_dummy, average='macro'))
    
    return np.mean(accuracies), np.mean(precisions), np.mean(recalls)

def model(cat_df, num_df, layers_model, dropout, num=True, cat=True, seed=0):
    """
    Creates a model instance
    ----------

    cat_df: pandas DataFrame
        Contains the categorial data
        
    num_df: pandas DataFrame
        Contains the numerical data
        
    layers_model: string
        String with comma delimiters, each item contains information about a
        layer
        Example: x,y,z creates 3 layers with x nodes in the first layer,
        y nodes in the second layer etc.
        
        
    dropout: float
        Floating point number between 0-1 that determines the dropout
        rate between layers
        
    num: Bool, optional
        When true the data contains numerical data
        
    cat: Bool, optional
        When true the data contains categorial variables
        
    Returns
    -------
    
    model: keras.Model
        The model instance with parameters as specified in the function
        
    """
    np.random.seed(seed)
    rn.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    tf.random.set_seed(seed)
    initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1., seed=seed)

    # check if there is numerical input and/or categorial input
    # create seperate input layers for both if they exist

    if num == True:
        input_num = Input(shape=(np.shape(num_df)[1],), name='input_num')
    if cat == True:
        input_cat = Input(shape=(np.shape(cat_df)[1],), name='input_cat')

    layers_model = layers_model.split(',')
    layers_model_int = []
    [layers_model_int.append(int(index)) for index in layers_model]
    layers_model = layers_model_int

    # if both numerical and categorial data exist in the data concatenate
    # the 2 input layers to creat one big input layer
    if num == True and cat == True:
        x = (Concatenate()([input_cat, input_num]))
    # if only numerical data exists make that the only input layer
    elif num == True:
        x = input_num
    # if only categorial data exists make that the only input layer
    elif cat == True:
        x = input_cat

    # create all layers as specified by the layers_model variable
    for i in range(len(layers_model) - 1):
        # normalize data
        x = BatchNormalization()(x)

        x = Dropout(float(dropout), seed=seed)(x)
        x = Dense(layers_model[i], activation='relu', kernel_initializer=initializer, bias_initializer='zeros')(x)

    # add final output layer
    x = BatchNormalization()(x)
    x = Dropout(float(dropout), seed=seed)(x)
    out = Dense(layers_model[-1], activation='sigmoid', name='output', kernel_initializer=initializer,
                bias_initializer='zeros')(x)

    # create model object with specified input and output layer
    if num == True and cat == True:
        model = Model(inputs=[input_cat, input_num], outputs=out)
    elif num == True:
        model = Model(inputs=input_num, outputs=out)
    elif cat == True:
        model = Model(inputs=input_cat, outputs=out)

    # compile model and select metrics to keep track of
    model.compile(loss='categorical_crossentropy', \
                  metrics=["categorical_accuracy", macro_precision, macro_recall])
    return model


def categorize_ui(df1, variable, lower, upper, new_name, filter_cat=None):
    """
    Changes numerical variables to categorial variables by giving values
    between lower and upper bound a new category name
    ----------

    df1: pandas DataFrame
        Contains all of the data
        
    variable: string
        name of the variable of interest
        
    lower: float
        Lower bound of the value to change
        
    upper: float
        Upper bound of the value to change
        
    new_name: string
        The name of the new category
        
    filter_cat: string
        Name of another category to use ase additional filter
        
    Returns
    -------
    
    df1: pandas DataFrame
        New dataframe that contains the new categorial data
        
    """

    for var in variable:
        filter1 = []
        filter2 = []
        [filter1.append(1) if isinstance(val, str) == False and val >= lower else filter1.append(0) for i, val in
         enumerate(np.ravel(df1[var]))]
        [filter2.append(1) if isinstance(val, str) == False and val <= upper else filter2.append(0) for i, val in
         enumerate(np.ravel(df1[var]))]
        filter1 = np.array(filter1)
        filter2 = np.array(filter2)

        # filter based on the additional category if given
        if filter_cat is not None:

            filter3 = []
            filter_cat = filter_cat.split('_')

            [filter3.append(1) if val == filter_cat[1] else filter3.append(0) for val in df1[filter_cat[0]]]
            filter3 = np.array(filter3)

            combined_filter = np.multiply(np.multiply(filter1, filter2), filter3)

        # filter based on the values of lower and upper bound
        else:
            combined_filter = np.multiply(filter1, filter2)

        row_filter = np.where(combined_filter == 1)[0]
        df1.iloc[row_filter, df1.columns.get_loc(var)] = new_name

    return df1


def abbreviate_label(label, max_length=10):
    """Abbreviates a label if it's longer than max_length."""
    if len(label) > max_length:
        return label[:max_length - 3] + "..."
    return label


def compare_dist_cat(fullset, subset):
    # Combine counts from all categorical variabls
    observed_counts = subset.sum(axis=0)
    full_counts = fullset.sum(axis=0)

    # Calculate proportions
    observed_prop = observed_counts / len(subset)
    full_prop = full_counts / len(fullset)

    # Calculate empirical CDFs
    observed_cdf = np.cumsum(observed_prop)

    full_cdf = np.cumsum(full_prop)


    # Use the K-S test
    ks_statistic, p_value = ks_2samp(observed_cdf, full_cdf)

    # Interpretation
    alpha = 0.05
    if p_value < alpha:
        print("The distributions of cat variables in the subset and the entire dataset are statistically different.")
    else:
        print(
            "The distributions of cat variables in the subset and the entire dataset are NOT statistically different.")

    fig, axs = plt.subplots(2, 1, figsize=(10, 12))
    
    # Calculate the difference between the two CDFs
    difference = observed_cdf - full_cdf
    
    # plot the two CDFs
    axs[0].plot(range(1, len(observed_cdf) + 1), observed_cdf, 'r--', label='Subset combined CDF', marker='o')
    axs[0].plot(range(1, len(full_cdf) + 1), full_cdf, 'b-', label='Full Dataset combined CDF', marker='o', alpha=0.4)
    axs[0].set_title('ECDFs for Subset and Full Dataset')
    axs[0].set_xlabel('Categories (encoded)')
    axs[0].set_ylabel('Cumulative Proportion')
    axs[0].legend()
    axs[0].grid(True)
    
    # plot the difference
    axs[1].plot(range(1, len(difference) + 1), difference, 'g-', label='Difference between CDFs', marker='o')
    axs[1].axhline(0, color='black', linestyle='--')  # Add a line at y=0 for reference
    axs[1].set_title('Difference between the two CDFs')
    axs[1].set_xlabel('Categories (encoded)')
    axs[1].set_ylabel('Difference in Cumulative Proportion')
    axs[1].legend()
    axs[1].grid(True)
    
    plt.tight_layout()
    plt.show()

def compare_dist_num(fullset, subset):
    # Number of variables
    n_vars = len(fullset.columns)

    # Create subplots
    fig, axes = plt.subplots(nrows=1, ncols=n_vars, figsize=(5 * n_vars, 6))

    for idx, col in enumerate(fullset.columns):
        ax = axes[idx]

        # KS Test for each variable
        ks_stat, p = ks_2samp(fullset[col], subset[col])

        # Plot boxplots
        ax.boxplot([fullset[col], subset[col]], patch_artist=True)

        # Abbreviate tick labels
        abbreviated_label_full = abbreviate_label(f"{col} Full")
        abbreviated_label_sub = abbreviate_label(f"{col} Sub")

        # Set the abbreviated tick labels
        ax.set_xticklabels([abbreviated_label_full, abbreviated_label_sub], fontsize=8)

        # Abbreviate title and set
        abbreviated_title = abbreviate_label(f"{col} Distribution")
        ax.set_title(f"{abbreviated_title} (p={p:.4f})", fontsize=10)

        # Indicate significance on the plot if p < 0.05
        if p < 0.05:
            y_text = max(max(fullset[col]), max(subset[col])) + 0.1
            ax.text(1.5, y_text, '*', fontsize=12, ha='center')

        # Reduce font size for y-axis labels
        ax.tick_params(axis='y', labelsize=8)

    plt.tight_layout()
    plt.show()


def resample(cat, num, ph, len_data, under, over, name, neighbours=1, seed=0, sampling_strat=1):
    """
    Resamples the data when there is an inbalance of classes using one of the
    alogrithms provided. Either oversamples which assigns majority class samples
    as one of the underepresented classes or undersamples which removes certain
    samples from the majority class
    ----------

    cat: pandas DataFrame
        Contains the categorial data
        
    num: pandas DataFrame
        Contains the numerical data
            
    ph: pandas DataFrame
        Contains the phenotypes
        
    len_data: int
        Contains size of the samples in the total data
        
    under: Bool
        True when undersampling is chosen
        
    over: Bool
        True when oversampling is chosen
        
    name: string
        Name of resampling method used
        
    neighbours: int, optional
        For 'Nearmiss'and 'CNN' selects the amount of neighbours used in the algorithm
        for more details check imblearn.under_sampling
        
    seed: int
        The value to which the seed will be set
        
    sampling_strat: float
        The ratio of the classes after resampling (1 gives equal amount of samples)
        
    Returns
    -------
    
    data_re_comb: pandas DataFrame
        Input dataframe which has been resampled
        
    pheno_re: pandas DataFrame
        The corresponding phenotype labels
           
    """

    # set seed
    np.random.seed(seed)
    rn.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    tf.random.set_seed(seed)

    pheno = ph
    original_indices = cat.index
    
    # Prepare Categorical Data
    if not cat.empty:
        data_cat = pd.get_dummies(cat)
        data_cat_cols = data_cat.columns

        # Normalize the one-hot encoded categorical data
        normal_transform = Normalizer()
        data_cat = pd.DataFrame(normal_transform.fit_transform(data_cat), columns=data_cat_cols)
    else:
        data_cat = pd.DataFrame()

    # Prepare Numerical Data
    if not num.empty:
        num_cols = num.columns

        # Normalize numerical data
        num = normal_transform.fit_transform(num)

        # Standardize numerical data
        scaler = StandardScaler()
        num = scaler.fit_transform(num)
        num = pd.DataFrame(num, columns=num_cols)
    else:
        num = pd.DataFrame()
        
    all_cols = list(data_cat.columns) + list(num.columns)
    data = pd.concat([data_cat, num], axis=1)
    data.index = original_indices
    
    # convert to numpy format
    data = np.array(data)
    pheno = np.array(pheno)
    pheno = np.multiply(pheno, np.array(list(range(1, len(pheno.T) + 1))))
    pheno = np.max(pheno, axis=1)

    if under == True:
        methods = ['Nearmiss1', 'Nearmiss2', 'Nearmiss3', 'CNN', 'TomeKLinks', 'Random']
        method = [method for method in methods if method == name][0]

        if method == 'Nearmiss1':
            undersample = NearMiss(version=1, n_neighbors=neighbours)
            data_re, pheno_re = undersample.fit_resample(data, pheno)

        if method == 'Nearmiss2':
            undersample = NearMiss(version=2, n_neighbors=neighbours)
            data_re, pheno_re = undersample.fit_resample(data, pheno)

        if method == 'Nearmiss3':
            undersample = NearMiss(version=3, n_neighbors_ver3=neighbours)
            data_re, pheno_re = undersample.fit_resample(data, pheno)

        if method == 'CNN':
            undersample = CondensedNearestNeighbour(n_neighbors=neighbours)
            data_re, pheno_re = undersample.fit_resample(data, pheno)

        if method == 'TomeKLinks':
            undersample = TomekLinks()
            data_re, pheno_re = undersample.fit_resample(data, pheno)

        if method == 'Random':
            undersample = RandomUnderSampler(random_state=seed, sampling_strategy=sampling_strat)
            data_re, pheno_re = undersample.fit_resample(data, pheno)

        selected_indices = [original_indices[i] for i in undersample.sample_indices_]
        data_re = pd.DataFrame(data_re, columns=all_cols, index=selected_indices)
        #pheno_re = pd.DataFrame(pheno_re, index=list(selected_indices))

    elif over == True:
        methods = ['SMOTE', 'ADASYN']
        method = [method for method in methods if method == name][0]

        if method == 'SMOTE':
            oversample = SMOTE()
            data_re, pheno_re = oversample.fit_resample(data, pheno)

        if method == 'ADASYN':
            oversample = ADASYN()
            data_re, pheno_re = oversample.fit_resample(data, pheno)
            
        # After resampling, check and correct any overlapping indices for the oversampled data
        data_re = pd.DataFrame(data_re, columns=all_cols)
        synthetic_indices = range(len_data + 1, len_data + 1 + len(data_re))
        data_re.index = synthetic_indices
        #pheno_re = pd.DataFrame(pheno_re, index=synthetic_indices)

    # convert back to dataframe format
    data_re_cat = data_re[data_cat.columns]
    data_re_num = data_re[num.columns]
    resampled_indices = data_re.index

    data_re_cat = undummify(data_re_cat)
    all_cols = list(cat.columns) + list(num.columns)
    data_re_comb = pd.DataFrame(pd.concat([data_re_cat, data_re_num], axis=1), columns=all_cols, index = resampled_indices)

    pheno_temp = np.zeros([len(pheno_re), np.max(pheno_re)])
    for i, p in enumerate(pheno_re):
        pheno_temp[i][p - 1] = 1

    pheno_re = pheno_temp
    pheno_re = pd.DataFrame(pheno_re, columns=ph.columns, index = resampled_indices)

    if not data_cat.empty:
        subset_cat = data_re[data_cat_cols]
        compare_dist_cat(data_cat, subset_cat)

    if not num.empty:
        subset_num = data_re[num_cols]
        compare_dist_num(num, subset_num)

    return data_re_comb, pheno_re


def aggregate_dummy_importance(loss_changes, feature_names, prefix_sep="_"):
    """
    Aggregate the loss changes for dummy variables into their original categorical variables

    Parameters
    ----------
    loss_changes: list floats
        Loss changes (from the original loss) for each feature
    feature_names: list of strings
        Names of the features

    Returns
    ----------
    aggregated_changes: dictionary of floats
        Aggregated loss changes for each original feature
    """

    aggregated_changes = {}
    for feature, change in zip(feature_names, loss_changes):
        # Check if the feature is a dummy feature
        if prefix_sep in feature:
            orig_feature = feature.split(prefix_sep)[0]
        else:
            orig_feature = feature

        # Aggregate the loss changes
        if orig_feature not in aggregated_changes:
            aggregated_changes[orig_feature] = []
        aggregated_changes[orig_feature].append(change)

    # estimate the log change over aggregated features
    for feature, changes in aggregated_changes.items():
        aggregated_changes[feature] = np.mean(np.array(changes))

    return aggregated_changes


def find_features(original_loss, permuted_losses, feature_names):
    """
    Extracts the features that have a large change in loss due to shuffling of 
    values in the column.
    The top 20 features will be extracted and plotted

    Parameters
    ----------
    original_loss: list/numpy array of floats
        Contains loss values of original model evaluation
    permuted_losses : TYPE
        Contains metric values of model eval with shuffeled data
    feature_names: list of string
        Name of the columns of the data

    Returns
    -------
    None.

    """

    loss_changes = []
    vals = []
    [vals.append(metric[1]) for metric in permuted_losses]
    [loss_changes.append(abs(original_loss - y)) for y in vals]

    # Aggregate the importance of dummy variables
    combined_losses = aggregate_dummy_importance(loss_changes, feature_names)

    # Sort the aggregated loss changes
    sorted_features = sorted(combined_losses, key=combined_losses.get, reverse=True)
    sorted_changes = [combined_losses[feature] for feature in sorted_features]

    # Extract the top features
    num_features = 30
    top_features = sorted_features[:num_features]
    top_changes = sorted_changes[:num_features]

    # Plotting
    fig, ax = plt.subplots()
    ax.barh(top_features, top_changes, align='center')
    ax.set_xlabel('Aggregated Loss Change when Permutated')
    ax.invert_yaxis()  # So that the feature with the highest change appears on top
    plt.show()

    # return top[0:10]


# https://stackoverflow.com/questions/50607740/reverse-a-get-dummies-encoding-in-pandas
def undummify(df, prefix_sep="_"):
    cols2collapse = {
        item.split(prefix_sep)[0]: (prefix_sep in item) for item in df.columns
    }
    series_list = []
    for col, needs_to_collapse in cols2collapse.items():
        if needs_to_collapse:
            undummified = (
                df.filter(like=col)
                    .idxmax(axis=1)
                    .apply(lambda x: x.split(prefix_sep, maxsplit=1)[1])
                    .rename(col)
            )
            series_list.append(undummified)
        else:
            series_list.append(df[col])
    undummified_df = pd.concat(series_list, axis=1)
    return undummified_df
