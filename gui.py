import pip
import os
import sys

os.environ['PYTHONHASHSEED'] = str(0)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'

import numpy as np

np.random.seed(0)
import random as rn

rn.seed(0)

from PyQt5.QtWidgets import QMainWindow, QApplication, \
    QWidget, QFileDialog, \
    QLineEdit, QTableWidgetItem, QDoubleSpinBox, QCheckBox
from methods import open_data, split_data, visualize_data, pca_analysis, model \
    , categorize_ui, set_seed_val, resample, compare_dist_num, compare_dist_cat, scale_for_model \
        , macro_precision, macro_recall, random_pred
        

import concurrent.futures
import pandas as pd

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from PyQt5.uic import loadUi
import tensorflow as tf

tf.keras.utils.set_random_seed(0)

# set global variables
global var
global pheno
global new_train_data
global new_test_data
global test_pheno
global train_pheno
global test_train_selection
global original_data
global w

session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)


def set_seeds(seed_val):
    """
    Sets seed globally

    """
    np.random.seed(seed_val)
    rn.seed(seed_val)
    tf.keras.utils.set_random_seed(seed_val)
    tf.random.set_seed(seed_val)


class MainWindow(QMainWindow):

    def __init__(self):
        """
        Class to get user input from the gui main window

        """
        super(MainWindow, self).__init__()
        loadUi("gui2.ui", self)

        self.browse_button.clicked.connect(self.browse_file)
        self.PCA_button.clicked.connect(self.run_pca)
        self.var_box.activated[str].connect(self.on_selected)
        self.comboBox.activated[str].connect(self.select_strat)
        self.train_or_test_box.activated[str].connect(self.select_train_test)
        self.strat = 'mean'
        self.add_tab.clicked.connect(self.insert_stack)
        self.add_model.clicked.connect(self.insert_model)
        self.remove_tab.clicked.connect(self.remove_stack)
        self.remove_model.clicked.connect(self.del_model)
        self.insert_stack()
        self.insert_model()
        self.save.clicked.connect(self.save_data)
        self.model_button.clicked.connect(self.get_params)
        self.model_button.clicked.connect(self.run_model)
        self.confirm.clicked.connect(self.get_all)
        self.confirm.clicked.connect(self.check_train_test_dist)
        self.set_settings.clicked.connect(self.settings)
        self.remove.clicked.connect(self.remove_columns)
        self.under_method.activated[str].connect(self.undersample_method)
        self.over_method.activated[str].connect(self.oversample_method)
        self.method_undersampling = 'Nearmiss1'
        self.method_oversampling = 'ADASYN'
        self.oversample.clicked.connect(self.oversampling)
        self.undersample.clicked.connect(self.undersampling)

        # default values
        self.seed_val = 42
        self.epoch_size = 100
        self.boot_size = 32


    def closeEvent(self, event):
        QApplication.quit()
        event.accept()
    
    def browse_file(self):
        """
        Gets the directory of the file with the data

        """
        global directory
        directory, __ = QFileDialog.getOpenFileName(self, "Select File")

    def drop_down(self, data):
        """
        Creates the drop down menu to visualize the variables

        """

        data_cat, data_num, __, __, __ = split_data(data)
        variables = list(data_cat) + ['Total numbers']
        self.var_box.addItems(variables)

    def select_strat(self, txt):
        """
        Extracts the strategy for imputation chosen in the gui

        """

        selection = txt
        self.strat = selection

    def on_selected(self, txt):
        """
        Extracts input on whether the user wants to view the train or test set

        """

        global selection
        global test_train_selection
        global new_train_data
        global new_test_data
        global pheno
        global train_pheno
        global test_pheno

        selection = txt
        if test_train_selection == 'Train':
            data_cat, data_num, __, __, __ = split_data(new_train_data)
            pheno_temp = train_pheno.drop('None', axis=1)
        else:
            data_cat, data_num, __, __, __ = split_data(new_test_data)
            pheno_temp = test_pheno.drop('None', axis=1)

        visualize_data(data_cat, selection, data_num, pheno_temp)

    def settings(self):
        """
        Extracts input for bootstrap size and epoch number from gui

        """
        if len(self.bootstrap_size.text()) != 0:
            self.boot_size = int(self.bootstrap_size.text())
        if len(self.epoch_num.text()) != 0:
            self.epoch_size = int(self.epoch_num.text())

    def remove_columns(self):
        """
        Remove columns from the data based on user input

        """

        global new_train_data
        global new_test_data

        text = self.field_col.text()

        # If a column number is given as input
        if len(text) != 0:
            columns = text.split(',')
            columns_int = []

            # Concatenate the train and test data
            combined_data = pd.concat([new_train_data, new_test_data], axis=0)

            [columns_int.append(int(col) - 1) for col in columns]
            new_combined_data = combined_data.drop(combined_data.columns[columns_int], axis=1)
            new_train_data = new_combined_data.iloc[:len(new_train_data)]
            new_test_data = new_combined_data.iloc[len(new_train_data):]

    def get_params(self):
        """
        Get hyper parameters for the neural network based on user input

        """
        widgets = self.models.findChildren(QLineEdit)

        self.names = []
        self.layer_info = []

        for widget in widgets:
            parent = widget.parentWidget()
            if 'Model' in parent.objectName():
                self.layer_info.append(widget.text())
                self.names.append(parent.objectName())

        names_int = []
        [names_int.append(int(name[6:])) for name in self.names]

        widgets = self.models.findChildren(QDoubleSpinBox)
        self.drop_info = []
        for widget in widgets:
            self.drop_info.append("%.2f" % widget.value())

        widgets = self.models.findChildren(QCheckBox)

        for i, widget in enumerate(widgets):
            if not widget.isChecked():
                self.drop_info[i] = 0

        self.layer_info = [x for _, x in sorted(zip(names_int, self.layer_info))]
        self.drop_info = [x for _, x in sorted(zip(names_int, self.drop_info))]
        self.names = sorted(self.names, key=lambda x: int(x[6:]))

    def get_all(self):

        """
        Obtains the data from the file in question. 
        Splits the data and the classes and numerical from categorical variables

        """

        global new_train_data
        global new_test_data
        global pheno
        global test_pheno
        global train_pheno
        global original_data

        if len(self.seed_value_set.text()) != 0:
            self.seed_val = int(self.seed_value_set.text())
            print(self.seed_val)

        set_seeds(self.seed_val)
        set_seed_val(self.seed_val)

        raw_data = open_data(directory)

        text = self.pheno_text.text()

        # Select classes/phenotypes based on input text
        # If no text 'mental problems' is default column
        if len(text) == 0:
            columns = ['Mental problems']
        else:
            columns = text.split(',')
        pheno = raw_data[columns]

        pheno = pheno.replace(to_replace=['no', 'yes'], value=[0, 1])
        if set(pheno) != {'0', '1'}:
            pheno = pd.get_dummies(pheno)

        data_no_pheno = raw_data.drop(columns, axis=1)

        # healthy contains all the phenotypes that are not specified
        healthy = []
        [healthy.append(1) if sum(subj) == 0 else healthy.append(0) for subj in np.array(pheno)]

        if len(healthy) != 0:
            pheno['None'] = healthy

        if len(self.test_size.text()) != 0:
            test_size = float(self.test_size.text())
        else:
            test_size = 1 / 4

        # split data in train and test set
        new_train_data, new_test_data, train_pheno, test_pheno = train_test_split(data_no_pheno, pheno,
                                                                                  test_size=test_size,
                                                                                  random_state=self.seed_val,
                                                                                  stratify=pheno)

        # save original data
        original_data = pd.concat([pd.concat([new_train_data, new_test_data], axis=0), \
                                   pd.concat([train_pheno, test_pheno], axis=0)], axis=1)
        original_data = original_data.loc[data_no_pheno.index]

        # impute the data but dont split yet
        __, __, new_train_data, imputer, imputer_cat = split_data(new_train_data, strat=self.strat)
        __, __, new_test_data, __, __ = split_data(new_test_data, strat=self.strat, test=True, imputer=imputer,
                                                   imputer_cat=imputer_cat)

        # self.drop_down(data_no_pheno)

    def check_train_test_dist(self):

        """
        Prepare data for the checking of the distributions between 2 sets
        for both categorical and numerical
        """
        global new_train_data
        global new_test_data

        data_cat_train, data_num_train, __, imputer, imputer_cat = split_data(new_train_data, strat=self.strat)
        data_cat_test, data_num_test, __, __, __ = split_data(new_test_data, strat=self.strat, test=True,
                                                              imputer=imputer, imputer_cat=imputer_cat)

        # Concatenate the train and test data
        combined_data = pd.concat([data_cat_train, data_cat_test], axis=0)

        # Dummify the combined data
        dummified_data = pd.get_dummies(combined_data)

        # Split the dummified data back into train and test
        dummified_train = dummified_data.iloc[:len(data_cat_train)]
        dummified_test = dummified_data.iloc[len(data_cat_train):]

        __, data_num_train, __, data_num_test = \
            scale_for_model(dummified_train, data_num_train, dummified_test, data_num_test)

        # check the distribution between test and training set
        if not data_num_train.empty:
            compare_dist_num(data_num_train, data_num_test)
        if not dummified_train.empty:
            compare_dist_cat(dummified_train, dummified_test)

    def select_train_test(self, txt):
        """
        Extracts user input on whether user wants to see test or train data
        """

        global new_train_data
        global new_test_data
        global train_pheno
        global test_pheno
        global test_train_selection
        test_train_selection = txt
        if test_train_selection == 'Train':
            self.drop_down(new_train_data)
        else:
            self.drop_down(new_test_data)

    def run_pca(self):

        """
        Prepare data for excecution of PCA analysis
        """

        global new_train_data
        global new_test_data
        global pheno
        global train_pheno
        global test_pheno

        pheno_train_temp = train_pheno.drop('None', axis=1)
        pheno_test_temp = test_pheno.drop('None', axis=1)

        data_cat_train, data_num_train, __, __, __ = split_data(new_train_data)
        data_cat_test, data_num_test, __, __, __ = split_data(new_test_data)
        try:
            if len(self.PCAcomponent1.text()) != 0 and len(self.PCAcomponent2.text()) != 0:
                x_comp = int(self.PCAcomponent1.text())
                y_comp = int(self.PCAcomponent2.text())
            else:
                x_comp = 0
                y_comp = 1
        except:
            print('Only integer values')

        # Concatenate the train and test data
        combined_data = pd.concat([data_cat_train, data_cat_test], axis=0)

        # Dummify the combined data
        dummified_data = pd.get_dummies(combined_data)

        # Split the dummified data back into train and test
        dummified_train = dummified_data.iloc[:len(data_cat_train)]
        dummified_test = dummified_data.iloc[len(data_cat_train):]

        pca_analysis(dummified_train, data_num_train, dummified_test, data_num_test, pheno_train_temp, pheno_test_temp,
                     x_comp, y_comp)

    def undersample_method(self, txt):
        """
        Get user input on which undersampling method to use
        """
        self.method_undersampling = txt

    def undersampling(self):
        """
        Prepare train data for undersampling
        """
        global new_train_data
        global new_test_data
        global train_pheno

        if len(self.neighbour_count.text()) != 0:
            neighbours = int(self.neighbour_count.text())
        else:
            neighbours = 1

        if len(self.s_strat.text()) != 0:
            strat = float(self.s_strat.text())
        else:
            strat = 1

        # Obtain indices from new_test_data
        length_data = len(new_train_data) + len(new_test_data) 
        
        selection = self.method_undersampling
        cat, num, __, __, __ = split_data(new_train_data)
        set_seeds(self.seed_val)
        set_seed_val(self.seed_val)
        new_train_data, train_pheno = resample(cat, num, train_pheno, length_data, True, False, selection, neighbours=neighbours,
                                               seed=self.seed_val,
                                               sampling_strat=strat)

    def oversample_method(self, txt):
        """
        Get user input on which undersampling method to use
        """
        self.method_oversampling = txt

    def oversampling(self):
        """
        Prepare train data for oversampling
        """
        global new_train_data
        global new_test_data
        global train_pheno
        
        length_data = len(new_train_data) + len(new_test_data) 
        selection = self.method_oversampling
        cat, num, __, __, __ = split_data(new_train_data)
        set_seeds(self.seed_val)
        set_seed_val(self.seed_val)
        new_train_data, train_pheno = resample(cat, num, train_pheno, length_data, False, True, selection, neighbours=1,
                                               seed=self.seed_val)   
    def run_model(self):
        """
        Runs the model(s) using the user input parameters.
        When multiple models are given it uses paralel threading to run
        multiple models at the same time.


        """

        global new_train_data
        global new_test_data
        global train_pheno
        global test_pheno
        global var

        set_seeds(self.seed_val)
        set_seed_val(self.seed_val)

        data_cat_train, data_num_train, __, __, __ = split_data(new_train_data)
        data_cat_test, data_num_test, __, __, __ = split_data(new_test_data)

        # Check whether data contains categorial and numerical values
        cat = False
        num = False

        if len(data_cat_train) != 0:
            cat = True

        if len(data_num_train) != 0:
            num = True

        # Concatenate the train and test data
        combined_data = pd.concat([data_cat_train, data_cat_test], axis=0)

        # Dummify the combined data
        dummified_data = pd.get_dummies(combined_data)

        # Split the dummified data back into train and test
        dummified_train = dummified_data.iloc[:len(data_cat_train)]
        dummified_test = dummified_data.iloc[len(data_cat_train):]

        dummified_train, data_num_train, dummified_test, data_num_test = \
            scale_for_model(dummified_train, data_num_train, dummified_test, data_num_test)

        # create a list containing all the models
        set_seeds(self.seed_val)
        set_seed_val(self.seed_val)
        models = []
        [models.append(model(dummified_train, data_num_train, self.layer_info[i], self.drop_info[i], cat=cat, num=num,
                             seed=self.seed_val)) for i
         in range(len(self.layer_info))]
        # run every model in the list using paralel threading
        histories = []

        with concurrent.futures.ThreadPoolExecutor() as ex:
            set_seeds(self.seed_val)
            set_seed_val(self.seed_val)
            if cat == True and num == True:

                results = [ex.submit(models[i].fit, [dummified_train, data_num_train], train_pheno,
                                     validation_data=([dummified_test, data_num_test], test_pheno),
                                     epochs=self.epoch_size,
                                     verbose=1, batch_size=self.boot_size, shuffle=True) for i in
                           range(len(self.layer_info))]
            elif cat == True:

                results = [
                    ex.submit(models[i].fit, dummified_train, train_pheno,
                              validation_data=(dummified_test, test_pheno),
                              epochs=self.epoch_size, verbose=1, batch_size=self.boot_size, shuffle=True) for i in
                    range(len(self.layer_info))]
            elif num == True:

                results = [ex.submit(models[i].fit, data_num_train, train_pheno,
                                     validation_data=(data_num_test, test_pheno), epochs=self.epoch_size, verbose=1,
                                     batch_size=self.boot_size, shuffle=True) for i in range(len(self.layer_info))]
            for f in results:
                histories.append(f.result().history)
                

        # create the ensemble model by combining fit results of all seperate models
        model_input_num = tf.keras.Input(len(list(data_num_train)), name='ensemble_num')
        model_input_cat = tf.keras.Input(len(list(dummified_train)), name='ensemble_cat')

        if cat == True and num == True:
            model_outputs = [mod([model_input_cat, model_input_num]) for mod in models]
        elif cat == True:
            model_outputs = [mod(model_input_cat) for mod in models]
        elif num == True:
            model_outputs = [mod(model_input_num) for mod in models]

        ensemble_output = tf.keras.layers.Average()(model_outputs)

        if cat == True and num == True:
            ensemble_model = tf.keras.Model(inputs=[model_input_cat, model_input_num], outputs=ensemble_output)
        elif cat == True:
            ensemble_model = tf.keras.Model(inputs=model_input_cat, outputs=ensemble_output)
        elif num == True:
            ensemble_model = tf.keras.Model(inputs=model_input_num, outputs=ensemble_output)

        # compile the ensemble model
        ensemble_model.compile(loss='categorical_crossentropy', metrics=["categorical_accuracy", macro_precision, macro_recall])

        var = list(dummified_test) + list(data_num_test)
        results_perm = []

        # run the ensemble model on the test data
        if cat == True and num == True:
            ensemble_result = (ensemble_model.evaluate([dummified_test, data_num_test], test_pheno, return_dict=True))
        elif cat == True:
            ensemble_result = (ensemble_model.evaluate(dummified_test, test_pheno, return_dict=True))
        elif num == True:
            ensemble_result = (ensemble_model.evaluate(data_num_test, test_pheno, return_dict=True))

        # use permutation of data in columns and the change in loss value resulting
        # to get a measure of variable importance in the model
        set_seeds(self.seed_val)
        set_seed_val(self.seed_val)
        [results_perm.append(ensemble_model.evaluate(self.shuffle_col(dummified_test, data_num_test, name), test_pheno))
         for
         i, name in enumerate(var)]

        find_features(ensemble_result['loss'], results_perm, var)

        # display the results of all the models
        print('Result is: \n')

        print(ensemble_result)
        
        # Get precision, recall and accuracy based on a random predictor
        random_acc, random_recall, random_prec = random_pred(test_pheno)

        for hist in histories:
            
            fig, axs = plt.subplots(3, figsize=(8, 6))  # Create a 4-subplot figure

            # accuracy
            axs[0].plot(hist['categorical_accuracy'], label='train accuracy')
            axs[0].plot(hist['val_categorical_accuracy'], label='validation accuracy')
            axs[0].axhline(random_acc, color='black', linestyle='--', label='random accuracy test')
            axs[0].set_xlabel('epoch number')
            axs[0].set_ylabel('Accuracy')
            axs[0].legend(loc="upper right")
    
            # macro precision
            axs[1].plot(hist['macro_precision'], label='train macro precision')  
            axs[1].plot(hist['val_macro_precision'], label='validation macro precision') 
            axs[1].axhline(random_prec, color='black', linestyle='--', label='random precision test')
            axs[1].set_xlabel('epoch number')
            axs[1].set_ylabel('Macro Precision')  # <-- Updated here
            axs[1].legend(loc="upper right")
    
            # macro recall
            axs[2].plot(hist['macro_recall'], label='train macro recall')  
            axs[2].plot(hist['val_macro_recall'], label='validation macro recall')  
            axs[2].axhline(random_recall, color='black', linestyle='--', label='random recall test')
            axs[2].set_xlabel('epoch number')
            axs[2].set_ylabel('Macro Recall') 
            axs[2].legend(loc="upper right")

            plt.tight_layout()
            plt.show()


    def shuffle_col(self, df, df2, name):
        """
        Shuffles the data in a column of the dataframe

        Parameters
        ----------
        df: pandas DataFrame
            Dataframe containng categorial data
        df2: pandas DataFrame
            Dataframe containing numerical data
        name: string
            Name of the column that is to be shuffled

        Returns
        -------
        df_shuffle: pandas DataFrame
            Dataframe containing the original data with one column shuffled
            
        df: pandas DataFrame
            Original categorial data
            
        df2: pandas DataFrame
            Original numerical data

        """
        if name in list(df):

            df_shuffle = df
            df_shuffle[name] = np.random.permutation(df[name].values)

            if len(df2) != 0:
                return df_shuffle, df2
            else:
                return df_shuffle

        else:

            df_shuffle = df2
            df_shuffle[name] = np.random.permutation(df2[name].values)

            if len(df) != 0:
                return df, df_shuffle
            else:
                return df_shuffle

    def stack_change(self):
        """
        Change the active stack in the gui

        """
        self.stack_index = self.stacks.currentIndex()

    def insert_stack(self):
        """
        Insert new stack in the gui

        """
        self.stacks.addTab(stack_instance(), 'new tab')

    def remove_stack(self):
        """
        Remove active stack in the gui

        """
        self.stacks.removeTab(self.stacks.currentIndex())

    def update_stack(self, name):
        """
        Update stack in the gui

        """
        self.stacks.setTabText(self.stacks.currentIndex(), name)

    def insert_model(self):
        """
        insert a new model tab in the gui this will increase allow a user to 
        set up a new model to run in parallel

        """
        number = self.models.count() + 1
        name = 'Model_{}'.format(number)
        self.models.addTab(model_instance(name), name)

    def del_model(self):
        """
        Delete a model tab and thus also its instance

        """

        widgets = self.models.findChildren(QWidget)
        for widget in widgets:
            parent = widget.parentWidget()
            if str(self.models.currentIndex() + 1) in parent.objectName():
                widget.setParent(None)

        self.models.removeTab(self.models.currentIndex())

    def save_data(self):
        """
        Save the data to an .csv file

        """
        global new_train_data
        global new_test_data

        combined_data = pd.concat([new_train_data, new_test_data], axis=0)

        combined_data.to_csv(directory.replace('.xls', '_edited.csv'), index=False)


class table_view(QWidget):
    def __init__(self, orig=False):
        """
        Generate a table widget to display the input data in a table format

        """
        super(QWidget, self).__init__()
        loadUi("table.ui", self)

        self.load_data(orig)

    def load_data(self, orig=False):
        """
        Load the data from a .csv file

        """
        global new_train_data
        global new_test_data
        global directory
        global original_data

        if orig == False:
            data = pd.concat([pd.concat([new_train_data, new_test_data], axis=0), \
                              pd.concat([train_pheno, test_pheno], axis=0)], axis=1)

            common_indices = data.index.intersection(original_data.index)
            other_indices = data.index.difference(original_data.index).sort_values()
            common_data = data.loc[common_indices]
            other_data = data.loc[other_indices]
            data = pd.concat([common_data, other_data])
        else:
            data = original_data

        for j, col in enumerate(list(data)):
            self.data_table.setItem(0, j, QTableWidgetItem(str(list(data)[j])))
            for i in range(data.shape[0]):
                self.data_table.setItem(i + 1, j, QTableWidgetItem(str(data[col].iloc[i])))


class stack_instance(QWidget):

    def __init__(self):
        """
        Create new stacks for the categorize tabs where user input can be
        read from
        """
        super().__init__()
        loadUi("stack.ui", self)
        self.confirm.clicked.connect(self.make_categories)
        self.show_table.clicked.connect(self.display_table)
        self.show_orig.clicked.connect(self.display_orig)
        self.filter_check.stateChanged.connect(self.clicked_box)
        self.status = self.filter_check.isChecked()
        self.filter_box.activated[str].connect(self.on_selected)
    
           
    def make_categories(self):
        """
        Prepare data to categorize the selected column based on user inputs
        """

        global new_train_data
        global new_test_data
        global w

        text_col = self.field_col.text().strip()

        text_col = text_col.split(',')
        self.text_to_int_col = []
        [self.text_to_int_col.append(int(index) - 1) for index in text_col]

        categories = new_train_data.columns[self.text_to_int_col]
        cat_in_string = ', '.join(categories)
        w.update_stack(cat_in_string)

        print(categories)
        text_name = self.field_name.text()

        if self.larger.text():
            lower_bound = int(self.larger.text())
        else:
            lower_bound = float('-inf')

        if self.smaller.text():
            upper_bound = int(self.smaller.text())
        else:
            upper_bound = float('inf')

        # Concatenate the train and test data
        combined_data = pd.concat([pd.concat([new_train_data, new_test_data], axis=0), \
                                   pd.concat([train_pheno, test_pheno], axis=0)], axis=1)

        if self.status == True:
            new_combined_data = categorize_ui(combined_data, categories, lower_bound, upper_bound, text_name,
                                              self.filter_cat)
        else:
            new_combined_data = categorize_ui(combined_data, categories, lower_bound, upper_bound, text_name)

        # split data in train and test again
        new_train_data = new_combined_data.iloc[:len(new_train_data)]
        new_test_data = new_combined_data.iloc[len(new_train_data):]

    def interact(self):
        """
        I honestly don't know anymore it does not seem to do anything but I dont
        want to try and find out
        """
        print('yes')

    def display_table(self):
        """
        Displays the table with updated data
        """

        self.sw = table_view()
        self.sw.show()

    def display_orig(self):
        """
        Displays the table with the original data
        """

        self.swo = table_view(orig=True)
        self.swo.show()

    def clicked_box(self):
        """
        When checkbox is clicked this updates the drop down menu
        """

        global new_train_data
        self.status = self.filter_check.isChecked()
        if self.status == True:
            data_cat, data_num, __, __, __ = split_data(new_train_data)
            dummies = pd.get_dummies(data_cat)
            variables = list(dummies)
            self.filter_box.addItems(variables)

    def on_selected(self, txt):
        """
        Stores the selection made on the filter requirement for later use
        """
        self.filter_cat = txt


class model_instance(QWidget):
    def __init__(self, name):
        """
        Create new model tab in the GUI where user input can be read from

        """
        super().__init__()
        loadUi("model_tab.ui", self)
        self.setObjectName(name)

    def read_layers(self):
        self.input_layers = self.layers.text()
        print(self.input_layers)


def aggregate_dummy_importance(loss_changes, feature_names, prefix_sep="_"):
    """
    Aggregate the loss changes for dummy variables into their original categorical variables.

    Parameters
    ----------
    loss_changes: list floats
        Loss changes (from the original loss) for each feature.
    feature_names: list of strings
        Names of the features.

    Returns
    ----------
    - Dictionary of floats
        Aggregated loss changes for each original feature.
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

def start_gui():
    global w
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())

