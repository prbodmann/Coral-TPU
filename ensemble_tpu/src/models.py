from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dropout, Activation, Average, Flatten, AveragePooling2D,Input, Reshape
from keras.models import Model, Sequential
from tensorflow.python.framework.ops import Tensor
from keras.engine import training
import keras
from typing import List
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
import os
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder #LabelBinarizer
from numpy.core.umath_tests import inner1d
from src.utils import tflite_converter, create_interpreter, get_scores, set_interpreter_input
import pickle

cce = keras.losses.CategoricalCrossentropy()

CONV_POOL_CNN_WEIGHT_FILE = os.path.join(os.getcwd(), 'weights', 'conv_pool_cnn_pretrained_weights.hdf5')
ALL_CNN_WEIGHT_FILE = os.path.join(os.getcwd(), 'weights', 'all_cnn_pretrained_weights.hdf5')
NIN_CNN_WEIGHT_FILE = os.path.join(os.getcwd(), 'weights', 'nin_cnn_pretrained_weights.hdf5')

def conv_pool_cnn(model_input: Tensor) -> training.Model:

    x = Conv2D(96, kernel_size=(3, 3), activation='relu', padding = 'same')(model_input)
    x = Conv2D(96, (3, 3), activation='relu', padding = 'same')(x)
    x = Conv2D(96, (3, 3), activation='relu', padding = 'same')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides = 2)(x)
    x = Conv2D(192, (3, 3), activation='relu', padding = 'same')(x)
    x = Conv2D(192, (3, 3), activation='relu', padding = 'same')(x)
    x = Conv2D(192, (3, 3), activation='relu', padding = 'same')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides = 2)(x)
    x = Conv2D(192, (3, 3), activation='relu', padding = 'same')(x)
    x = Conv2D(192, (1, 1), activation='relu')(x)
    x = Conv2D(10, (1, 1))(x)
    x = AveragePooling2D(pool_size=(7,7))(x)
    x = Activation(activation='softmax')(x)
    x = Flatten()(x)
    model = Model(model_input, x, name='conv_pool_cnn')
    model.compile(loss=cce, optimizer="adam")
    return model


def all_cnn(model_input: Tensor) -> training.Model:

    x = Conv2D(96, kernel_size=(3, 3), activation='relu', padding = 'same')(model_input)
    x = Conv2D(96, (3, 3), activation='relu', padding = 'same')(x)
    x = Conv2D(96, (3, 3), activation='relu', padding = 'same', strides = 2)(x)
    x = Conv2D(192, (3, 3), activation='relu', padding = 'same')(x)
    x = Conv2D(192, (3, 3), activation='relu', padding = 'same')(x)
    x = Conv2D(192, (3, 3), activation='relu', padding = 'same', strides = 2)(x)
    x = Conv2D(192, (3, 3), activation='relu', padding = 'same')(x)
    x = Conv2D(192, (1, 1), activation='relu')(x)
    x = Conv2D(10, (1, 1))(x)
    x = AveragePooling2D(pool_size=(8,8))(x)
    x = Activation(activation='softmax')(x)
    x = Flatten()(x)
    model = Model(model_input, x, name='all_cnn')
    model.compile(loss=cce, optimizer="adam")
    return model

def nin_cnn(model_input: Tensor) -> training.Model:

    #mlpconv block 1
    x = Conv2D(32, (5, 5), activation='relu',padding='valid')(model_input)
    x = Conv2D(32, (1, 1), activation='relu')(x)
    x = Conv2D(32, (1, 1), activation='relu')(x)
    x = MaxPooling2D((2,2))(x)
    x = Dropout(0.5)(x)

    #mlpconv block2
    x = Conv2D(64, (3, 3), activation='relu',padding='valid')(x)
    x = Conv2D(64, (1, 1), activation='relu')(x)
    x = Conv2D(64, (1, 1), activation='relu')(x)
    x = MaxPooling2D((2,2))(x)
    x = Dropout(0.5)(x)

    #mlpconv block3
    x = Conv2D(128, (3, 3), activation='relu',padding='valid')(x)
    x = Conv2D(32, (1, 1), activation='relu')(x)
    x = Conv2D(10, (1, 1))(x)

    x = AveragePooling2D(pool_size=(4,4))(x)
    x = Activation(activation='softmax')(x)
    x = Flatten()(x)
    model = Model(model_input, x, name='nin_cnn')
    model.compile(loss=cce, optimizer="adam")
    return model

def ensemble(models: List [training.Model], model_input: Tensor) -> training.Model:

    outputs = [model.outputs[0] for model in models]
    y = Average()(outputs)

    model = Model(model_input, y, name='ensemble')
    model.compile(loss=cce, optimizer="adam")
    return model

def conv_all(models: List [training.Model], model_input: Tensor) -> training.Model:

    outputs = [model.outputs[0] for model in models]
    y = Average()(outputs)

    model = Model(model_input, y, name='ensemble')
    model.compile(loss=cce, optimizer="adam")
    return model
def conv_nin(models: List [training.Model], model_input: Tensor) -> training.Model:

    outputs = [model.outputs[0] for model in models]
    y = Average()(outputs)

    model = Model(model_input, y, name='ensemble')
    model.compile(loss=cce, optimizer="adam")
    return model
def all_nin(models: List [training.Model], model_input: Tensor) -> training.Model:

    outputs = [model.outputs[0] for model in models]
    y = Average()(outputs)

    model = Model(model_input, y, name='ensemble')
    model.compile(loss=cce, optimizer="adam")
    return model

class AdaBoostClassifier(object):
    '''
    Parameters
    -----------
    base_estimator: object
        The base model from which the boosted ensemble is built.
    n_estimators: integer, optional(default=50)
        The maximum number of estimators
    learning_rate: float, optional(default=1)
    algorithm: {'SAMME','SAMME.R'}, optional(default='SAMME.R')
        SAMME.R uses predicted probabilities to update wights, while SAMME uses class error rate
    random_state: int or None, optional(default=None)
    Attributes
    -------------
    estimators_: list of base estimators
    estimator_weights_: array of floats
        Weights for each base_estimator
    estimator_errors_: array of floats
        Classification error for each estimator in the boosted ensemble.
    Reference:
    1. [multi-adaboost](https://web.stanford.edu/~hastie/Papers/samme.pdf)
    2. [scikit-learn:weight_boosting](https://github.com/scikit-learn/
    scikit-learn/blob/51a765a/sklearn/ensemble/weight_boosting.py#L289)
    '''

    def __init__(self, *args, **kwargs):
        if kwargs and args:
            raise ValueError(
                '''AdaBoostClassifier can only be called with keyword
                   arguments for the following keywords: base_estimator ,n_estimators,
                    learning_rate,algorithm,random_state''')
        allowed_keys = ['base_estimator', 'n_estimators', 'learning_rate', 'algorithm', 'random_state', 'epochs']
        keywords_used = kwargs.keys()
        for keyword in keywords_used:
            if keyword not in allowed_keys:
                raise ValueError(keyword + ":  Wrong keyword used --- check spelling")

        n_estimators = 50
        learning_rate = 1
        algorithm = 'SAMME.R'
        random_state = None
        #### CNN (5)
        epochs = 20

        if kwargs and not args:
            if 'base_estimator' in kwargs:
                base_estimator = kwargs.pop('base_estimator')
            else:
                raise ValueError('''base_estimator can not be None''')
            if 'n_estimators' in kwargs: n_estimators = kwargs.pop('n_estimators')
            if 'learning_rate' in kwargs: learning_rate = kwargs.pop('learning_rate')
            if 'algorithm' in kwargs: algorithm = kwargs.pop('algorithm')
            if 'random_state' in kwargs: random_state = kwargs.pop('random_state')
            ### CNN:
            if 'epochs' in kwargs: epochs = kwargs.pop('epochs')


        self.base_estimator_ = base_estimator
        self.n_estimators_ = n_estimators
        self.learning_rate_ = learning_rate
        self.algorithm_ = algorithm
        self.random_state_ = random_state
        self.estimators_ = list()
        self.tpu_estimators_ = list()
        self.estimator_weights_ = np.zeros(self.n_estimators_)
        self.estimator_errors_ = np.ones(self.n_estimators_)
        self.n_classes_=10
        self.epochs= epochs


    def _samme_proba(self, estimator, n_classes, X):
        """Calculate algorithm 4, step 2, equation c) of Zhu et al [1].

        References
        ----------
        .. [1] J. Zhu, H. Zou, S. Rosset, T. Hastie, "Multi-class AdaBoost", 2009.

        """
        proba = estimator.predict(X)

        # Displace zero probabilities so the log is defined.
        # Also fix negative elements which may occur with
        # negative sample weights.
        proba[proba < np.finfo(proba.dtype).eps] = np.finfo(proba.dtype).eps
        log_proba = np.log(proba)

        return (n_classes - 1) * (log_proba - (1. / n_classes)
                                  * log_proba.sum(axis=1)[:, np.newaxis])


    def fit(self, X, y, batch_size):

        ## CNN:
        self.batch_size = batch_size

#        self.epochs = epochs
        self.n_samples = X.shape[0]
        # There is hidden trouble for classes, here the classes will be sorted.
        # So in boost we have to ensure that the predict results have the same classes sort

        self.classes_ = np.zeros(10)

        ############for CNN (2):
#        yl = np.argmax(y)
#        self.classes_ = np.array(sorted(list(set(yl))))

        self.n_classes_ = len(self.classes_)
        for iboost in range(self.n_estimators_):
            if iboost == 0:
                sample_weight = np.ones(self.n_samples) / self.n_samples

            sample_weight, estimator_weight, estimator_error = self.boost(X, y, sample_weight)

            # early stop
            if estimator_error == None:
                break

            # append error and weight
            self.estimator_errors_[iboost] = estimator_error
            self.estimator_weights_[iboost] = estimator_weight

            if estimator_error <= 0:
                break

        return self


    def boost(self, X, y, sample_weight):
        if self.algorithm_ == 'SAMME':
            return self.discrete_boost(X, y, sample_weight)
        elif self.algorithm_ == 'SAMME.R':
            return self.real_boost(X, y, sample_weight)


    def real_boost(self, X, y, sample_weight):
        #            estimator = deepcopy(self.base_estimator_)
        ############################################### my code:

        if len(self.estimators_) == 0:
            #Copy CNN to estimator:
            estimator = self.deepcopy_CNN(self.base_estimator_)#deepcopy of self.base_estimator_
        else:
            #estimator = deepcopy(self.estimators_[-1])
            estimator = self.deepcopy_CNN(self.estimators_[-1])#deepcopy CNN
    ###################################################
        if self.random_state_:
                estimator.set_params(random_state=1)
#        estimator.fit(X, y, sample_weight=sample_weight)
 #################################### CNN (3) binery label:
        # lb=LabelBinarizer()
        # y_b = lb.fit_transform(y)

        #lb=OneHotEncoder(sparse=False)
        #y_b=y.reshape(len(y),1)
        y_b=y#lb.fit_transform(y_b)
        print(X.shape)
        print(y_b.shape)
        estimator.fit(X, y_b, sample_weight=sample_weight, epochs = self.epochs, batch_size = self.batch_size)
############################################################
        y_pred = estimator.predict(X)
        ############################################ (4) CNN :
        y_pred_l = np.argmax(y_pred, axis=1)
        incorrect = y_pred_l !=  np.argmax(y, axis=1)
#########################################################
        estimator_error = np.dot(incorrect, sample_weight) / np.sum(sample_weight, axis=0)
        print(estimator_error.shape)
        # if worse than random guess, stop boosting
        if estimator_error >= 1.0 - 1 / self.n_classes_:
            return None, None, None

        y_predict_proba = estimator.predict(X)

        # repalce zero
        y_predict_proba[y_predict_proba < np.finfo(y_predict_proba.dtype).eps] = np.finfo(y_predict_proba.dtype).eps

        y_codes = np.array([-1. / (self.n_classes_ - 1), 1.])
        print(y_codes)
        y_coding = y_codes.take(self.classes_ == y)
        print(y_coding.shape)
        # for sample weight update
        intermediate_variable = (-1. * self.learning_rate_ * (((self.n_classes_ - 1) / self.n_classes_) *
                                                              inner1d(y_coding, np.log(
                                                                  y_predict_proba))))  #dot iterate for each row

        # update sample weight
        sample_weight *= np.exp(intermediate_variable)

        sample_weight_sum = np.sum(sample_weight, axis=0)
        if sample_weight_sum <= 0:
            return None, None, None

        # normalize sample weight
        sample_weight /= sample_weight_sum

        # append the estimator
        self.estimators_.append(estimator)

        return sample_weight, 1, estimator_error

    def deepcopy_CNN(self, base_estimator0):
        #Copy CNN (self.base_estimator_) to estimator:
        config=base_estimator0.get_config()
        #estimator = Models.model_from_config(config)
        estimator = Sequential.from_config(config)


        weights = base_estimator0.get_weights()
        estimator.set_weights(weights)
        estimator.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        return estimator

    def discrete_boost(self, X, y, sample_weight):
#        estimator = deepcopy(self.base_estimator_)
         ############################################### my code:

        if len(self.estimators_) == 0:
            #Copy CNN to estimator:
            estimator = self.deepcopy_CNN(self.base_estimator_)#deepcopy of self.base_estimator_
        else:
            #estimator = deepcopy(self.estimators_[-1])
            estimator = self.deepcopy_CNN(self.estimators_[-1])#deepcopy CNN
    ###################################################

        if self.random_state_:
            estimator.set_params(random_state=1)
#        estimator.fit(X, y, sample_weight=sample_weight)
#################################### CNN (3) binery label:
        # lb=LabelBinarizer()
        # y_b = lb.fit_transform(y)

        lb=OneHotEncoder(sparse=False)
        y_b=y.reshape(len(y),1)
        y_b=lb.fit_transform(y_b)

        estimator.fit(X, y_b, sample_weight=sample_weight, epochs = self.epochs, batch_size = self.batch_size)
############################################################
        y_pred = estimator.predict(X)

        #incorrect = y_pred != y
 ############################################ (4) CNN :
        y_pred_l = np.argmax(y_pred, axis=1)
        incorrect = y_pred_l != y
#######################################################
        estimator_error = np.dot(incorrect, sample_weight) / np.sum(sample_weight, axis=0)

        # if worse than random guess, stop boosting
        if estimator_error >= 1 - 1 / self.n_classes_:
            return None, None, None

        # update estimator_weight
#        estimator_weight = self.learning_rate_ * np.log((1 - estimator_error) / estimator_error) + np.log(
#            self.n_classes_ - 1)
        estimator_weight = self.learning_rate_ * (np.log((1. - estimator_error) / estimator_error) + np.log(self.n_classes_ - 1.))

        if estimator_weight <= 0:
            return None, None, None

        # update sample weight
        sample_weight *= np.exp(estimator_weight * incorrect)

        sample_weight_sum = np.sum(sample_weight, axis=0)
        if sample_weight_sum <= 0:
            return None, None, None

        # normalize sample weight
        sample_weight /= sample_weight_sum

        # append the estimator
        self.estimators_.append(estimator)

        return sample_weight, estimator_weight, estimator_error

    def predict(self, X):
        n_classes = self.n_classes_
        classes = self.classes_[:, np.newaxis]
        pred = None

        if self.algorithm_ == 'SAMME.R':
            # The weights are all 1. for SAMME.R
            pred = sum(self._samme_proba(estimator, n_classes, X) for estimator in self.estimators_)
        else:  # self.algorithm == "SAMME"
#            pred = sum((estimator.predict(X) == classes).T * w
#                       for estimator, w in zip(self.estimators_,
#                                               self.estimator_weights_))
########################################CNN disc
            pred = sum((estimator.predict(X).argmax(axis=1) == classes).T * w
                       for estimator, w in zip(self.estimators_,
                                               self.estimator_weights_))
###########################################################
        pred /= self.estimator_weights_.sum()
        if n_classes == 2:
            pred[:, 0] *= -1
            pred = pred.sum(axis=1)
            return self.classes_.take(pred > 0, axis=0)

        return self.classes_.take(np.argmax(pred, axis=1), axis=0)


    def predict_proba(self, X):
        n_classes = self.n_classes_
        if self.algorithm_ == 'SAMME.R':
            # The weights are all 1. for SAMME.R
            proba = sum(self._samme_proba(estimator, self.n_classes_, X)
                        for estimator in self.estimators_)
        else:  # self.algorithm == "SAMME"
            proba = sum(estimator.predict_proba(X) * w
                        for estimator, w in zip(self.estimators_,
                                                self.estimator_weights_))

        proba /= self.estimator_weights_.sum()
        proba = np.exp((1. / (n_classes - 1)) * proba)
        normalizer = proba.sum(axis=1)[:, np.newaxis]
        normalizer[normalizer == 0.0] = 1.0
        proba /= normalizer

        return proba
    
    def _samme_proba_tpu(self, n_classes, X):
        """Calculate algorithm 4, step 2, equation c) of Zhu et al [1].
        References
        ----------
        .. [1] J. Zhu, H. Zou, S. Rosset, T. Hastie, "Multi-class AdaBoost", 2009.
        """
        proba=[]
        for img in X:
            for estimator in self.tpu_estimators_:
                set_interpreter_input(estimator, img)
                perform_inference(estimator)
                proba.append(get_scores(estimator))

        # Displace zero probabilities so the log is defined.
        # Also fix negative elements which may occur with
        # negative sample weights.
        proba[proba < np.finfo(proba.dtype).eps] = np.finfo(proba.dtype).eps
        log_proba = np.log(proba)

        return (n_classes - 1) * (log_proba - (1. / n_classes)
                                * log_proba.sum(axis=1)[:, np.newaxis])
    def predict_proba_tpu(self, X):
        if self.algorithm_ == 'SAMME.R':
            # The weights are all 1. for SAMME.R
            proba = sum(self._samme_proba_tpu(self.n_classes_, X)
                        for estimator in self.tpu_estimators_)
        else:  # self.algorithm == "SAMME"
            raise NotImplementedError

        proba /= self.estimator_weights_.sum()
        proba = np.exp((1. / (10 - 1)) * proba)
        normalizer = proba.sum(axis=0)[:, np.newaxis]
        normalizer[normalizer == 0.0] = 1.0
        proba /= normalizer

        return proba
    def convert_tflite(self,x,model_name):

        for idx,i in enumerate(self.estimators_):
            tflite_converter(i,x,f"{model_name}_{idx}.tflite")
        with open(f"{model_name}_weights.npy",'wb') as fd:
            pickle.dump(self.estimator_weights_,fd)
        

    def load_tflite_model(self,model_name):
        for i in range(self.n_estimators_):
            self.tpu_estimators_.append(create_interpreter(model_name+"_"+str(i)+"_edgetpu.tflite"))
        with open(f"{model_name}_weights.npy",'rb') as fd:
            self.estimator_weights_=pickle.load(fd)


