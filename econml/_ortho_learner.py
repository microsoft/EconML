# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""

Orthogonal Machine Learning is a general approach to estimating causal models
by formulating them as minimizers of some loss function that depends on
auxiliary regression models that also need to be estimated from data. The
class in this module implements the general logic in a very versatile way
so that various child classes can simply instantiate the appropriate models
and save a lot of code repetition.

References
----------

Dylan Foster, Vasilis Syrgkanis (2019). Orthogonal Statistical Learning.
    ACM Conference on Learning Theory. https://arxiv.org/abs/1901.09036

Xinkun Nie, Stefan Wager (2017). Quasi-Oracle Estimation of Heterogeneous Treatment Effects.
    https://arxiv.org/abs/1712.04912

Chernozhukov et al. (2017). Double/debiased machine learning for treatment and structural parameters.
    The Econometrics Journal. https://arxiv.org/abs/1608.00060

"""

import numpy as np
import copy
from warnings import warn
from .utilities import (shape, reshape, ndim, hstack, cross_product, transpose,
                        broadcast_unit_treatments, reshape_treatmentwise_effects,
                        StatsModelsLinearRegression, LassoCVWrapper)
from sklearn.model_selection import KFold, StratifiedKFold, check_cv
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.preprocessing import (PolynomialFeatures, LabelEncoder, OneHotEncoder,
                                   FunctionTransformer)
from sklearn.base import clone, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.utils import check_random_state
from .cate_estimator import (BaseCateEstimator, LinearCateEstimator,
                             TreatmentExpansionMixin, StatsModelsCateEstimatorMixin)
from .inference import StatsModelsInference


def _crossfit(model, folds, *args, **kwargs):
    """
    General crossfit based calculation of nuisance parameters.

    Parameters
    ----------
    model : object
        An object that supports fit and predict. Fit must accept all the args
        and the keyword arguments kwargs. Similarly predict must all accept
        all the args as arguments and kwards as keyword arguments. The fit
        function estimates a model of the nuisance function, based on the input
        data to fit. Predict evaluates the fitted nuisance function on the input
        data to predict.
    folds : list of tuples
        The crossfitting fold structure. Every entry in the list is a tuple whose
        first element are the training indices of the args and kwargs data and
        the second entry are the test indices. If the union of the test indices
        is not the full set of all indices, then the remaining nuisance parameters
        for the missing indices have value NaN.
    args : a sequence of (numpy matrices or None)
        Each matrix is a data variable whose first index corresponds to a sample
    kwargs : a sequence of key-value args, with values being (numpy matrices or None)
        Each keyword argument is of the form Var=x, with x a numpy array. Each
        of these arrays are data variables. The model fit and predict will be
        called with signature: `model.fit(*args, **kwargs)` and
        `model.predict(*args, **kwargs)`. Key-value arguments that have value
        None, are ommitted from the two calls. So all the args and the non None
        kwargs variables must be part of the models signature.

    Returns
    -------
    nuisances : tuple of numpy matrices
        Each entry in the tuple is a nuisance parameter matrix. Each row i-th in the
        matric corresponds to the value of the nuisancee parameter for the i-th input
        sample.
    model_list : list of objects of same type as input model
        The cloned and fitted models for each fold. Can be used for inspection of the
        variability of the fitted models across folds.
    fitted_inds : np array1d
        The indices of the arrays for which the nuisance value was calculated. This
        corresponds to the union of the indices of the test part of each fold in
        the input fold list.

    Examples
    --------

    .. highlight:: python
    .. code-block:: python

        import numpy as np
        from sklearn.model_selection import KFold
        from sklearn.linear_model import Lasso
        class Wrapper:
            def __init__(self, model):
                self._model = model
            def fit(self, X, y, W=None):
                self._model.fit(X, y)
                return self
            def predict(self, X, y, W=None):
                return self._model.predict(X)
        np.random.seed(123)
        X = np.random.normal(size=(5000, 3))
        y = X[:, 0] + np.random.normal(size=(5000,))
        folds = list(KFold(2).split(X, y))
        model = Lasso(alpha=0.01)
        nuisance, model_list, fitted_inds = _crossfit(Wrapper(model), folds, X, y, W=y, Z=None)

    >>> nuisance
    (array([-1.1057289 , -1.53756637, -2.4518278 , ...,  1.10628792,
       -1.82966233, -1.78227335]),)
    >>> model_list
    [<__main__.Wrapper object at 0x12f41e518>, <__main__.Wrapper object at 0x12f41e6d8>]
    >>> fitted_inds
    array([   0,    1,    2, ..., 4997, 4998, 4999])

    """
    model_list = []
    fitted_inds = []
    for idx, (train_idxs, test_idxs) in enumerate(folds):
        model_list.append(clone(model, safe=False))
        fitted_inds = np.concatenate((fitted_inds, test_idxs))

        args_train = ()
        args_test = ()
        for var in args:
            args_train += (var[train_idxs],) if var is not None else (None,)
            args_test += (var[test_idxs],) if var is not None else (None,)

        kwargs_train = {}
        kwargs_test = {}
        for key, var in kwargs.items():
            if var is not None:
                kwargs_train[key] = var[train_idxs]
                kwargs_test[key] = var[test_idxs]

        model_list[idx].fit(*args_train, **kwargs_train)

        nuisance_temp = model_list[idx].predict(*args_test, **kwargs_test)

        if not isinstance(nuisance_temp, tuple):
            nuisance_temp = (nuisance_temp,)

        if idx == 0:
            nuisances = tuple([np.full((args[0].shape[0],) + nuis.shape[1:], np.nan) for nuis in nuisance_temp])

        for it, nuis in enumerate(nuisance_temp):
            nuisances[it][test_idxs] = nuis

    return nuisances, model_list, np.sort(fitted_inds.astype(int))


class _OrthoLearner(TreatmentExpansionMixin, LinearCateEstimator):
    """
    Base class for all orthogonal learners. This class is a parent class to any method that has
    the following architecture:

    1. The CATE :math:`\\theta(X)` is the minimizer of some expected loss function

    .. math ::
        \\mathbb{E}[\\ell(V; \\theta(X), h(V))]

    where :math:`V` are all the random variables and h is a vector of nuisance functions.

    2. To estimate :math:`\\theta(X)` we first fit the h functions can calculate :math:`h(V_i)` for each sample
    :math:`i` in a crossfit manner:

        - Estimate a model :math:`\\hat{h}` for h using half of the data
        - Evaluate the learned :math:`\\hat{h}` model on the other half

    Or more generally in a KFold fit/predict approach with more folds

    3. Estimate the model for :math:`\\theta(X)` by minimizing the empirical (regularized) plugin loss:

    .. math ::
        \\mathbb{E}_n[\\ell(V; \\theta(X), \\hat{h}(V))]

    The method is a bit more general in that the final step does not need to be a loss minimization step.
    The class takes as input a model for fitting an estimate of the nuisance h given a set of samples
    and predicting the value of the learned nuisance model on any other set of samples. It also
    takes as input a model for the final estimation, that takes as input the data and their associated
    estimated nuisance values from the first stage and fits a model for the CATE :math:`\\theta(X)`. Then
    at predict time, the final model given any set of samples of the X variable, returns the estimated
    :math:`\\theta(X)`.

    The method essentially implements all the crossfit and plugin logic, so that any child classes need
    to only implement the appropriate `model_nuisance` and `model_final` and essentially nothing more.
    It also implements the basic preprocessing logic behind the expansion of discrete treatments into
    one-hot encodings.

    Parameters
    ----------
    model_nuisance: estimator
        The estimator for fitting the nuisance function. Must implement
        `fit` and `predict` methods that both have signatures:

        .. highlight:: python
        .. code-block:: python

            model_nuisance.fit(Y, T, X=X, W=W, Z=Z,
                               sample_weight=sample_weight, sample_var=sample_var)
            model_nuisance.predict(Y, T, X=X, W=W, Z=Z,
                                   sample_weight=sample_weight, sample_var=sample_var)

        In fact we allow for the model method signatures to skip any of the keyword arguments
        as long as the class is always called with the omitted keyword argument set to `None`.
        This can be enforced in child classes by re-implementing the fit and the various effect
        methods.

    model_final: estimator for fitting the response residuals to the features and treatment residuals
        Must implement `fit` and `predict` methods that must have signatures:

        .. highlight:: python
        .. code-block:: python

            model_final.fit(Y, T, X=X, W=W, Z=Z, nuisances=nuisances,
                            sample_weight=sample_weight, sample_var=sample_var)
            model_final.predict(X=X)

        Predict, should just take the features X and return the constant marginal effect. In fact we allow for the
        model method signatures to skip any of the keyword arguments as long as the class is always called with the
        omitted keyword argument set to `None`. Moreover, the predict function of the final model can take no argument
        if the class is always called with `X=None`. This can be enforced in child classes by re-implementing the fit
        and the various effect methods.

    discrete_treatment: bool
        Whether the treatment values should be treated as categorical, rather than continuous, quantities

    n_splits: int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 3-fold cross-validation,
        - integer, to specify the number of folds.
        - :term:`CV splitter`
        - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if the treatment is discrete
        :class:`~sklearn.model_selection.StratifiedKFold` is used, else,
        :class:`~sklearn.model_selection.KFold` is used
        (with a random shuffle in either case).

        Unless an iterable is used, we call `split(X,T)` to generate the splits.

    random_state: int, :class:`~numpy.random.mtrand.RandomState` instance or None
        If int, random_state is the seed used by the random number generator;
        If :class:`~numpy.random.mtrand.RandomState` instance, random_state is the random number generator;
        If None, the random number generator is the :class:`~numpy.random.mtrand.RandomState` instance used
        by `np.random`.

    Examples
    --------

    The example code below implements a very simple version of the double machine learning
    method on top of the :py:class:`~econml._ortho_learner._OrthoLearner` class, for expository purposes.
    For a more elaborate implementation of a Double Machine Learning child class of the class
    :py:class:`~econml._ortho_learner._OrthoLearner` checkout :py:class:`~econml.dml.DMLCateEstimator`
    and its child classes.

    .. highlight:: python
    .. code-block:: python

        import numpy as np
        from sklearn.linear_model import LinearRegression
        from econml._ortho_learner import _OrthoLearner
        class ModelNuisance:
            def __init__(self, model_t, model_y):
                self._model_t = model_t
                self._model_y = model_y
            def fit(self, Y, T, W=None):
                self._model_t.fit(W, T)
                self._model_y.fit(W, Y)
                return self
            def predict(self, Y, T, W=None):
                return Y - self._model_y.predict(W), T - self._model_t.predict(W)
        class ModelFinal:
            def __init__(self):
                return
            def fit(self, Y, T, W=None, nuisances=None):
                Y_res, T_res = nuisances
                self.model = LinearRegression().fit(T_res.reshape(-1, 1), Y_res)
                return self
            def predict(self):
                return self.model.coef_[0]
            def score(self, Y, T, W=None, nuisances=None):
                Y_res, T_res = nuisances
                return np.mean(Y_res - self.model.coef_[0]*T_res)**2
        np.random.seed(123)
        X = np.random.normal(size=(100, 3))
        y = X[:, 0] + X[:, 1] + np.random.normal(size=(100,))
        est = _OrthoLearner(ModelNuisance(LinearRegression(), LinearRegression()), ModelFinal(),
                            n_splits=2, discrete_treatment=False, random_state=None)
        est.fit(y, X[:, 0], W=X[:, 1:])

    >>> est.effect()
    array([1.23440172])
    >>> est.score(y, X[:, 0], W=X[:, 1:])
    0.0003880489502537651
    """

    def __init__(self, model_nuisance, model_final,
                 discrete_treatment, n_splits, random_state):
        self._model_nuisance = clone(model_nuisance, safe=False)
        self._models_nuisance = []
        self._model_final = clone(model_final, safe=False)
        self._n_splits = n_splits
        self._discrete_treatment = discrete_treatment
        self._random_state = check_random_state(random_state)
        if discrete_treatment:
            self._label_encoder = LabelEncoder()
            self._one_hot_encoder = OneHotEncoder(categories='auto', sparse=False)
        super().__init__()

    def _check_input_dims(self, Y, T, X=None, W=None, Z=None, sample_weight=None, sample_var=None):
        assert shape(Y)[0] == shape(T)[0], "Dimension mis-match!"
        assert (X is None) or (X.shape[0] == Y.shape[0]), "Dimension mis-match!"
        assert (W is None) or (W.shape[0] == Y.shape[0]), "Dimension mis-match!"
        assert (Z is None) or (Z.shape[0] == Y.shape[0]), "Dimension mis-match!"
        assert (sample_weight is None) or (sample_weight.shape[0] == Y.shape[0]), "Dimension mis-match!"
        assert (sample_var is None) or (sample_var.shape[0] == Y.shape[0]), "Dimension mis-match!"
        self._d_x = X.shape[1:] if X is not None else None

    def _check_fitted_dims(self, X):
        if X is None:
            assert self._d_x is None, "X was not None when fitting, so can't be none for effect"
        else:
            assert self._d_x == X.shape[1:], "Dimension mis-match of X with fitted X"

    def _subinds_check_none(self, var, inds):
        return var[inds] if var is not None else None

    def _filter_none_kwargs(self, **kwargs):
        non_none_kwargs = {}
        for key, value in kwargs.items():
            if value is not None:
                non_none_kwargs[key] = value
        return non_none_kwargs

    @BaseCateEstimator._wrap_fit
    def fit(self, Y, T, X=None, W=None, Z=None, sample_weight=None, sample_var=None, inference=None):
        """
        Estimate the counterfactual model from data, i.e. estimates functions τ(·,·,·), ∂τ(·,·).

        Parameters
        ----------
        Y: (n × d_y) matrix or vector of length n
            Outcomes for each sample
        T: (n × dₜ) matrix or vector of length n
            Treatments for each sample
        X: optional (n × dₓ) matrix
            Features for each sample
        W: optional (n × d_w) matrix
            Controls for each sample
        Z: optional (n × d_z) matrix
            Instruments for each sample
        sample_weight: optional (n,) vector
            Weights for each row
        sample_var: optional (n,) vector
            Sample variance
        inference: string, `Inference` instance, or None
            Method for performing inference.  This estimator supports 'bootstrap'
            (or an instance of `BootstrapInference`).

        Returns
        -------
        self
        """
        self._check_input_dims(Y, T, X, W, Z, sample_weight, sample_var)
        nuisances, fitted_inds = self.fit_nuisances(Y, T, X, W, Z, sample_weight=sample_weight)
        self.fit_final(self._subinds_check_none(Y, fitted_inds),
                       self._subinds_check_none(T, fitted_inds),
                       X=self._subinds_check_none(X, fitted_inds),
                       W=self._subinds_check_none(W, fitted_inds),
                       Z=self._subinds_check_none(Z, fitted_inds),
                       nuisances=tuple([self._subinds_check_none(nuis, fitted_inds) for nuis in nuisances]),
                       sample_weight=self._subinds_check_none(sample_weight, fitted_inds),
                       sample_var=self._subinds_check_none(sample_var, fitted_inds))
        return self

    def fit_nuisances(self, Y, T, X=None, W=None, Z=None, sample_weight=None):
        # use a binary array to get stratified split in case of discrete treatment
        splitter = check_cv(self._n_splits, [0], classifier=self._discrete_treatment)
        # if check_cv produced a new KFold or StratifiedKFold object, we need to set shuffle and random_state
        if splitter != self._n_splits and isinstance(splitter, (KFold, StratifiedKFold)):
            splitter.shuffle = True
            splitter.random_state = self._random_state

        all_vars = [var if np.ndim(var) == 2 else var.reshape(-1, 1) for var in [Z, W, X] if var is not None]
        if all_vars:
            all_vars = np.hstack(all_vars)
            folds = splitter.split(all_vars, T)
        else:
            folds = splitter.split(np.ones((T.shape[0], 1)), T)

        if self._discrete_treatment:
            T = self._label_encoder.fit_transform(T)
            # drop first column since all columns sum to one
            T = self._one_hot_encoder.fit_transform(reshape(T, (-1, 1)))[:, 1:]
            self._d_t = shape(T)[1:]
            self.transformer = FunctionTransformer(
                func=(lambda T:
                      self._one_hot_encoder.transform(
                          reshape(self._label_encoder.transform(T), (-1, 1)))[:, 1:]),
                validate=False)

        nuisances, fitted_models, fitted_inds = _crossfit(self._model_nuisance, folds,
                                                          Y, T, X=X, W=W, Z=Z, sample_weight=sample_weight)
        self._models_nuisance = fitted_models
        return nuisances, fitted_inds

    def fit_final(self, Y, T, X=None, W=None, Z=None, nuisances=None, sample_weight=None, sample_var=None):
        self._model_final.fit(Y, T, **self._filter_none_kwargs(X=X, W=W, Z=Z,
                                                               nuisances=nuisances, sample_weight=sample_weight,
                                                               sample_var=sample_var))

    def const_marginal_effect(self, X=None):
        """
        Calculate the constant marginal CATE :math:`\\theta(·)`.

        The marginal effect is conditional on a vector of
        features on a set of m test samples {Xᵢ}.

        Parameters
        ----------
        X: optional (m × dₓ) matrix
            Features for each sample.
            If X is None, it will be treated as a column of ones with a single row

        Returns
        -------
        theta: (m × d_y × dₜ) matrix
            Constant marginal CATE of each treatment on each outcome for each sample.
            Note that when Y or T is a vector rather than a 2-dimensional array,
            the corresponding singleton dimensions in the output will be collapsed
            (e.g. if both are vectors, then the output of this method will also be a vector)
        """
        self._check_fitted_dims(X)
        if X is None:
            return self._model_final.predict()
        else:
            return self._model_final.predict(X)

    def const_marginal_effect_interval(self, X=None, *, alpha=0.1):
        self._check_fitted_dims(X)
        return super().const_marginal_effect_interval(X, alpha=alpha)

    def effect_interval(self, X=None, T0=0, T1=1, *, alpha=0.1):
        self._check_fitted_dims(X)
        return super().effect_interval(X, T0=T0, T1=T1, alpha=alpha)

    def score(self, Y, T, X=None, W=None, Z=None):
        X, T = self._expand_treatments(X, T)
        n_splits = len(self._models_nuisance)
        for idx, mdl in enumerate(self._models_nuisance):
            nuisance_temp = mdl.predict(Y, T, **self._filter_none_kwargs(X=X, W=W, Z=Z))
            if not isinstance(nuisance_temp, tuple):
                nuisance_temp = (nuisance_temp,)

            if idx == 0:
                nuisances = [np.zeros((n_splits,) + nuis.shape) for nuis in nuisance_temp]

            for it, nuis in enumerate(nuisance_temp):
                nuisances[it][idx] = nuis

        for it in range(len(nuisances)):
            nuisances[it] = np.mean(nuisances[it], axis=0)

        return self._model_final.score(Y, T, **self._filter_none_kwargs(X=X, W=W, Z=Z, nuisances=nuisances))

    @property
    def model_final(self):
        return self._model_final
