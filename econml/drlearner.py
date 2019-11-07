# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Doubly Robust Learner. The method uses the doubly robust correction to construct doubly
robust estimates of all the potential outcomes of each samples. Then estimates a CATE model
by regressing the potential outcome differences on the heterogeneity features X.

References
----------

Dylan Foster, Vasilis Syrgkanis (2019).
    Orthogonal Statistical Learning.
    ACM Conference on Learning Theory. https://arxiv.org/abs/1901.09036

Robins, J.M., Rotnitzky, A., and Zhao, L.P. (1994).
    Estimation of regression coefficients when some regressors are not always observed.
    Journal of the American Statistical Association 89,846–866.

Bang, H. and Robins, J.M. (2005).
    Doubly robust estimation in missing data and causal inference models.
    Biometrics 61,962–972.

Tsiatis AA (2006).
    Semiparametric Theory and Missing Data.
    New York: Springer; 2006.

"""

import numpy as np
from sklearn.linear_model import LogisticRegressionCV, LinearRegression, LassoCV
from econml.utilities import WeightedLassoCV
from sklearn.base import clone
from econml._ortho_learner import _OrthoLearner
from econml.cate_estimator import StatsModelsCateEstimatorDiscreteMixin
from econml.utilities import StatsModelsLinearRegression
from sklearn.preprocessing import PolynomialFeatures


def _filter_none_kwargs(**kwargs):
    out_kwargs = {}
    for key, value in kwargs.items():
        if value is not None:
            out_kwargs[key] = value
    return out_kwargs


class DRLearner(_OrthoLearner):
    """
    CATE estimator that uses doubly-robust correction techniques to account for
    covariate shift (selection bias) between the treatment arms. The estimator is a special
    case of an :class:`~econml._ortho_learner._OrthoLearner` estimator, so it follows the two
    stage process, where a set of nuisance functions are estimated in the first stage in a crossfitting
    manner and a final stage estimates the CATE model. See the documentation of
    :class:`~econml._ortho_learner._OrthoLearner` for a description of this two stage process.

    In this estimator, the CATE is estimated by using the following estimating equations. If we let:

    .. math ::
        Y_{i, t}^{DR} = E[Y | X_i, W_i, T_i]\
            + \\sum_{t=0}^{n_t} \\frac{Y_i - E[Y | X_i, W_i, T_i]}{Pr[T=t | X_i, W_i]} \\cdot 1\\{T_i=t\\}

    Then the following estimating equation holds:

    .. math ::
        E\\left[Y_{i, t}^{DR} - Y_{i, 0}^{DR} | X_i\\right] = \\theta_t(X_i)

    Thus if we estimate the nuisance functions :math:`h(X, W, T) = E[Y | X, W, T]` and
    :math:`p_t(X, W)=Pr[T=t | X, W]` in the first stage, we can estimate the final stage cate for each
    treatment t, by running a regression, regressing :math:`Y_{i, t}^{DR} - Y_{i, 0}^{DR}` on :math:`X_i`.

    The problem of estimating the nuisance function :math:`p` is a simple multi-class classification
    problem of predicting the label :math:`T` from :math:`X, W`. The :class:`~econml.drlearner.DRLearner`
    class takes as input the parameter :code:`model_propensity`, which is an arbitrary scikit-learn
    classifier, that is internally used to solve this classification problem.

    The second nuisance function :math:`h` is a simple regression problem and the :class:`~econml.drlearner.DRLearner`
    class takes as input the parameter :code:`model_regressor`, which is an arbitrary scikit-learn regressor that
    is internally used to solve this regression problem.

    The final stage is multi-task regression problem with outcomes the labels :math:`Y_{i, t}^{DR} - Y_{i, 0}^{DR}`
    for each non-baseline treatment t. The :class:`~econml.drlearner.DRLearner` takes as input parameter
    :code:`model_final`, which is any scikit-learn regressor that is internally used to solve this multi-task
    regresion problem. If the parameter :code:`multitask_model_final` is False, then this model is assumed
    to be a mono-task regressor, and separate clones of it are used to solve each regression target
    separately.

    Parameters
    ----------
    model_propensity : scikit-learn classifier
        Estimator for Pr[T=t | X, W]. Trained by regressing treatments on (features, controls) concatenated.
        Must implement `fit` and `predict_proba` methods. The `fit` method must be able to accept X and T,
        where T is a shape (n, ) array.

    model_regression : scikit-learn regressor
        Estimator for E[Y | X, W, T]. Trained by regressing Y on (features, controls, one-hot-encoded treatments)
        concatenated. The one-hot-encoding excludes the baseline treatment. Must implement `fit` and
        `predict` methods. If different models per treatment arm are desired, see the
        :class:`~econml.utilities.MultiModelWrapper` helper class.

    model_final :
        estimator for the final cate model. Trained on regressing the doubly robust potential outcomes
        on (features X).

        - If X is None, then the fit method of model_final should be able to handle X=None.
        - If featurizer is not None and X is not None, then it is trained on the outcome of
          featurizer.fit_transform(X).
        - If multitask_model_final is True, then this model must support multitasking
          and it is trained by regressing all doubly robust target outcomes on (featurized) features simultanteously.
        - The output of the predict(X) of the trained model will contain the CATEs for each treatment compared to
          baseline treatment (lexicographically smallest). If multitask_model_final is False, it is assumed to be a
          mono-task model and a separate clone of the model is trained for each outcome. Then predict(X) of the t-th
          clone will be the CATE of the t-th lexicographically ordered treatment compared to the baseline.

    multitask_model_final : optional bool (default=False)
        Whether the model_final should be treated as a multi-task model. See description of model_final.

    featurizer : sklearn featurizer or None
        Must support fit_transform and transform. Used to create composite features in the final CATE regression.
        It is ignored if X is None. The final CATE will be trained on the outcome of featurizer.fit_transform(X).
        If featurizer=None, then CATE is trained on X.

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

        Unless an iterable is used, we call `split(concat[W, X], T)` to generate the splits. If all
        W, X are None, then we call `split(ones((T.shape[0], 1)), T)`.

    random_state: int, :class:`~numpy.random.mtrand.RandomState` instance or None
        If int, random_state is the seed used by the random number generator;
        If :class:`~numpy.random.mtrand.RandomState` instance, random_state is the random number generator;
        If None, the random number generator is the :class:`~numpy.random.mtrand.RandomState` instance used
        by `np.random`.

    Examples
    --------
    A simple example with the default models::

        import numpy as np
        import scipy.special
        from econml.drlearner import DRLearner

        np.random.seed(123)
        X = np.random.normal(size=(1000, 3))
        T = np.random.binomial(2, scipy.special.expit(X[:, 0]))
        sigma = 0.001
        y = (1 + .5*X[:, 0]) * T + X[:, 0] + np.random.normal(0, sigma, size=(1000,))
        est = DRLearner()
        est.fit(y, T, X=X, W=None)

    >>> est.const_marginal_effect(X[:2])
    array([[ 0.5215622 ,  0.82215814],
           [ 0.37704938,  0.21466424],
           [-0.07505456, -0.77963048]])
    >>> est.effect(X[:2], T0=0, T1=1)
    array([0.5215622 , 0.37704938])
    >>> est.score_
    10.243375492811202
    >>> est.score(y, T, X=X)
    8.489141208026698
    >>> est.model_cate(T=1).coef_
    array([1.00761575, 0.47127132, 0.01092897, 0.05185222])
    >>> est.model_cate(T=2).coef_
    array([ 1.92481336,  1.09654124,  0.08919048, -0.00413531])
    >>> est.cate_feature_names
    ['1', 'x0', 'x1', 'x2']
    >>> [mdl.coef_ for mdl in est.models_regression]
    [array([ 1.43608627e+00,  9.16715532e-04, -7.66401138e-03,  6.73985763e-01,
             1.98864974e+00]),
     array([ 1.49529047e+00, -2.43886553e-03,  1.74824661e-03,  6.81810603e-01,
             2.03340844e+00])]
    >>> [mdl.coef_ for mdl in est.models_propensity]
    [array([[-1.05971312,  0.09307097,  0.11409781],
            [ 0.09002839,  0.03464788, -0.09079638],
            [ 0.96968473, -0.12771885, -0.02330143]]),
     array([[-0.98251905,  0.09248893, -0.12248101],
            [ 0.04591711, -0.03486403, -0.07891743],
            [ 0.93660195, -0.05762491,  0.20139844]])]

    Beyond default models::

        import scipy.special
        import numpy as np
        from sklearn.linear_model import LassoCV
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
        from econml.drlearner import DRLearner

        np.random.seed(123)
        X = np.random.normal(size=(1000, 3))
        T = np.random.binomial(2, scipy.special.expit(X[:, 0]))
        sigma = 0.01
        y = (1 + .5*X[:, 0]) * T + X[:, 0] + np.random.normal(0, sigma, size=(1000,))
        est = DRLearner(model_propensity=GradientBoostingClassifier(),
                        model_regression=GradientBoostingRegressor(),
                        model_final=LassoCV(cv=3),
                        featurizer=None)
        est.fit(y, T, X=X, W=None)

    >>> est.score_
    3.172135302455655
    >>> est.const_marginal_effect(X[:3])
    array([[ 0.55038338,  1.14558174],
           [ 0.32065866,  0.75638221],
           [-0.07514842, -0.03658315]])
    >>> est.model_cate(T=2).coef_
    array([ 0.86420672,  0.01628151, -0.        ])
    >>> est.model_cate(T=2).intercept_
    2.067552713536296
    >>> est.model_cate(T=1).coef_
    array([0.43487391, 0.02968939, 0.        ])
    >>> est.model_cate(T=1).intercept_
    0.9928852195090293

    Attributes
    ----------
    models_propensity: list of objects of type(model_propensity)
        A list of instances of the model_propensity object. Each element corresponds to a crossfitting
        fold and is the model instance that was fitted for that training fold.
    models_regression: list of objects of type(model_regression)
        A list of instances of the model_regression object. Each element corresponds to a crossfitting
        fold and is the model instance that was fitted for that training fold.
    model_cate(T=t) : object of type(model_final)
        An instance of the model_final object that was fitted after calling fit which corresponds
        to the CATE model for treatment T=t, compared to baseline. Available only when multitask_model_final=False.
    multitask_model_cate : object of type(model_final)
        An instance of the model_final object that was fitted after calling fit which corresponds whose
        vector of outcomes correspond to the CATE model for each treatment, compared to baseline.
        Available only when multitask_model_final=False.
    featurizer : object of type(featurizer)
        An instance of the fitted featurizer that was used to preprocess X in the final CATE model training.
        Available only when featurizer is not None and X is not None.
    cate_feature_names(input_feature_names=None) : list of feature names or None
        A list of feature names that correspond to the input features in the final CATE model. If
        input_feature_names is not None and featurizer is None, then the input_feature_names are returned.
        If the featurizer is not None, then this attribute is available only when the featurizer has
        a method: `get_feature_names(input_feature_names)`. Otherwise None is returned.
    score_ : float
        The MSE in the final doubly robust potential outcome regressions, i.e.

        .. math::
            \\frac{1}{n_t} \\sum_{t=1}^{n_t} \\frac{1}{n} \\sum_{i=1}^n (Y_{i, t}^{DR} - \\hat{\\theta}_t(X_i))^2

        where n_t is the number of treatments (excluding control).

        If `sample_weight` is not None at fit time, then a weighted average across samples is returned.


    """

    def __init__(self, model_propensity=LogisticRegressionCV(cv=3, solver='lbfgs', multi_class='auto'),
                 model_regression=WeightedLassoCV(cv=3),
                 model_final=StatsModelsLinearRegression(),
                 multitask_model_final=False,
                 featurizer=None,
                 n_splits=2,
                 random_state=None):
        class ModelNuisance:
            def __init__(self, model_propensity, model_regression):
                self._model_propensity = model_propensity
                self._model_regression = model_regression

            def _combine(self, X, W):
                return np.hstack([arr for arr in [X, W] if arr is not None])

            def fit(self, Y, T, X=None, W=None, *, sample_weight=None):
                assert np.ndim(Y) == 1, "Can only accept single dimensional outcomes Y! Use Y.ravel()."
                if (X is None) and (W is None):
                    raise AttributeError("At least one of X or W has to not be None!")
                if np.any(np.all(T == 0, axis=0)) or (not np.any(np.all(T == 0, axis=1))):
                    raise AttributeError("Provided crossfit folds contain training splits that " +
                                         "don't contain all treatments")
                XW = self._combine(X, W)
                filtered_kwargs = _filter_none_kwargs(sample_weight=sample_weight)
                self._model_propensity.fit(XW, np.matmul(T, np.arange(1, T.shape[1] + 1)).ravel().astype(int),
                                           **filtered_kwargs)
                self._model_regression.fit(np.hstack([XW, T]), Y, **filtered_kwargs)
                return self

            def predict(self, Y, T, X=None, W=None, *, sample_weight=None):
                XW = self._combine(X, W)
                propensities = self._model_propensity.predict_proba(XW)
                Y_pred = np.zeros((T.shape[0], T.shape[1] + 1))
                T_counter = np.zeros(T.shape)
                Y_pred[:, 0] = self._model_regression.predict(np.hstack([XW, T_counter]))
                Y_pred[:, 0] += (Y - Y_pred[:, 0]) * np.all(T == 0, axis=1) / propensities[:, 0]
                for t in np.arange(T.shape[1]):
                    T_counter = np.zeros(T.shape)
                    T_counter[:, t] = 1
                    Y_pred[:, t + 1] = self._model_regression.predict(np.hstack([XW, T_counter]))
                    Y_pred[:, t + 1] += (Y - Y_pred[:, t + 1]) * (T[:, t] == 1) / propensities[:, t + 1]
                return Y_pred

        class ModelFinal:
            def __init__(self, model_final, featurizer, multitask_model_final):
                self._model_final = clone(model_final, safe=False)
                self._featurizer = clone(featurizer, safe=False)
                self._multitask_model_final = multitask_model_final
                return

            def _filter_none_kwargs(self, **kwargs):
                out_kwargs = {}
                for key, value in kwargs.items():
                    if value is not None:
                        out_kwargs[key] = value
                return out_kwargs

            def fit(self, Y, T, X=None, W=None, nuisances=None, *, sample_weight=None, sample_var=None):
                Y_pred, = nuisances
                if (X is not None) and (self._featurizer is not None):
                    X = self._featurizer.fit_transform(X)
                filtered_kwargs = _filter_none_kwargs(sample_weight=sample_weight, sample_var=sample_var)
                if self._multitask_model_final:
                    self.model_cate = clone(self._model_final, safe=False).fit(
                        X, Y_pred[:, 1:] - Y_pred[:, [0]], **filtered_kwargs)
                else:
                    self.models_cate = [clone(self._model_final, safe=False).fit(X, Y_pred[:, t] - Y_pred[:, 0],
                                                                                 **filtered_kwargs)
                                        for t in np.arange(1, Y_pred.shape[1])]
                return self

            def predict(self, X=None):
                if (X is not None) and (self._featurizer is not None):
                    X = self._featurizer.transform(X)
                if self._multitask_model_final:
                    return self.model_cate.predict(X)
                else:
                    return np.array([mdl.predict(X) for mdl in self.models_cate]).T

            def score(self, Y, T, X=None, W=None, nuisances=None, *, sample_weight=None, sample_var=None):
                if (X is not None) and (self._featurizer is not None):
                    X = self._featurizer.transform(X)
                Y_pred, = nuisances
                if sample_weight is None:
                    sample_weight = np.ones(Y.shape[0])
                if self._multitask_model_final:
                    return np.mean(np.average((Y_pred[:, 1:] - Y_pred[:, [0]] - self.model_cate.predict(X))**2,
                                              weights=sample_weight, axis=0))
                else:
                    return np.mean([np.average((Y_pred[:, t] - Y_pred[:, 0] - self.models_cate[t - 1].predict(X))**2,
                                               weights=sample_weight, axis=0)
                                    for t in np.arange(1, Y_pred.shape[1])])

        self._multitask_model_final = multitask_model_final
        super().__init__(ModelNuisance(model_propensity, model_regression),
                         ModelFinal(model_final, featurizer, multitask_model_final),
                         n_splits=n_splits, discrete_treatment=True,
                         random_state=random_state)

    def fit(self, Y, T, X=None, W=None, *, sample_weight=None, sample_var=None, inference=None):
        """
        Estimate the counterfactual model from data, i.e. estimates function: math: `\\theta(\\cdot)`.

        Parameters
        ----------
        Y: (n,) vector of length n
            Outcomes for each sample
        T: (n,) vector of length n
            Treatments for each sample
        X: optional(n, d_x) matrix or None (Default=None)
            Features for each sample
        W: optional(n, d_w) matrix or None (Default=None)
            Controls for each sample
        sample_weight: optional(n,) vector or None (Default=None)
            Weights for each samples
        sample_var: optional(n,) vector or None (Default=None)
            Sample variance for each sample
        inference: string, `Inference` instance, or None
            Method for performing inference.  This estimator supports 'bootstrap'
            (or an instance of `BootstrapInference`).

        Returns
        -------
        self: DRLearner instance
        """
        # Replacing fit from _OrthoLearner, to enforce Z=None and improve the docstring
        return super().fit(Y, T, X=X, W=W, sample_weight=sample_weight, sample_var=sample_var, inference=inference)

    def score(self, Y, T, X=None, W=None):
        """
        Score the fitted CATE model on a new data set. Generates nuisance parameters
        for the new data set based on the fitted residual nuisance models created at fit time.
        It uses the mean prediction of the models fitted by the different crossfit folds.
        Then calculates the MSE of the final residual Y on residual T regression.

        If model_final does not have a score method, then it raises an `AttributeError`

        Parameters
        ----------
        Y: (n,) vector of length n
            Outcomes for each sample
        T: (n,) vector of length n
            Treatments for each sample
        X: optional(n, d_x) matrix or None (Default=None)
            Features for each sample
        W: optional(n, d_w) matrix or None (Default=None)
            Controls for each sample

        Returns
        -------
        score: float
            The MSE of the final CATE model on the new data.
        """
        # Replacing score from _OrthoLearner, to enforce Z=None and improve the docstring
        return super().score(Y, T, X=X, W=W)

    @property
    def multitask_model_cate(self):
        if not self._multitask_model_final:
            raise AttributeError("Separte CATE models were fitted for each treatment! Use model_cate.")
        return super().model_final.model_cate

    def model_cate(self, T=1):
        if self._multitask_model_final:
            raise AttributeError("A single multitask model was fitted for all treatments! Use multitask_model_cate.")
        _, T = self._expand_treatments(None, T)
        ind = (T @ np.arange(1, T.shape[1] + 1)).astype(int)[0] - 1
        return super().model_final.models_cate[ind]

    @property
    def models_propensity(self):
        return [mdl._model_propensity for mdl in super().models_nuisance]

    @property
    def models_regression(self):
        return [mdl._model_regression for mdl in super().models_nuisance]

    @property
    def featurizer(self):
        return super().model_final._featurizer

    def cate_feature_names(self, input_feature_names=None):
        if self.featurizer is None:
            return input_feature_names
        elif hasattr(self.featurizer, 'get_feature_names'):
            return self.featurizer.get_feature_names(input_feature_names)
        else:
            raise AttributeError("Featurizer does not have a method: get_feature_names!")


class LinearDRLearner(StatsModelsCateEstimatorDiscreteMixin, DRLearner):
    """Special case of the :class:`~econml.drlearner.DRLearner` where the final stage
    is a Linear Regression on a low dimensional set of features. In this case, inference
    can be performed via the asymptotic normal characterization of the estimated parameters.
    This is computationally faster than bootstrap inference. Set :code:`inference='statsmodels'`
    at fit time, to enable inference via asymptotic normality.

    Parameters
    ----------
    model_propensity : scikit-learn classifier
        Estimator for Pr[T=t | X, W]. Trained by regressing treatments on (features, controls) concatenated.
        Must implement `fit` and `predict_proba` methods. The `fit` method must be able to accept X and T,
        where T is a shape (n, ) array.

    model_regression : scikit-learn regressor
        Estimator for E[Y | X, W, T]. Trained by regressing Y on (features, controls, one-hot-encoded treatments)
        concatenated. The one-hot-encoding excludes the baseline treatment. Must implement `fit` and
        `predict` methods. If different models per treatment arm are desired, see the
        :class:`~econml.utilities.MultiModelWrapper` helper class.

    featurizer : sklearn featurizer or None
        Must support fit_transform and transform. Used to create composite features in the final CATE regression.
        It is ignored if X is None. The final CATE will be trained on the outcome of featurizer.fit_transform(X).
        If featurizer=None, then CATE is trained on X.

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
    A simple example with the default models::

        import numpy as np
        import scipy.special
        from econml.drlearner import DRLearner, LinearDRLearner

        np.random.seed(123)
        X = np.random.normal(size=(1000, 3))
        T = np.random.binomial(2, scipy.special.expit(X[:, 0]))
        y = (1 + .5*X[:, 0]) * T + X[:, 0] + np.random.normal(size=(1000,))
        est = LinearDRLearner()
        est.fit(y, T, X=X, W=None, inference='statsmodels')

    >>> est.effect(X[:3])
    array([ 0.45450782,  0.32446905, -0.07040134])
    >>> est.effect_interval(X[:3])
    (array([ 0.18655358, -0.11752159, -0.58922191]),
     array([0.72246206, 0.76645968, 0.44841923]))
    >>> est.model_cate(T=1).coef_
    array([0.4097647 , 0.01972211, 0.05364835])
    >>> est.model_cate(T=1).coef__interval()
    (array([ 0.14622515, -0.2045328 , -0.17625388]),
    array([0.67330426, 0.24397702, 0.28355057]))
    >>> est.model_cate(T=1).intercept_
    0.8645098360137696
    >>> est.model_cate(T=1).intercept__interval()
    (0.641858878564784, 1.0871607934627552)

    Attributes
    ----------
    models_propensity: list of objects of type(model_propensity)
        A list of instances of the model_propensity object. Each element corresponds to a crossfitting
        fold and is the model instance that was fitted for that training fold.
    models_regression: list of objects of type(model_regression)
        A list of instances of the model_regression object. Each element corresponds to a crossfitting
        fold and is the model instance that was fitted for that training fold.
    model_cate(T=t) : object of type(model_final)
        An instance of the model_final object that was fitted after calling fit which corresponds
        to the CATE model for treatment T=t, compared to baseline. Available only when multitask_model_final=False.
    featurizer : object of type(featurizer)
        An instance of the fitted featurizer that was used to preprocess X in the final CATE model training.
        Available only when featurizer is not None and X is not None.
    cate_feature_names(input_feature_names=None) : list of feature names or None
        A list of feature names that correspond to the input features in the final CATE model. If
        input_feature_names is not None and featurizer is None, then the input_feature_names are returned.
        If the featurizer is not None, then this attribute is available only when the featurizer has
        a method: `get_feature_names(input_feature_names)`. Otherwise None is returned.
    score_ : float
        The MSE in the final doubly robust potential outcome regressions, i.e.

        .. math::
            \\frac{1}{n_t} \\sum_{t=1}^{n_t} \\frac{1}{n} \\sum_{i=1}^n (Y_{i, t}^{DR} - \\hat{\\theta}_t(X_i))^2

        where n_t is the number of treatments (excluding control).

        If `sample_weight` is not None at fit time, then a weighted average across samples is returned.

    """

    def __init__(self,
                 model_propensity=LogisticRegressionCV(cv=3, solver='lbfgs', multi_class='auto'),
                 model_regression=WeightedLassoCV(cv=3),
                 featurizer=None,
                 fit_cate_intercept=True,
                 n_splits=2, random_state=None):
        super().__init__(model_propensity=model_propensity,
                         model_regression=model_regression,
                         model_final=StatsModelsLinearRegression(fit_intercept=fit_cate_intercept),
                         featurizer=featurizer,
                         multitask_model_final=False,
                         n_splits=n_splits,
                         random_state=random_state)

    @property
    def statsmodels(self):
        return self.model_final._model_final

    @property
    def statsmodels_fitted(self):
        return self.model_final.models_cate
