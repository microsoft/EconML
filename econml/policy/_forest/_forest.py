# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#
# This code contains snippets of code from
# https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/ensemble/_forest.py
# published under the following license and copyright:
# BSD 3-Clause License
#
# Copyright (c) 2007-2020 The scikit-learn developers.
# All rights reserved.

import numbers
from warnings import catch_warnings, simplefilter, warn
from abc import ABCMeta, abstractmethod
import numpy as np
import threading
from ..._ensemble import (BaseEnsemble, _partition_estimators, _get_n_samples_subsample, _accumulate_prediction)
from ...utilities import check_inputs, cross_product
from ...tree._tree import DTYPE, DOUBLE
from ._tree import PolicyTree
from joblib import Parallel, delayed
from scipy.sparse import hstack as sparse_hstack
from sklearn.utils import check_random_state, compute_sample_weight
from sklearn.utils.validation import _check_sample_weight, check_is_fitted
from sklearn.utils import check_X_y

__all__ = ["PolicyForest"]

MAX_INT = np.iinfo(np.int32).max

# =============================================================================
# Policy Forest
# =============================================================================


class PolicyForest(BaseEnsemble, metaclass=ABCMeta):
    """ Welfare maximization policy forest.

    Parameters
    ----------
    n_estimators : integer, optional (default=100)
        The total number of trees in the forest. The forest consists of a
        forest of sqrt(n_estimators) sub-forests, where each sub-forest
        contains sqrt(n_estimators) trees.

    criterion : {``'neg_welfare'``}, default='neg_welfare'
        The criterion type

    splitter : {"best"}, default="best"
        The strategy used to choose the split at each node. Supported
        strategies are "best" to choose the best split.

    max_depth : int, default=None
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int or float, default=10
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

    min_samples_leaf : int or float, default=5
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

    min_weight_fraction_leaf : float, default=0.0
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_features : int, float or {"auto", "sqrt", "log2"}, default=None
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=n_features`.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.
        The weighted impurity decrease equation is the following::

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.
        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.

    min_balancedness_tol: float in [0, .5], default=.45
        How imbalanced a split we can tolerate. This enforces that each split leaves at least
        (.5 - min_balancedness_tol) fraction of samples on each side of the split; or fraction
        of the total weight of samples, when sample_weight is not None. Default value, ensures
        that at least 5% of the parent node weight falls in each side of the split. Set it to 0.0 for no
        balancedness and to .5 for perfectly balanced splits. For the formal inference theory
        to be valid, this has to be any positive constant bounded away from zero.

    honest: bool, default=True
        Whether the data should be split in two equally sized samples, such that the one half-sample
        is used to determine the optimal split at each node and the other sample is used to determine
        the value of every node.

    n_jobs : int or None, optional (default=-1)
        The number of jobs to run in parallel for both `fit` and `predict`.
        ``None`` means 1 unless in a :func:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    verbose : int, optional (default=0)
        Controls the verbosity when fitting and predicting.

    random_state: int, :class:`~numpy.random.mtrand.RandomState` instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If :class:`~numpy.random.mtrand.RandomState` instance, random_state is the random number generator;
        If None, the random number generator is the :class:`~numpy.random.mtrand.RandomState` instance used
        by :mod:`np.random<numpy.random>`.

    warm_start : bool, default=False
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit a whole
        new forest.
    """

    def __init__(self,
                 n_estimators=100, *,
                 criterion='neg_welfare',
                 max_depth=None,
                 min_samples_split=10,
                 min_samples_leaf=5,
                 min_weight_fraction_leaf=0.,
                 max_features="auto",
                 min_impurity_decrease=0.,
                 max_samples=.5,
                 min_balancedness_tol=.45,
                 honest=True,
                 n_jobs=-1,
                 random_state=None,
                 verbose=0,
                 warm_start=False):
        super().__init__(
            base_estimator=PolicyTree(),
            n_estimators=n_estimators,
            estimator_params=("criterion", "max_depth", "min_samples_split",
                              "min_samples_leaf", "min_weight_fraction_leaf",
                              "max_features", "min_impurity_decrease", "honest",
                              "min_balancedness_tol",
                              "random_state"))

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.min_impurity_decrease = min_impurity_decrease
        self.min_balancedness_tol = min_balancedness_tol
        self.honest = honest
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.warm_start = warm_start
        self.max_samples = max_samples

    def apply(self, X):
        """
        Apply trees in the forest to X, return leaf indices.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float64``.

        Returns
        -------
        X_leaves : ndarray of shape (n_samples, n_estimators)
            For each datapoint x in X and for each tree in the forest,
            return the index of the leaf x ends up in.
        """
        X = self._validate_X_predict(X)
        results = Parallel(n_jobs=self.n_jobs, verbose=self.verbose, backend="threading")(
            delayed(tree.apply)(X, check_input=False)
            for tree in self.estimators_)

        return np.array(results).T

    def decision_path(self, X):
        """
        Return the decision path in the forest.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float64``.

        Returns
        -------
        indicator : sparse matrix of shape (n_samples, n_nodes)
            Return a node indicator matrix where non zero elements indicates
            that the samples goes through the nodes. The matrix is of CSR
            format.
        n_nodes_ptr : ndarray of shape (n_estimators + 1,)
            The columns from indicator[n_nodes_ptr[i]:n_nodes_ptr[i+1]]
            gives the indicator value for the i-th estimator.
        """
        X = self._validate_X_predict(X)
        indicators = Parallel(n_jobs=self.n_jobs, verbose=self.verbose, backend='threading')(
            delayed(tree.decision_path)(X, check_input=False)
            for tree in self.estimators_)

        n_nodes = [0]
        n_nodes.extend([i.shape[1] for i in indicators])
        n_nodes_ptr = np.array(n_nodes).cumsum()

        return sparse_hstack(indicators).tocsr(), n_nodes_ptr

    def fit(self, X, y, *, sample_weight=None, **kwargs):
        """
        Build a forest of trees from the training set (X, y) and any other auxiliary variables.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples. Internally, its dtype will be converted
            to ``dtype=np.float64``.
        y : array-like of shape (n_samples,) or (n_samples, n_outcomes)
            The outcome values for each sample.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node.
        **kwargs : dictionary of array-like items of shape (n_samples, d_var)
            Auxiliary random variables

        Returns
        -------
        self : object
        """

        X, y = check_X_y(X, y, multi_output=True)

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X, DOUBLE)

        # Remap output
        n_samples, self.n_features_ = X.shape

        y = np.atleast_1d(y)
        if y.ndim == 1:
            # reshape is necessary to preserve the data contiguity against vs
            # [:, np.newaxis] that does not.
            y = np.reshape(y, (-1, 1))

        self.n_outputs_ = y.shape[1]

        if getattr(y, "dtype", None) != DOUBLE or not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=DOUBLE)

        if getattr(X, "dtype", None) != DTYPE:
            X = X.astype(DTYPE)

        # Get subsample sample size
        n_samples_subsample = _get_n_samples_subsample(
            n_samples=n_samples,
            max_samples=self.max_samples
        )

        # Check parameters
        self._validate_estimator()

        random_state = check_random_state(self.random_state)
        # We re-initialize the subsample_random_seed_ only if we are not in warm_start mode or
        # if this is the first `fit` call of the warm start mode.
        if (not self.warm_start) or (not hasattr(self, 'subsample_random_seed_')):
            self.subsample_random_seed_ = random_state.randint(MAX_INT)
        else:
            random_state.randint(MAX_INT)  # just advance random_state
        subsample_random_state = check_random_state(self.subsample_random_seed_)

        if not self.warm_start or not hasattr(self, "estimators_"):
            # Free allocated memory, if any
            self.estimators_ = []
            self.slices_ = []
            # the below are needed to replicate randomness of subsampling when warm_start=True
            self.n_samples_ = []
            self.n_samples_subsample_ = []

        n_more_estimators = self.n_estimators - len(self.estimators_)

        if n_more_estimators < 0:
            raise ValueError('n_estimators=%d must be larger or equal to '
                             'len(estimators_)=%d when warm_start==True'
                             % (self.n_estimators, len(self.estimators_)))

        elif n_more_estimators == 0:
            warn("Warm-start fitting without increasing n_estimators does not "
                 "fit new trees.", UserWarning)
        else:

            if self.warm_start and len(self.estimators_) > 0:
                # We draw from the random state to get the random state we
                # would have got if we hadn't used a warm_start.
                random_state.randint(MAX_INT, size=len(self.estimators_))

            trees = [self._make_estimator(append=False,
                                          random_state=random_state).init()
                     for i in range(n_more_estimators)]

            if self.warm_start:
                # Advancing subsample_random_state. Assumes each prior fit call has the same number of
                # samples at fit time. If not then this would not exactly replicate a single batch execution,
                # but would still advance randomness enough so that tree subsamples will be different.
                for _, n_, ns_ in zip(range(len(self.estimators_)), self.n_samples_, self.n_samples_subsample_):
                    subsample_random_state.choice(n_, ns_, replace=False)
            s_inds = [subsample_random_state.choice(n_samples, n_samples_subsample, replace=False)
                      for _ in range(n_more_estimators)]

            # Parallel loop: we prefer the threading backend as the Cython code
            # for fitting the trees is internally releasing the Python GIL
            # making threading more efficient than multiprocessing in
            # that case. However, for joblib 0.12+ we respect any
            # parallel_backend contexts set at a higher level,
            # since correctness does not rely on using threads.
            trees = Parallel(n_jobs=self.n_jobs, verbose=self.verbose, backend='threading')(
                delayed(t.fit)(X[s], y[s],
                               sample_weight=sample_weight[s] if sample_weight is not None else None,
                               check_input=False)
                for t, s in zip(trees, s_inds))

            # Collect newly grown trees
            self.estimators_.extend(trees)
            self.n_samples_.extend([n_samples] * len(trees))
            self.n_samples_subsample_.extend([n_samples_subsample] * len(trees))

        return self

    def get_subsample_inds(self,):
        """ Re-generate the example same sample indices as those at fit time using same pseudo-randomness.
        """
        check_is_fitted(self)
        subsample_random_state = check_random_state(self.subsample_random_seed_)
        return [subsample_random_state.choice(n_, ns_, replace=False)
                for n_, ns_ in zip(self.n_samples_, self.n_samples_subsample_)]

    def feature_importances(self, max_depth=4, depth_decay_exponent=2.0):
        """
        The feature importances based on the amount of parameter heterogeneity they create.
        The higher, the more important the feature.

        Parameters
        ----------
        max_depth : int, default=4
            Splits of depth larger than `max_depth` are not used in this calculation
        depth_decay_exponent: double, default=2.0
            The contribution of each split to the total score is re-weighted by 1 / (1 + `depth`)**2.0.
        Returns
        -------
        feature_importances_ : ndarray of shape (n_features,)
            Normalized total parameter heterogeneity inducing importance of each feature
        """
        check_is_fitted(self)

        all_importances = Parallel(n_jobs=self.n_jobs, backend='threading')(
            delayed(tree.feature_importances)(
                max_depth=max_depth, depth_decay_exponent=depth_decay_exponent)
            for tree in self.estimators_ if tree.tree_.node_count > 1)

        if not all_importances:
            return np.zeros(self.n_features_, dtype=np.float64)

        all_importances = np.mean(all_importances,
                                  axis=0, dtype=np.float64)
        return all_importances / np.sum(all_importances)

    @property
    def feature_importances_(self):
        return self.feature_importances()

    def _validate_X_predict(self, X):
        """
        Validate X whenever one tries to predict, apply, and other predict methods."""
        check_is_fitted(self)

        return self.estimators_[0]._validate_X_predict(X, check_input=True)

    def predict_value(self, X):

        check_is_fitted(self)
        # Check data
        X = self._validate_X_predict(X)

        # Assign chunk of trees to jobs
        n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)

        # avoid storing the output of every estimator by summing them here
        y_hat = np.zeros((X.shape[0], self.n_outputs_), dtype=np.float64)

        # Parallel loop
        lock = threading.Lock()
        Parallel(n_jobs=n_jobs, verbose=self.verbose, backend='threading', require="sharedmem")(
            delayed(_accumulate_prediction)(e.predict_value, X, [y_hat], lock)
            for e in self.estimators_)

        y_hat /= len(self.estimators_)

        return y_hat

    def predict(self, X):
        return np.argmax(self.predict_value(X), axis=1)