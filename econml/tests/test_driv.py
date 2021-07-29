# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import unittest
import pytest
import pickle
import numpy as np
from scipy import special
from sklearn.linear_model import LinearRegression, LogisticRegression
from econml.iv.dr._dr import _DummyCATE
from econml.sklearn_extensions.linear_model import StatsModelsLinearRegression
from sklearn.preprocessing import PolynomialFeatures
from econml.utilities import shape
from econml.iv.dr import (DRIV, LinearDRIV, SparseLinearDRIV, ForestDRIV, IntentToTreatDRIV, LinearIntentToTreatDRIV,)


class TestDRIV(unittest.TestCase):
    def test_cate_api(self):
        def prel_model_effect():
            return _DummyCATE()

        def const_marg_eff_shape(n, d_x, binary_T):
            return (n if d_x else 1,) + ((1,) if binary_T else ())

        def marg_eff_shape(n, binary_T):
            return (n,) + ((1,) if binary_T else ())

        def eff_shape(n, d_x):
            return (n if d_x else 1,)

        n = 1000
        y = np.random.normal(size=(n,))

        for d_w in [None, 10]:
            if d_w is None:
                W = None
            else:
                W = np.random.normal(size=(n, d_w))
            for d_x in [None, 3]:
                if d_x is None:
                    X = None
                else:
                    X = np.random.normal(size=(n, d_x))
                for binary_T in [True, False]:
                    if binary_T:
                        T = np.random.choice(["a", "b"], size=(n,))
                    else:
                        T = np.random.normal(size=(n,))
                    for binary_Z in [True, False]:
                        if binary_Z:
                            Z = np.random.choice(["c", "d"], size=(n,))
                        else:
                            Z = np.random.normal(size=(n,))
                        for projection in [True, False]:
                            for featurizer in [
                                None,
                                PolynomialFeatures(degree=2, include_bias=False),
                            ]:
                                est_list = [
                                    DRIV(
                                        prel_model_effect=prel_model_effect(),
                                        model_final=StatsModelsLinearRegression(
                                            fit_intercept=False
                                        ),
                                        fit_cate_intercept=True,
                                        projection=projection,
                                        discrete_instrument=binary_Z,
                                        discrete_treatment=binary_T,
                                        featurizer=featurizer,
                                    ),
                                    LinearDRIV(
                                        prel_model_effect=prel_model_effect(),
                                        fit_cate_intercept=True,
                                        projection=projection,
                                        discrete_instrument=binary_Z,
                                        discrete_treatment=binary_T,
                                        featurizer=featurizer,
                                    ),
                                    SparseLinearDRIV(
                                        prel_model_effect=prel_model_effect(),
                                        fit_cate_intercept=True,
                                        projection=projection,
                                        discrete_instrument=binary_Z,
                                        discrete_treatment=binary_T,
                                        featurizer=featurizer,
                                    ),
                                    ForestDRIV(
                                        prel_model_effect=prel_model_effect(),
                                        projection=projection,
                                        discrete_instrument=binary_Z,
                                        discrete_treatment=binary_T,
                                        featurizer=featurizer,
                                    ),
                                ]

                                if X is None:
                                    est_list = est_list[:-1]

                                if binary_T and binary_Z:
                                    est_list += [
                                        IntentToTreatDRIV(
                                            flexible_model_effect=StatsModelsLinearRegression(
                                                fit_intercept=False
                                            ),
                                            fit_cate_intercept=True,
                                            featurizer=featurizer,
                                        ),
                                        LinearIntentToTreatDRIV(
                                            flexible_model_effect=StatsModelsLinearRegression(
                                                fit_intercept=False
                                            ),
                                            featurizer=featurizer,
                                        ),
                                    ]

                                for est in est_list:
                                    with self.subTest(d_w=d_w, d_x=d_x, binary_T=binary_T, binary_Z=binary_Z,
                                                      projection=projection, featurizer=featurizer,
                                                      est=est):

                                        # ensure we can serialize unfit estimator
                                        pickle.dumps(est)

                                        est.fit(y, T, Z=Z, X=X, W=W)

                                        # ensure we can serialize fit estimator
                                        pickle.dumps(est)

                                        # expected effect size
                                        const_marginal_effect_shape = const_marg_eff_shape(n, d_x, binary_T)
                                        marginal_effect_shape = marg_eff_shape(n, binary_T)
                                        effect_shape = eff_shape(n, d_x)
                                        # test effect
                                        const_marg_eff = est.const_marginal_effect(X)
                                        self.assertEqual(shape(const_marg_eff), const_marginal_effect_shape)
                                        marg_eff = est.marginal_effect(T, X)
                                        self.assertEqual(shape(marg_eff), marginal_effect_shape)
                                        T0 = "a" if binary_T else 0
                                        T1 = "b" if binary_T else 1
                                        eff = est.effect(X, T0=T0, T1=T1)
                                        self.assertEqual(shape(eff), effect_shape)

                                        # test inference
                                        const_marg_eff_int = est.const_marginal_effect_interval(X)
                                        marg_eff_int = est.marginal_effect_interval(T, X)
                                        eff_int = est.effect_interval(X, T0=T0, T1=T1)
                                        self.assertEqual(shape(const_marg_eff_int), (2,) + const_marginal_effect_shape)
                                        self.assertEqual(shape(marg_eff_int), (2,) + marginal_effect_shape)
                                        self.assertEqual(shape(eff_int), (2,) + effect_shape)

                                        # test can run score
                                        est.score(y, T, Z=Z, X=X, W=W)

                                        if X is not None:
                                            # test cate_feature_names
                                            expect_feat_len = featurizer.fit(
                                                X).n_output_features_ if featurizer else d_x
                                            self.assertEqual(len(est.cate_feature_names()), expect_feat_len)

                                            # test can run shap values
                                            shap_values = est.shap_values(X[:10])

    def test_accuracy(self):
        np.random.seed(123)
        # helper function

        def prel_model_effect():
            return _DummyCATE()

        # dgp (binary T, binary Z)
        def dgp(n, p, true_fn):
            X = np.random.normal(0, 1, size=(n, p))
            Z = np.random.binomial(1, 0.5, size=(n,))
            nu = np.random.uniform(0, 10, size=(n,))
            coef_Z = 0.8
            C = np.random.binomial(
                1, coef_Z * special.expit(0.4 * X[:, 0] + nu)
            )  # Compliers when recomended
            C0 = np.random.binomial(
                1, 0.06 * np.ones(X.shape[0])
            )  # Non-compliers when not recommended
            T = C * Z + C0 * (1 - Z)
            y = true_fn(X) * T + 2 * nu + 5 * (X[:, 3] > 0) + 0.1 * np.random.uniform(0, 1, size=(n,))
            return y, T, Z, X

        ests_list = [LinearIntentToTreatDRIV(
            flexible_model_effect=StatsModelsLinearRegression(fit_intercept=False), fit_cate_intercept=True
        ), LinearDRIV(
            prel_model_effect=prel_model_effect(),
            fit_cate_intercept=True,
            projection=False,
            discrete_instrument=True,
            discrete_treatment=True,
        )]

        # no heterogeneity
        n = 1000
        p = 10
        true_ate = 10

        def true_fn(X):
            return true_ate
        y, T, Z, X = dgp(n, p, true_fn)
        for est in ests_list:
            with self.subTest(est=est):
                est.fit(y, T, Z=Z, X=None, W=X, inference="auto")
                ate_lb, ate_ub = est.ate_interval()
                np.testing.assert_array_less(ate_lb, true_ate)
                np.testing.assert_array_less(true_ate, ate_ub)

        # with heterogeneity
        true_coef = 10

        def true_fn(X):
            return true_coef * X[:, 0]
        y, T, Z, X = dgp(n, p, true_fn)
        for est in ests_list:
            with self.subTest(est=est):
                est.fit(y, T, Z=Z, X=X[:, [0]], W=X[:, 1:], inference="auto")
                coef_lb, coef_ub = est.coef__interval()
                intercept_lb, intercept_ub = est.intercept__interval(alpha=0.05)
                np.testing.assert_array_less(coef_lb, true_coef)
                np.testing.assert_array_less(true_coef, coef_ub)
                np.testing.assert_array_less(intercept_lb, 0)
                np.testing.assert_array_less(0, intercept_ub)
