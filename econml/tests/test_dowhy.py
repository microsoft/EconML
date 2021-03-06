# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import unittest
from econml.dml import LinearDML, CausalForestDML
from econml.orf import DROrthoForest
from econml.dr import DRLearner
from econml.metalearners import XLearner
from econml.iv.dml import DMLATEIV
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso


class TestDowhy(unittest.TestCase):

    def _get_data(self):
        X = np.random.normal(0, 1, size=(500, 5))
        T = np.random.binomial(1, .5, size=(500,))
        Y = np.random.normal(0, 1, size=(500,))
        Z = np.random.normal(0, 1, size=(500,))
        return Y, T, X[:, [0]], X[:, 1:], Z

    def test_dowhy(self):
        def reg():
            return LinearRegression()

        def clf():
            return LogisticRegression()

        Y, T, X, W, Z = self._get_data()
        # test at least one estimator from each category
        models = {"dml": LinearDML(model_y=reg(), model_t=clf(), discrete_treatment=True,
                                   linear_first_stages=False),
                  "dr": DRLearner(model_propensity=clf(), model_regression=reg(),
                                  model_final=reg()),
                  "xlearner": XLearner(models=reg(), cate_models=reg(), propensity_model=clf()),
                  "cfdml": CausalForestDML(model_y=reg(), model_t=clf(), discrete_treatment=True),
                  "orf": DROrthoForest(n_trees=10, propensity_model=clf(), model_Y=reg()),
                  "dmlateiv": DMLATEIV(model_Y_W=reg(),
                                       model_T_W=clf(),
                                       model_Z_W=reg(),
                                       discrete_treatment=True,
                                       discrete_instrument=False)}
        for name, model in models.items():
            with self.subTest(name=name):
                est = model
                if name == "xlearner":
                    est_dowhy = est.dowhy.fit(Y, T, X=np.hstack((X, W)), W=None)
                elif name == "dmlateiv":
                    est_dowhy = est.dowhy.fit(Y, T, W=W, Z=Z)
                else:
                    est_dowhy = est.dowhy.fit(Y, T, X=X, W=W)
                # test causal graph
                est_dowhy.view_model()
                # test refutation estimate
                est_dowhy.refute_estimate(method_name="random_common_cause")
                if name != "orf":
                    est_dowhy.refute_estimate(method_name="add_unobserved_common_cause",
                                              confounders_effect_on_treatment="binary_flip",
                                              confounders_effect_on_outcome="linear",
                                              effect_strength_on_treatment=0.1,
                                              effect_strength_on_outcome=0.1,)
                    est_dowhy.refute_estimate(method_name="placebo_treatment_refuter", placebo_type="permute",
                                              num_simulations=3)
                    est_dowhy.refute_estimate(method_name="data_subset_refuter", subset_fraction=0.8,
                                              num_simulations=3)
