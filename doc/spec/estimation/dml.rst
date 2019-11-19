==================================
Orthogonal/Double Machine Learning
==================================

What is it?
==================================

Double Machine Learning is a method for estimating (heterogeneous) treatment effects when
all potential confounders/controls (factors that simultaneously had a direct effect on the treatment decision in the
collected data and the observed outcome) are observed, but are either too many (high-dimensional) for
classical statistical approaches to be applicable or their effect on 
the treatment and outcome cannot be satisfactorily modeled by parametric functions (non-parametric).
Both of these latter problems can be addressed via machine learning techniques (see e.g. [Chernozhukov2016]_).

The method reduces the problem to first estimating *two predictive tasks*: 
    
    1) predicting the outcome from the controls,
    2) predicting the treatment from the controls;

Then the method combines these two predictive models in a final stage estimation so as to create a
model of the heterogeneous treatment efffect. The approach allows for *arbitrary Machine Learning algorithms* to be
used for the two predictive tasks, while maintaining many favorable statistical properties related to the final
model (e.g. small mean squared error, asymptotic normality, construction of confidence intervals).

Our package offers several variants for the final model estimation. Many of these variants also
provide *valid inference* (confidence interval construction) for measuring the uncertainty of the learned model.


What are the relevant estimator classes?
========================================

This section describes the methodology implemented in the classes, :py:class:`~econml._rlearner._RLearner`,
:py:class:`~econml.dml.DMLCateEstimator`, :py:class:`~econml.dml.LinearDMLCateEstimator`,
:py:class:`~econml.dml.SparseLinearDMLCateEstimator`, :py:class:`~econml.dml.KernelDMLCateEstimator`. Click on each of these links for a detailed module documentation and input parameters of each class.


When should you use it?
==================================

Suppose you have observational (or experimental from an A/B test) historical data, where some treatment(s)/intervention(s)/action(s) 
:math:`T` were chosen and some outcome(s) :math:`Y` were observed and all the variables :math:`W` that could have
potentially gone into the choice of :math:`T`, and simultaneously could have had a direct effect on the outcome :math:`Y` (aka controls or confounders) are also recorder in the dataset.

If your goal is to understand what was the effect of the treatment on the outcome as a function of a set of observable
characteristics :math:`X` of the treated samples, then one can use this method. For instance call::

    from econml.dml import LinearDMLCateEstimator
    est = LinearDMLCateEstimator()
    est.fit(y, T, X, W)
    est.const_marginal_effect(X)

This way an optimal treatment policy can be learned, by simply inspecting for which :math:`X` the effect was positive.

Most of the methods provided make a parametric form assumption on the heterogeneous treatment effect model (e.g.
linear on some pre-defined; potentially high-dimensional; featurization). For fullly non-parametric heterogeneous treatment effect models, check out the :ref:`Orthogonal Random Forest User Guide <orthoforestuserguide>` or, if your treatment is categorical, then also check the :ref:`Meta Learners User Guide <metalearnersuserguide>`.


Overview of Formal Methodology
==================================

The model makes the following structural equation assumptions on the data generating process.

.. math::

    Y =~& \theta(X) \cdot T + g(X, W) + \epsilon ~~~&~~~ \E[\epsilon | X, W] = 0 \\ 
    T =~& f(X, W) + \eta & \E[\eta \mid X, W] = 0 \\
    ~& \E[\eta \cdot \epsilon | X, W] = 0

What is particularly attractive about DML is that it makes no further structural assumptions on :math:`g` and :math:`f` and estimates them 
non-parametrically using arbitrary non-parametric Machine Learning methods. Our goal is to estimate
the constant marginal CATE :math:`\theta(X)`.

The idea to estimate :math:`\theta(X)` is as follows: we can re-write the structural equations as

.. math::

    Y - \E[Y | X, W]
    = \theta(X) \cdot (T - \E[T | X, W]) + \epsilon

Thus if one can estimate the conditional expectation functions (both of which are non-parametric regression tasks):

.. math::

    q(X, W) =~& \E[Y | X, W]\\
    f(X, W) =~& \E[T | X, W]

Then we can compute the residuals:

.. math::

    \tilde{Y} =~& Y - q(X, W)\\
    \tilde{T} =~& T - f(X, W) = \eta

which are subsequently related by the equation:

.. math::

    \tilde{Y} = \theta(X) \cdot \tilde{T} + \epsilon

Subsequently, since :math:`\E[\epsilon \cdot \eta | X]=0`, estimating :math:`\theta(X)` is a final regression problem, regressing :math:`\tilde{Y}` on :math:`X, \tilde{T}` (albeit over models that are linear in :math:`\tilde{T}`), i.e.

.. math::
    :nowrap:

    \begin{equation}
    \hat{\theta} = \arg\min_{\theta \in \Theta} \E_n\left[ (\tilde{Y} - \theta(X)\cdot \tilde{T})^2 \right]
    \end{equation}

This approach has been analyzed in multiple papers in the literature, for different model classes :math:`\Theta`.
[Chernozhukov2016]_ consider the case where :math:`\theta(X)` is a constant (average treatment effect) or a low dimensional
linear function,
[Nie2017]_ consider the case where :math:`\theta(X)` falls in a Reproducing Kernel Hilbert Space (RKHS),
[Chernozhukov2017]_, [Chernozhukov2018]_ consider the case of a high dimensional sparse linear space, where :math:`\theta(X)=\langle \theta, \phi(X)\rangle` for some known high-dimensional feature mapping and where :math:`\theta_0` has very few non-zero entries (sparse), [Athey2019]_ (among other results) consider the case where :math:`\theta(X)` is a non-parametric lipschitz function and use random forest models to fit the function, [Foster2019]_ allow for arbitrary models :math:`\theta(X)` and give results based on sample complexity measures of the model space (e.g. Rademacher complexity, metric entropy).


The main advantage of DML is that if one makes parametric assumptions on :math:`\theta(X)`, then one achieves fast estimation rates and 
asymptotic normality on the second stage estimate :math:`\hat{\theta}`, even if the first stage estimates on :math:`q(X, W)` 
and :math:`f(X, W)` are only :math:`n^{1/4}` consistent, in terms of RMSE. For this theorem to hold, the nuisance
estimates need to be fitted in a cross-fitting manner (see :py:class:`~econml._ortho_learner._OrthoLearner`).
The latter robustness property follows from the fact that the moment equations that correspond to the final 
least squares estimation (i.e. the gradient of the squared loss), satisfy a Neyman orthogonality condition with respect to the
nuisance parameters :math:`q, f`. For a more detailed exposition of how Neyman orthogonality 
leads to robustness we refer the reader to [Chernozhukov2016]_, [Mackey2017]_, [Nie2017]_, [Chernozhukov2017]_,
[Chernozhukov2018]_, [Foster2019]_. 

Class Hierarchy Structure
==================================

In this library we implement variants of several of the approaches mentioned in the last section. The hierarchy
structure of the implemented CATE estimators is as follows.

    .. inheritance-diagram:: econml.dml.LinearDMLCateEstimator econml.dml.SparseLinearDMLCateEstimator econml.dml.KernelDMLCateEstimator
        :parts: 1
        :private-bases:
        :top-classes: econml._rlearner._RLearner, econml.cate_estimator.StatsModelsCateEstimatorMixin

Below we give a brief description of each of these classes:

    * *DMLCateEstimator.* The class :py:class:`~econml.dml.DMLCateEstimator` assumes that the effect model for each outcome :math:`i` and treatment :math:`j` is linear, i.e. takes the form :math:`\theta_{ij}(X)=\langle \theta_{ij}, \phi(X)\rangle`, and allows for any arbitrary scikit-learn linear estimator to be defined as the final stage (e.g.    
      :py:class:`~sklearn.linear_model.ElasticNet`, :py:class:`~sklearn.linear_model.Lasso`, :py:class:`~sklearn.linear_model.LinearRegression` and their multi-task variations in the case where we have mulitple outcomes, i.e. :math:`Y` is a vector). The final linear model will be fitted on features that are derived by the Kronecker-product
      of the vectors :math:`T` and :math:`\phi(X)`, i.e. :math:`\tilde{T}\otimes \phi(X) = \mathtt{vec}(\tilde{T}\cdot \phi(X)^T)`. This regression will estimate the coefficients :math:`\theta_{ijk}` 
      for each outcome :math:`i`, treatment :math:`j` and feature :math:`k`. The final model is minimizing a regularized empirical square loss of the form:
      
      .. math::
    
            \hat{\alpha} = \arg\min_{\alpha} \E_n\left[ (\tilde{Y} - \Theta \cdot \tilde{T}\otimes \phi(X) \right] + \lambda R(\Theta)

      for some strongly convex regularizer :math:`R`, where :math:`\Theta` is the parameter matrix of dimensions (number of outcomes, number of treatments * number of features). For instance, if :math:`Y` is single dimensional and the lasso is used as model final, i.e.::

        from sklearn.linear_model import LassoCV
        from sklearn.ensemble import GradientBoostingRegressor
        est = DMLCateEstimator(model_y=GradientBoostingRegressor(),
                               model_t=GradientBoostingRegressor(),    
                               model_final=LassoCV(),
                               featurizer=PolynomialFeatures(degree=1, include_bias=True))

      then :math:`R(\Theta) =\|\Theta\|_1`, 
      if ElasticNet is used as model final, i.e.::

        from sklearn.linear_model import ElasticNetCV
        from sklearn.ensemble import GradientBoostingRegressor
        est = DMLCateEstimator(model_y=GradientBoostingRegressor(),
                               model_t=GradientBoostingRegressor(),
                               model_final=ElasticNetCV(),
                               featurizer=PolynomialFeatures(degree=1, include_bias=True))

      then :math:`R(\Theta)=\kappa \|\Theta\|_2 + (1-\kappa)\|\Theta\|_1`. For multi-dimensional :math:`Y`, 
      one can impose several extensions to the matrix of parameters :math:`\alpha`, such as the one corresponding to the MultiTask Lasso 
      :math:`\sum_{j} \sum_{i} \theta_{ij}^2` or MultiTask ElasticNet or nuclear norm regularization  [Jaggi2010]_, which enforces low-rank 
      constraints on the matrix :math:`\Theta`.
      This essentially implements the techniques analyzed in [Chernozhukov2016]_, [Nie2017]_, [Chernozhukov2017]_, [Chernozhukov2018]_
        
        - *LinearDMLCateEstimator.* The child class  :py:class:`~econml.dml.LinearDMLCateEstimator`, uses an unregularized final linear model and  
          essentially works only when the feature vector :math:`\phi(X)` is low dimensional. Given that it is an unregularized
          low dimensional final model, this class also offers confidence intervals via asymptotic normality 
          arguments. This is achieved by essentially using the :py:class:`~econml.utilities.StatsModelsLinearRegression`
          (which is an extension of the scikit-learn LinearRegression estimator, that also supports inference
          functionalities) as a final model. The theoretical foundations of this class essentially follow the arguments in [Chernozhukov2016]_.
          For instance, to get confidence intervals on the effect of going
          from any treatment T0 to any other treatment T1, one can simply call::

            est = LinearDMLCateEstimator()
            est.fit(y, T, X, W, inference='statsmodels')
            point = est.effect(X, T0=T0, T1=T1)
            lb, ub = est.effect_interval(X, T0=T0, T1=T1, alpha=0.05)

          One could also construct bootstrap based confidence intervals by setting `inference='bootstrap'`.

        - *SparseLinearDMLCateEstimator.* The child class :py:class:`~econml.dml.SparseLinearDMLCateEstimator`, uses an :math:`\ell_1`-regularized final    
          model. In particular, it uses an implementation of the DebiasedLasso algorithm [Buhlmann2011]_ (see :py:class:`~econml.sklearn_extensions.linear_model.DebiasedLasso`). Using the asymptotic normality properties
          of the debiased lasso, this class also offers asymptotically normal based confidence intervals.
          The theoretical foundations of this class essentially follow the arguments in [Chernozhukov2017]_, [Chernozhukov2018]_.
          For instance, to get confidence intervals on the effect of going
          from any treatment T0 to any other treatment T1, one can simply call::
            
            est = SparseLinearDMLCateEstimator()
            est.fit(y, T, X, W, inference='debiasedlasso')
            point = est.effect(X, T0=T0, T1=T1)
            lb, ub = est.effect_interval(X, T0=T0, T1=T1, alpha=0.05)

        - *KernelDMLCateEstimator.* The child class :py:class:`~econml.dml.KernelDMLCateEstimator` performs a variant of the RKHS approach proposed in 
          [Nie2017]_. It approximates any function in the RKHS by creating random Fourier features. Then runs a ElasticNet
          regularized final model. Thus it approximately implements the results of [Nie2017], via the random fourier feature
          approximate representation of functions in the RKHS. Moreover, given that we use Random Fourier Features this class
          asssumes an RBF kernel.
    
    - *_RLearner.* The internal private class :py:class:`~econml._rlearner._RLearner` is a parent of the :py:class:`~econml.dml.DMLCateEstimator`
      and allows the user to specify any way of fitting a final model that takes as input the residual :math:`\tilde{T}`,
      the features :math:`X` and predicts the residual :math:`\tilde{Y}`. Moreover, the nuisance models take as input
      :math:`X` and :math:`W` and predict :math:`T` and :math:`Y` respectively. Since these models take non-standard
      input variables, one cannot use out-of-the-box scikit-learn estimators as inputs to this class. Hence, it is
      slightly more cumbersome to use, which is the reason why we designated it as private. However, if one wants to
      fit for instance a neural net model for :math:`\theta(X)`, then this class can be used (see the implementation
      of the :py:class:`~econml.dml.DMLCateEstimator` of how to wrap sklearn estimators and pass them as inputs to the
      :py:class:`~econml._rlearner._RLearner`. This private class essentially follows the general arguments and
      terminology of the RLearner presented in [Nie2017]_, and allows for the full flexibility of the final model
      estimation that is presented in [Foster2019]_.



Usage FAQs
==========

- **What if I want confidence intervals?**

    For valid confidence intervals use the :py:class:`~econml.dml.LinearDMLCateEstimator` if the number of features :math:`X`,
    that you want to use for heterogeneity are small compared to the number of samples that you have. If the number of
    features is comparable to the number of samples, then use :py:class:`~econml.dml.SparseLinearDMLCateEstimator`.
    e.g.::

        from econml.dml import LinearDMLCateEstimator
        est = LinearDMLCateEstimator()
        est.fit(y, T, X, W, inference='statsmodels')
        lb, ub = est.const_marginal_effect_interval(X, alpha=.05)
        lb, ub = est.coef__interval(alpha=.05)
        lb, ub = est.effect_interval(X, T0=T0, T1=T1, alpha=.05)

- **Why not just run a simple big linear regression with all the treatments, features and controls?**

    If you want to estimate an average treatment effect with accompanied confidence intervals then one
    potential approach one could take is simply run a big linear regression, regressing :math:`Y` on
    :math:`T, X, W` and then looking at the coefficient associated with the :math:`T` variable and
    the corresponding confidence interval (e.g. using statistical packages like
    :py:class:`~statsmodels.api.OLS`). However, this will not work if:

        1) The number of control variables :math:`X, W` that you have is large and comparable
        to the number of samples. This could for instance arise if one wants to control for
        unit fixed effects, in which case the number of controls is at least the number of units.
        In such high-dimensional settings, ordinary least squares (OLS) is not a reasonable approach.
        Typically, the covariance matrix of the controls, will be ill-posed and the inference
        will be invalid. The DML method bypasses this by using ML approaches to appropriately
        regularize the estimation and provide better models on how the controls affect the outcome,
        given the number of samples that you have.

        2) The effect of the variables :math:`X, W` on the outcome :math:`Y` is not linear.
        In this case, OLS will not provide a consistent model, which could lead to heavily
        biased effect results. The DML approach, when combined with non-linear first stage
        models, like Random Forests or Gradient Boosted Forests, can capture such non-linearities
        and provide unbiased estimates of the effect of :math:`T` on :math:`Y`. Moreover,
        it does so in a manner that is robust to the estimation mistakes that these ML algorithms
        might be making.
    
    Moreover, one may typically want to estimate treatment effect hetergoeneity,
    which the above OLS approach wouldn't provide. One potential way of providing such heterogeneity
    is to include product features of the form :math:`X\cdot T` in the OLS model. However, then
    one faces again the same problems as above:

        1) If effect heterogeneity does not have a linear form, then this approach is not valid.
        One might want to then create more complex featurization, in which case the problem could
        become too high-dimensional for OLS. Our :py:class:`~econml.dml.SparseLinearDMLCateEstimator`
        can handle such settings via the use of the debiased Lasso. Also see the :ref:`Orthogonal Random Forest User Guide <orthoforestuserguide>` or, if your treatment is categorical, then also check the :ref:`Meta Learners User Guide <metalearnersuserguide>`, if you want even more flexible CATE models.

        2) If the number of features :math:`X` is comparable to the number of samples, then even
        with a linear model, the OLS approach is not feasible or has very small statistical power.


- **What if I have no idea how heterogeneity looks like?**

    Either use a flexible featurizer, e.g. a polynomial featurizer with many degrees and use
    the :py:class:`~econml.dml.SparseLinearDMLCateEstimator`::

        from econml.dml import SparseLinearDMLCateEstimator
        from sklearn.preprocessing import PolynomialFeatures
        est = SparseLinearDMLCateEstimator(featurizer=PolynomialFeatures(degree=4))
        est.fit(y, T, X, W, inference='debiasedlasso')
        lb, ub = est.const_marginal_effect_interval(X, alpha=.05)
    
    Alternatively, if your number of features :math:`X` is small compared to your number of samples, then
    you can look into the check out the :ref:`Orthogonal Random Forest User Guide <orthoforestuserguide>` or the
    :ref:`Meta Learners User Guide <metalearnersuserguide>`.

- **What if I have too many features that can create heterogeneity?**

    Use the :py:class:`~econml.dml.SparseLinearDMLCateEstimator` (see above).

- **What if I have too many features I want to control for?**

    Use first stage models that work well with high dimensional features. For instance, the Lasso or the 
    ElasticNet or gradient boosted forests are all good options (the latter allows for 
    non-linearities in the model but can typically handle fewer features than the former), e.g.::

        from econml.dml import SparseLinearDMLCateEstimator
        from sklearn.linear_model import LassoCV, ElasticNetCV
        from sklearn.ensemble import GradientBoostingRegressor
        est = SparseLinearDMLCateEstimator(model_y=LassoCV(), model_t=LassoCV())
        est = SparseLinearDMLCateEstimator(model_y=ElasticNetCV(), model_t=ElasticNetCV())
        est = SparseLinearDMLCateEstimator(model_y=GradientBoostingRegressor(),
                                           model_t=GradientBoostingRegressor())
    
    The confidence intervals will still be valid, provided that these first stage models achieve small
    mean squared error.

- **What should I use for first stage estimation?**

    See above. The first stage problems are pure predictive tasks, so any ML approach that is relevant for your
    prediction problem is good.

- **How do I select the hyperparameters of the first stage models?**

    You can use cross-validated models that automatically choose the hyperparameters, e.g. the
    :py:class:`~sklearn.linear_model.LassoCV` instead of the :py:class:`~sklearn.linear_model.Lasso`. Similarly,
    for forest based estimators you can wrap them with a grid search CV, :py:class:`~sklearn.model_selection.GridSearchCV`, e.g.::

        from econml.dml import DMLCateEstimator
        from sklearn.model_selection import GridSearchCV
        first_stage = lambda: GridSearchCV(
                        estimator=RandomForestRegressor(),
                        param_grid={
                                'max_depth': [3, None],
                                'n_estimators': (10, 30, 50, 100, 200, 400, 600, 800, 1000),
                                'max_features': (2,4,6)
                            }, cv=10, n_jobs=-1, scoring='neg_mean_squared_error'
                        )
        est = SparseLinearDMLCateEstimator(model_y=first_stage(), model_t=first_stage())

- **How do I select the hyperparameters of the final model (if any)?**

    You can use cross-validated classes for the final model too. Our default debiased lasso performs cross validation
    for hyperparameter selection. For custom final models you can also use CV versions, e.g.::

        from econml.dml import DMLCateEstimator
        from sklearn.linear_model import ElasticNetCV
        from sklearn.ensemble import GradientBoostingRegressor
        est = DMLCateEstimator(model_y=GradientBoostingRegressor(),
                               model_t=GradientBoostingRegressor(),
                               model_final=ElasticNetCV())
        est.fit(y, T, X, W)
        point = est.const_marginal_effect(X)
        point = est.effect(X, T0=T0, T1=T1)

- **What if I have many treatments?**

    The method is going to assume that each of these treatments enters linearly into the model. So it cannot capture complementarities or substitutabilities
    of the different treatments. For that you can also create composite treatments that look like the product 
    of two base treatments. Then these product will enter in the model and an effect for that product will be estimated.
    This effect will be the substitute/complement effect of both treatments being present, i.e.::

        from econml.dml import LinearDMLCateEstimator
        from sklearn.preprocessing import PolynomialFeatures
        poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        est = LinearDMLCateEstimator()
        T_composite = poly.fit_transform(T)
        est.fit(y, T_composite, X, W)
        point = est.const_marginal_effect(X)
        est.effect(X, T0=poly.transform(T0), T1=poly.transform(T1)) 

    If your treatments are too many, then you can use the :py:class:`~econml.dml.SparseLinearDMLCateEstimator`. However,
    this method will essentially impose a regularization that only a small subset of them has any effect.

- **What if my treatments are continuous and don't have a linear effect on the outcome?**

    You can create composite treatments and add them as extra treatment variables (see above). This would require
    imposing a particular form of non-linearity.

- **What if my treatment is categorical/binary?**

    You can simply set `discrete_treatment=True` in the parameters of the class. Then use any classifier for 
    `model_t`, that has a `predict_proba` method::

        from econml.dml import LinearDMLCateEstimator
        from sklearn.linear_model import LogisticRegressionCV
        est = LinearDMLCateEstimator(model_t=LogisticRegressionCV(), discrete_treatment=True)
        est.fit(y, T, X, W)
        point = est.const_marginal_effect(X)
        est.effect(X, T0=T0, T1=T1)

- **How can I assess the performance of the CATE model?**

    Each of the DML classes have an attribute `score_` after they are fitted. So one can access that
    attribute and compare the performance accross different modeling parameters (lower score is better)::

        from econml.dml import DMLCateEstimator
        from sklearn.linear_model import ElasticNetCV
        from sklearn.ensemble import RandomForestRegressor
        est = DMLCateEstimator(model_y=RandomForestRegressor(oob_score=True),
                            model_t=RandomForestRegressor(oob_score=True),
                            model_final=ElasticNetCV(), featurizer=PolynomialFeatures(degree=1))
        est.fit(Y, T, X, W)
        est.score_

    This essentially measures the score based on the final stage loss. Moreover, one can assess the out-of-sample score by calling the `score` method on a separate validation sample that was not
    used for training::

        est.score(Y_val, T_val, X_val, W_val)

    Moreover, one can independently check the goodness of fit of the fitted first stage models by
    inspecting the fitted models. You can access the list of fitted first stage models (one for each
    fold of the crossfitting structure) via the methods: `models_t` and `models_y`. Then if those models
    also have a score associated attribute, that can be used as an indicator of performance of the first
    stage. For instance in the case of Random Forest first stages as in the above example, if the `oob_score`
    is set to `True`, then the estimator has a post-fit measure of performance::

        [mdl.oob_score_ for mdl in est.models_y]

    If one uses cross-validated estimators as first stages, then model selection for the first stage models
    is performed automatically.

- **How should I set the parameter `n_splits`?**

    This parameter defines the number of data partitions to create in order to fit the first stages in a
    crossfittin manner (see :py:class:`~econml._ortho_learner._OrthoLearner`). The default is 2, which
    is the minimal. However, larger values like 5 or 6 can lead to greater statistical stability of the method,
    especially if the number of samples is small. So we advise that for small datasets, one should raise this
    value. This can increase the computational cost as more first stage models are being fitted.


Usage Examples
==================================


Single Outcome, Single Treatment
---------------------------------------------------

We consider some example use cases of the library when :math:`Y` and :math:`T` are :math:`1`-dimensional.

.. rubric:: Random Forest First Stages

A classical non-parametric regressor for the first stage estimates is a Random Forest. Using RandomForests in our API is as simple as:

.. code-block:: python3
    :caption: Random Forest First Stage

    from econml.dml import LinearDMLCateEstimator
    from sklearn.ensemble import RandomForestRegressor
    est = LinearDMLCateEstimator(model_y=RandomForestRegressor(),
                                model_t=RandomForestRegressor())
    est.fit(y, T, X, W, inference='statsmodels')
    pnt_effect = est.const_marginal_effect(X)
    lb_effect, ub_effect = est.const_marginal_effect_interval(X, alpha=.05)
    pnt_coef = est.coef_
    lb_coef, ub_coef = est.coef__interval(alpha=.05)


.. rubric:: Polynomial Features for Heterogeneity

Suppose that we believe that the treatment effect is a polynomial of :math:`X`, i.e.

.. math::
    
    Y = (\alpha_0 + \alpha_1 X + \alpha_2 X^2 + \ldots) \cdot T + g(X, W, \epsilon)

Then we can estimate the coefficients :math:`\alpha_i` by running:

.. code-block:: python3
    :caption: Polynomial Second Stage Features

    from econml.dml import LinearDMLCateEstimator
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import PolynomialFeatures
    est = LinearDMLCateEstimator(model_y=RandomForestRegressor(),
                                model_t=RandomForestRegressor(),
                                featurizer=PolynomialFeatures(degree=4, include_bias=True))
    est.fit(y, T, X, W)

    # To get the coefficients of the polynomial fitted in the final stage we can
    # access the `coef_` attribute of the fitted second stage model. This would 
    # return the coefficients in front of each term in the vector T⊗ϕ(X).
    est.coef_


.. rubric:: Fixed Effects

To add fixed effect heterogeneity, we can create one-hot encodings of the id, which is assumed to be part of the input:

.. code-block:: python3
    :caption: Fixed Effect Heterogeneity

    from econml.dml import LinearDMLCateEstimator
    from sklearn.preprocessing import OneHotEncoder
    # removing one id to avoid colinearity, as is standard for fixed effects
    X_oh = OneHotEncoder(sparse=False).fit_transform(X)[:, 1:]

    est = LinearDMLCateEstimator(model_y=RandomForestRegressor(),
                                model_t=RandomForestRegressor())
    est.fit(y, T, X_oh, W)
    # The latter will fit a model for θ(x) of the form ̂α_0 + ̂α_1 𝟙{id=1} + ̂α_2 𝟙{id=2} + ...
    # The vector of α can be extracted as follows
    est.coef_

.. rubric:: Custom Features

One can also define a custom featurizer, as long as it supports the fit\_transform interface of sklearn.

.. code-block:: python3
    :caption: Custom Featurizer

    from sklearn.ensemble import RandomForestRegressor
    class LogFeatures(object):
        """Augments the features with logarithmic features and returns the augmented structure"""
        def fit(self, X):
            return self
        def transform(self, X):
            return np.concatenate((X, np.log(1+X)), axis=1)
        def fit_transform(self, X):
            return self.fit(X).transform(X)

    est = LinearDMLCateEstimator(model_y=RandomForestRegressor(),
                                model_t=RandomForestRegressor(),
                                featurizer=LogFeatures())
    est.fit(y, T, X, W)

We can even create a Pipeline or Union of featurizers that will apply multiply featurizations, e.g. first creating log features and then adding polynomials of them:

.. code-block:: python3
    :caption: Pipeline Featurizer

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import PolynomialFeatures
    est = DMLCateEstimator(model_y=RandomForestRegressor(), 
                            model_t=RandomForestRegressor(),
                            featurizer=Pipeline([('log', LogFeatures()), 
                                                 ('poly', PolynomialFeatures(degree=3))]))
    est.fit(y, T, X, W)


Single Outcome, Multiple Treatments
------------------------------------------------------

Suppose that we believed that our treatment was affecting the outcome in a non-linear manner. 
Then we could expand the treatment vector to contain also polynomial features:

.. code-block:: python3
    :caption: Polynomial Treatments

    import numpy as np
    est = LinearDMLCateEstimator()
    est.fit(y, np.concatenate((T, T**2), axis=1), X, W)

Multiple Outcome, Multiple Treatments
--------------------------------------------------------

In settings like demand estimation, we might want to fit the demand of multiple products as a function of the price of each one of them, i.e. fit the matrix of cross price elasticities. The latter can be done, by simply setting :math:`Y` to be the vector of demands and :math:`T` to be the vector of prices. Then we can recover the 
matrix of cross price elasticities as:

.. code-block:: python3
    :caption: Cross-Price Elasticities

    est = LinearDMLCateEstimator(model_y=MultiTaskElasticNet(alpha=0.1),
                                 model_t=MultiTaskElasticNet(alpha=0.1))
    est.fit(Y, T, None, W)

    # a_hat[i,j] contains the elasticity of the demand of product i on the price of product j
    a_hat = est.const_marginal_effect()

If we have too many products then the cross-price elasticity matrix contains many parameters and we need
to regularize. Given that we want to estimate a matrix, it makes sense in this application to consider
the case where this matrix has low rank: all the products can be embedded in some low dimensional feature
space and the cross-price elasticities is a linear function of these low dimensional embeddings. This corresponds
to well-studied latent factor models in pricing. Our framework can easily handle this by using 
a nuclear norm regularized multi-task regression in the final stage. For instance the 
lightning package implements such a class:

.. code-block:: python3
    :caption: Cross-Price Elasticities with Low-Rank Regularization

    from econml.dml import DMLCateEstimator
    from sklearn.preprocessing import PolynomialFeatures
    from lightning.regression import FistaRegressor
    from econml.bootstrap import BootstrapEstimator
    from sklearn.linear_model import MultiTaskElasticNet

    est = DMLCateEstimator(model_y=MultiTaskElasticNet(alpha=0.1),
                           model_t=MultiTaskElasticNet(alpha=0.1),
                           model_final=FistaRegressor(penalty='trace', C=0.0001),
                           featurizer=PolynomialFeatures(degree=1, include_bias=False))
    est.fit(Y, T, X, W)
    te_pred = est.const_marginal_effect(np.array([[np.median(X, axis=0)]]))
    print(te_pred)
    print(np.linalg.svd(te_pred[0]))
