Orthogonal/Double Machine Learning
==================================

Orthogonal/Double Machine Learning (DML) is a particular approach to estimating heterogeneous treatment effects in models where the response is 
linear in the treatment and where we do not assume that there is any observed instrument :math:`Z`. Furthermore, we will assume that the 
error :math:`\eta` in the treatment equation enters linearly and is exogenous and independent of any other random variable. 

.. math::

    Y =~& H(X, W) \cdot T + g(X, W, \epsilon) \\ 
    T =~& f(X, W) + \eta & \E[\eta|X, W, \epsilon]=0

What is particularly attractive about DML is that it makes no further structural assumptions on :math:`g` and :math:`f` and estimates them 
non-parametrically using arbitrary non-parametric Machine Learning methods. Since we are in a linear in treatment response setting, 
the whole goal of the estimation part is to fit the constant marginal CATE :math:`\theta(\vec{x})`. All other quantities of interest are 
directly computable given :math:`\theta(\vec{x})`.

The idea to estimate :math:`\theta(\vec{x})` is as follows: we can re-write the structural equations as

.. math::

    Y - \E[Y | X, W] 
    = H(X, W) \cdot (T - \E[T | X, W]) + \underbrace{g(X, W, \epsilon) - \E[g(X, W, \epsilon) | X, W]}_{\zeta}

Thus if one can estimate the conditional expectation functions (both of which are non-parametric regression tasks):

.. math::

    q(\vec{x}, \vec{w}) =~& E[Y | X, W]\\
    f(\vec{x}, \vec{w}) =~& E[T | X, W]

Then we can compute the residuals:

.. math::

    \tilde{Y} =~& Y - q(\vec{x}, \vec{w})\\
    \tilde{T} =~& T - f(\vec{x}, \vec{w}) = \eta

which are subsequently related by the equation:

.. math::

    \tilde{Y} = H(X, W) \cdot \tilde{T} + \zeta

Subsequently, since :math:`\E[\zeta | X]=0` and :math:`\E[H(X, W)| X, \eta] = E[H(X, W)|X]`, we have that:

.. math::
    :nowrap:

    \begin{equation}
    \E[\tilde{Y} | X, \tilde{T}] = E[H(X, W) | X] \cdot \tilde{T}  = \theta(X) \cdot \tilde{T}
    \end{equation}

Thus estimating :math:`\theta(X)` is a final regression problem, regressing :math:`\tilde{Y}` on :math:`X, \tilde{T}`. 
If one makes parametric assumptions on the form of :math:`\theta(X)=h(X; \alpha)`, then one can estimate :math:`\alpha` via 
a plugin least squares :math:`M`-estimation:

.. math::
    :nowrap:

    \begin{equation}
    \hat{\alpha} = \arg\min_{\alpha} \E_n\left[ (\tilde{Y} - h(X;\alpha)\cdot \tilde{T})^2 \right]
    \end{equation}

The main advantage of DML is that if one makes parametric assumptions on :math:`\theta(X)`, then one achieves asymptotic rates and 
asymptotic normality on the second stage estimate :math:`\hat{\alpha}`, even if the first stage estimates on :math:`q(\vec{x}, \vec{w})` 
and :math:`f(\vec{x}, \vec{w})` are only :math:`n^{1/4}` consistent, in terms of RMSE and if they are fitted on a separate sample than 
the one used for the second stage estimation. The latter follows from the fact that the moment equations that correspond to the final 
least squares estimation (i.e. the gradient of the squared loss), satisfy a Neyman orthogonality condition with respect to the
nuisance parameters :math:`q(\vec{x}, \vec{w})` and :math:`f(\vec{x}, \vec{w})`. For a more detailed exposition of Neyman orthogonality 
we refer the reader to [Chernozhukov2016]_, [Mackey2017]_.

In fact, one can achieve a reduction in variance by not fully sample splitting, but following an approach called cross-fitting, where half 
of the sample is used to fit estimates :math:`\hat{f}, \hat{q}` that are used to compute residuals on the other half, and vice versa. 
Finally, in the second stage we can minimize the squared loss evaluated on the union of all the residuals.

If the parameterization is a linear one in some feature expansion of :math:`X`, i.e. the effect of treatment :math:`j` on outcome :math:`i` 
takes the form :math:`h_{ij}(X; \alpha) = \ldot{\alpha_{ij}}{\phi(X)}`, then the final stage problem can be written as a multi-task linear 
regression between :math:`\tilde{Y}` and the vector produced by taking the Kronecker-product of the vectors :math:`T` and :math:`\phi(X)`, 
i.e. :math:`T\otimes \phi(X) = \mathtt{vec}(T\cdot \phi(X)^T)`. This regression will estimate the coefficients :math:`\alpha_{ijk}` 
for each outcome :math:`i`, treatment :math:`j` and feature :math:`k`. To avoid invoking non-convex minimization processes as part of the 
final stage estimation, we will focus on such a linear parametrization of the treatment effect in our implementation. 

One might also want to regularize the second stage if the parameterization :math:`h(X; \alpha)` is too flexible compared to the 
sample size :math:`n`. In that case, the final stage corresponds to a regularized :math:`M` estimation:

.. math::
    
    \hat{\alpha} = \arg\min_{\alpha} \E_n\left[ (\tilde{Y} - h(X;\alpha)\cdot \tilde{T})^2 \right] + \lambda R(\alpha)

for some strongly convex regularizer :math:`R`. For instance, if :math:`Y` is single dimensional and :math:`R(a) =\|\alpha\|_1`, 
we recover the Lasso, if :math:`R(a)=\kappa \|\alpha\|_2 + (1-\kappa)\|\alpha\|_1` we recover the ElasticNet. For multi-dimensional :math:`Y`, 
one can impose several extensions to the matrix of parameters :math:`\alpha`, such as the one corresponding to the MultiTask Lasso 
:math:`\sum_{j} \sum_{i} \alpha_i^2` or MultiTask ElasticNet or nuclear norm regularization  [Jaggi2010]_, which enforces low-rank 
constraints on the matrix :math:`\alpha`. Even under regularized second stage estimation, there is benefit in using the Double ML approach 
as it still renders the MSE of second stage estimation robust to first stage errors (see e.g.  [Chernozhukov2017]_, [Chernozhukov2018]_). 

API and Pseudocode for General Double ML Estimator
--------------------------------------------------

\kbcomment{The implementation for \texttt{const\_marginal\_effect} below is wrong: for the output to have the right number of elements (:math:`m \times d_y \times d_t`),
we need to pass in :math:`d_t` times as much input - and it should not be all ones but rather have a single one per row.  Sadly, I don't know of a concise way to describe the necessary contortions required to reshape the input and output. }

.. code-block:: python3
    :caption: Double ML CATE Estimator Class

    class DMLCateEstimator(LinearCateEstimator):

        def __init__(self, model_y, model_t, 
                        model_final=LinearRegression(fit_intercept=False),
                        featurizer=PolynomialFeatures(degree=1, include_bias=True)):
            ''' Initialize models and feature creator.
            
            Parameters
            model_y: (sklearn model) used to fit the regression of Y on X, W
            model_t: (sklearn model) used to fit the regression of T on X, W
            model_final: (sklearn linear model) used to fit the final regression
            featurizer: (sklearn preprocessor) used to create features ϕ(X) of X
                        in the final stage
            '''
            self.models_y = [clone(model_y), clone(model_y)]
            self.models_t = [clone(model_t), clone(model_t)]
            self.model_final = clone(model_final)
            self.featurizer = clone(featurizer)

        def fit(self, Y, T, X=None, W=None):
            ''' Fits a model of the heterogeneous constant marginal CATE based on
            the Double ML process.
        
            Parameters:
            Y: (n × d_y) matrix of outcomes for each sample
            T: (n × d_t) matrix of treatments for each sample
            X: optional (n × d_x) matrix of features for each sample
            W: optional (n × d_w) matrix of controls for each sample
            '''
            kf = KFold(n_splits=2)
            y_res = np.zeros(np.shape(Y))
            T_res = np.zeros(np.shape(T))
            for fid, (train_index, test_index) in enumerate(kf.split(X)):
                Y_train, Y_test = Y[train_index], Y[test_index]            
                T_train, T_test = T[train_index], T[test_index]
                X_train, X_test = X[train_index], X[test_index]            
                W_train, W_test = W[train_index], W[test_index]
                
                # Fit treatment model on co-variates from train data
                self.models_t[fid].fit((X_train, W_train), T_train)
                # Compute treatment residuals for test data
                T_res[test_index] = T_test - self.models_t[fid].predict((X_test, W_test))
                # Fit outcome model on co-variates from train data
                self.models_y[fid].fit((X_train, W_train), Y_train)
                # Compute outcome residuals for test data
                y_res[test_index] = Y_test - self.models_y[fid].predict((X_test, W_test))
            
            
            self.model_final.fit(product(T_res, self.featurizer.fit_transform(X)), y_res)

        
        def const_marginal_effect(self, X=None):
            ''' Calculates the constant marginal CATE θ(·) conditional on a vector of
                features on a set of m test samples {X_i}
            
                Parameters:
                X: optional (m × d_x) matrix of features for each sample
            
                Returns:
                theta: (m × d_y × d_t) matrix of constant marginal CATE of each treatment
                        on each outcome for each sample
            '''
            return self.model_final.predict(product(ones, self.featurizer.fit_transform(X)))
        
        @property
        def coef_(self):
            ''' Returns the sparse three dimensional tensor α, whose α[i,j,k] entry is the 
            coefficient associated with outcome i, treatment j and feature k'''
            return self.model_final.coef_.reshape(d_y, d_t, d_{phi(x)})
            
        def fitted_models_y(self):
            return self.models_y if fitted else raise error
        
        def fitted_models_t(self):
            return self.models_t if fitted else raise error
        
        def fitted_model_final(self):
            return self.model_final if fitted else raise error
	
API and Pseudocode for Sparse Linear Double ML Estimator
--------------------------------------------------------

One particularly attractive special case of the DML framework is the case when :math:`W` is a high-dimensional vector (i.e. :math:`d_w >> n`) and further the nuisance functions :math:`f, g` are assumed to be linear in :math:`X, W, \epsilon`, and :math:`H(X, W)` is also linear in :math:`\phi(X), W`, i.e.: 

.. math::
    :nowrap:

    \begin{align}
    H_{ij}(X, W) =~& \ldot{\alpha_{ij}}{\phi(X)} + \ldot{\tilde{\alpha}_{ij}}{W} \\
    g_i(X, W, \epsilon) =~& \ldot{\beta_i}{(X; W)} + \epsilon\\
    f_i(X, W) =~& \ldot{\gamma_i}{(X; W)}\\
    \end{align}

In this case we have a more structural form for the two regression tasks of estimating :math:`q` and :math:`p`. In particular, we can write:

.. math::
    :nowrap:

    \begin{align*}
    q_i(\vec{x}, \vec{w}) =~& \ldot{\delta_i}{(\vec{x}; \vec{w}; (\phi(\vec{x}); \vec{w}) \otimes (\vec{x}; \vec{w}))}\\
    f_i(\vec{x}, \vec{w}) =~& \ldot{\gamma_i}{(\vec{x}; \vec{w})}
    \end{align*}

Thus one can use the Lasso regression to estimate the nuisance functions :math:`q` and :math:`p` in the first stage of the Double ML process. This high-dimensional linear structural assumption enables provable worst-case rates of :math:`n^{-1/4}` from the first stage estimates as long as the sparsity of the coefficients :math:`\delta` and :math:`\gamma` is small enough. Hence, the assumptions of the DML framework are provably satisfied. 

For this reason our library also provides a subclass of the DML estimator class that is tailored to sparse linear models for the nuisance functions. 

.. code-block:: python3
    :caption: Sparse Linear Double ML CATE Estimator Class

    class MultiTaskWrapper(BaseEstimator):
        ''' This is a generic MultiTask wrapper for any sklearn base estimator.
        Essentially takes any estimator that is supposed to predict a 1-dimensional
        label y, and turns it into an estimator that predict a d-dimensional
        label y, produced by running d independent estimation problems for each
        output. This is mostly a utility class.
        '''

        def __init__(self, base_model):
            self.base_model = base_model
        
        def fit(self, X, Y):
            self.base_models = [clone(self.base_model).fit(X, Y[:, i]) 
                                    for i in range(Y.shape[1])]	
        
        def predict(self, X):
            return np.array([model.predict(X) for model in self.base_models]).T
        
        def	__getattr__(self, name):
            return [model.__getattr__(name) for model in self.base_models]
        
        def __setattr__(self, name, value):
            [model.__setattr__(name, value) for model in self.base_models]
        
    class SparseLinearDMLCateEstimator(DMLCateEstimator):
        ''' This is a specialization of the DMLCateEstimator to sparse linear models
        for the nuisance functions.
        '''
        
        def __init__(self, linear_model_y=LassoCV(), linear_model_t=LassoCV(), 
                        model_final=LinearRegression(fit_intercept=False),
                        featurizer=PolynomialFeatures(degree=1)):
            ''' Initialize models and feature creator.
            
            Parameters
            model_y: (sklearn linear model) used to fit the regression of each Y_i
                        on X, W, (X; W) ⊗ (ϕ(X); W)
            model_t: (sklearn linear model) used to fit the regression of T_i on X, W
            model_final: (sklearn linear model) used to fit the final regression
            featurizer: (sklearn preprocessor) used to create features ϕ(X) of X
                        in the final stage
            '''
            self.linear_model_y = linear_model_y
            self.linear_model_t = linear_model_t
            super().__init__(None, None, model_final, featurizer)
        
        def fit(self, Y, T, X=None, W=None):
            ''' Fits based on a sparse linear DML model. Builds the right composite models
            from the two base linear models that the user specified. The model for Y
            first transforms the data to add the cross product terms and then calls the base
            linear model on the transformed data for every coordinate of Y. For the model
            for T it calls the base estimator for every coordinate of T.
            '''
            def transform(XW, dX=X.shape[1]):
                return (XW; cross_product(XW, (XW[:dX], self.featurizer.fit_transform(XW[dX:])))) 
            
            self.model_y = Pipeline(transform, MultiTaskWrapper(clone(self.linear_model_y)))
            self.model_t = MultiTaskWrapper(self.linear_model_t)
            
            super().fit(Y, T, X, W)


Example Use Cases: Single Outcome, Single Treatment
---------------------------------------------------

We consider some example use cases of the library when :math:`Y` and :math:`T` are :math:`1`-dimensional.

.. rubric:: Random Forest First Stages

A classical non-parametric regressor for the first stage estimates is a Random Forest. Using RandomForests in our API is as simple as:

.. code-block:: python3
    :caption: Random Forest First Stage

    est = DMLCateEstimator(model_y=sklearn.ensemble.RandomForestRegressor(), 
                            model_t=sklearn.ensemble.RandomForestRegressor())
    est.fit(y, T, X, W)


.. rubric:: Polynomial Features for Heterogeneity

Suppose that we believe that the treatment effect is a polynomial of :math:`X`, i.e.

.. math::
    
    Y = (\alpha_0 + \alpha_1 X + \alpha_2 X^2 + \ldots) \cdot T + g(X, W, \epsilon)

Then we can estimate the coefficients :math:`\alpha_i` by running:

.. code-block:: python3
    :caption: Polynomial Second Stage Features

    est = DMLCateEstimator(model_y=sklearn.ensemble.RandomForestRegressor(), 
                            model_t=sklearn.ensemble.RandomForestRegressor(),
                            featurizer=sklearn.preprocessing.PolynomialFeatures(degree=5))
    est.fit(y, T, X, W)

    # To get the coefficients of the polynomial fitted in the final stage we can
    # access the coef_ attribute of the fitted second stage model. This would 
    # return the coefficients in front of each term in the vector T⊗ϕ(X).
    a_hat = est.sparse_coef_


.. rubric:: Fixed Effects

To add fixed effect heterogeneity, we can create one-hot encodings of the id, which is assumed to be part of the input:

.. code-block:: python3
    :caption: Custom Featurizer

    # removing one id to avoid colinearity, as is standard for fixed effects
    X = sklearn.preprocessing.CategoricalEncoder().fit_transform(id)[1:] 
    # the default featurizer also augments the Z features with a bias term. 
    # So a treatment effect offset will also be fitted
    est = DMLCateEstimator(model_y=sklearn.ensemble.RandomForestRegressor(), 
                            model_t=sklearn.ensemble.RandomForestRegressor())
    est.fit(y, T, X, W)
    # The latter will fit a model for θ(x) of the form ̂α_0 + ̂α_1 𝟙{id=1} + ̂α_2 𝟙{id=2} + ...
    # The vector of α can be extracted as follows
    a_hat = est.sparse_coef_

.. rubric:: Custom Features

One can also define a custom featurizer, as long as it supports the fit\_transform interface of sklearn.

.. code-block:: python3
    :caption: Custom Featurizer

    class LogFeatures(object):
        ''' Augments the features with logarithmic features and returns the augmented structure'''
        def fit_transform(self, X):
            return np.concatenate((X, np.log(X)), axis=1)
            
    est = DMLCateEstimator(model_y=sklearn.ensemble.RandomForestRegressor(), 
                            model_t=sklearn.ensemble.RandomForestRegressor(),
                            featurizer=sklearn.preprocessing.LogFeatures())
    est.fit(y, T, X, W)
    a_hat = est.sparse_coef_

We can even create a Pipeline or Union of featurizers that will apply multiply featurizations, e.g. first creating log features and then adding polynomials of them:

.. code-block:: python3
    :caption: Pipeline Featurizer

    est = DMLCateEstimator(model_y=sklearn.ensemble.RandomForestRegressor(), 
                            model_t=sklearn.ensemble.RandomForestRegressor(),
                            featurizer=Pipeline({sklearn.preprocessing.LogFeatures(), 
                                            sklearn.preprocessing.PolynomialFeatures(degree=3)}))
    est.fit(y, T, X, W)
    a_hat = est.sparse_coef_


.. rubric:: Sparse Linear Models

If we also want to assume that the nuisance models are sparse linear and use the elasticNet instead of the LassoCV, then we can simply call:

.. code-block:: python3
    :caption: Sparse Linear Nuisance Models

    est = SparseLinearDMLCateEstimator(linear_model_y=ElasticNetCV(), 
                            model_t=ElasticNetCV(),
                            featurizer=sklearn.preprocessing.PolynomialFeatures(degree=3))
    est.fit(y, T, X, W)

We can also access the coefficients in front of :math:`X` and :math:`W` in the first stage treatment model (propensity model) by looking at the coef\_ of the fitted first stage models
on each split:

.. code-block:: python3
    :caption: Examining First Stage Treatment Models

    gamma_hat1, gamma_hat2 = [model.coef_ for model in est.fitted_models_t]

The first :math:`d_x` coordinates of these coefficients correspond to coefficients in front of :math:`X` and the remainder the coefficients in front of :math:`W`. 


Example Use Cases: Single Outcome, Multiple Treatments
------------------------------------------------------

Suppose that we believed our DGP looks as in the example used in the general section:

.. math::

    Y =~& \gamma T^2 + \delta X T + \ldot{\zeta}{W} + \kappa + \epsilon \\
    T =~& \ldot{\alpha}{W}  + \eta

Then we could fit such a model by. using polynomial features for :math:`Z` and expanding the treatment vector to contain also polynomial features:

.. code-block:: python3
    :caption: Polynomial Treatments

    est = DMLCateEstimator(model_y=sklearn.ensemble.RandomForestRegressor(), 
                            model_t=sklearn.ensemble.RandomForestRegressor(),
                            featurizer=PolynomialFeatures(degree=2))
    est.fit(y, np.concatenate((T, T**2), axis=1), X, W)

    # the coefficients α_ij corresponding to each term T^i X^j for i+j=2 can be recovered by
    a_hat = est.sparse_coef_
    # entry j*d_T+i = j*2 + i of this vector contains the coefficient α_ij

The latter would fit a slightly more general model effect model of the form:

.. math::

    Y = \alpha_{01} T + \alpha_{02} T^2 + \alpha_{11} X T + \alpha_{12} X T^2 + \alpha_{21} X^2 T + \alpha_{22} X^2 T^2 + \ldot{\zeta}{W} + \kappa + \epsilon

If one wants to enforce sparsity of the :math:`\alpha_{ij}` coefficients, then a Lasso or DebiasedLasso model could be used for the final stage.

.. code-block:: python3
    :caption: Lasso or Debiased Lasso Second Stage

    est = DMLCateEstimator(model_y=sklearn.ensemble.RandomForestRegressor(), 
                            model_t=sklearn.ensemble.RandomForestRegressor(),
                            model_final=LassoCV() or DebiasedLasso(),
                            featurizer=PolynomialFeatures(degree=2))
    est.fit(y, np.concatenate((T, T**2), axis=1), X, W)


Alternatively, we can estimate the more constraint model by building augmented features :math:`XT` and not using any :math:`X` for heterogeneity:

.. code-block:: python3
    :caption: Direct Composite Treatments

    est = DMLCateEstimator(model_y=sklearn.ensemble.RandomForestRegressor(), 
                            model_t=sklearn.ensemble.RandomForestRegressor())
    est.fit(y, np.concatenate((T, T*X), axis=1), None, W)

However, the latter would also orthogonalize :math:`X` on :math:`W`, which could lead to an increase in variance. 

Example Use Cases: Multiple Outcome, Multiple Treatments
--------------------------------------------------------

In settings like demand estimation, we might want to fit the demand of multiple products as a function of the price of each one of them, i.e. fit the matrix of cross price elasticities. The latter can be done, by simply setting as :math:`Y` to be the vector of demands and :math:`T` to be the vector of prices. Then we can recover the 
matrix of cross price elasticities as:

.. code-block:: python3
    :caption: Cross-Price Elasticities

    est = DMLCateEstimator(model_y=sklearn.ensemble.RandomForestRegressor(), 
                            model_t=sklearn.ensemble.RandomForestRegressor())
    est.fit(Y, T, None, W)

    # a_hat[i,j] contains the elasticity of the demand of product i on the price of product j
    a_hat = est.constant_marginal_effect()


\kbcomment{Note that the model here is extremely general: the prices of every product can depend on the features of all other products; while this may be desirable in some cases, it also limits the ability to put useful priors on the price setter's behavior.  Is this where we should introduce a discussion of panel creation?}

Similarly we can get heterogeneous cross-price elasticities with respect to some variables :math:`X`.

.. code-block:: python3
    :caption: Heterogeneous Cross-Price Elasticities

    X = 1\{Christmas\}
    est = DMLCateEstimator(model_y=sklearn.ensemble.RandomForestRegressor(), 
                            model_t=sklearn.ensemble.RandomForestRegressor())
    est.fit(Y, T, X, W)

    # est.coef(1) contains the cross-price elasticities when X=1, i.e. during christmas. 
    a_christmas = est.constant_marginal_effect([[1]])
    # Similarly est.coef(0) contains the cross price elasticities when it is not christmas.
    a_non_christmas = est.constant_marginal_effect([[0]])

We can create even more complex conditional statements, such as store specific elasticities during christmas:

.. code-block:: python3
    :caption: Heterogeneous Cross-Price Elasticities

    X = (1\{Christmas\}, 1\{Store=Online\})
    est = DMLCateEstimator(model_y=sklearn.ensemble.RandomForestRegressor(), 
                            model_t=sklearn.ensemble.RandomForestRegressor())
    est.fit(Y, T, X, W)

    # est.coef(1, 1) contains the cross-price elasticities in the online store during christmas. 
    a_christmas = est.constant_marginal_effect([[1, 1]])
    # est.coef(0, 1) contains the cross price elasticities in the online store
    # when it is not christmas, etc.
    a_non_christmas = est.constant_marginal_effect([[0, 1]])
