# AzureML
from azureml.core.experiment import Experiment
from azureml.core import Workspace
from azureml.train.automl.automlconfig import AutoMLConfig
from sklearn.multioutput import MultiOutputRegressor
# helper imports
import time
import copy

"""Automated Machine Learning Support For EconML Estimators. This allows analysts
to use AutomatedML to automate the process of selecting models for models Y, T,
and final of their causal inferenve estimator.
"""


def setAutomatedMLWorkspace(create_workspace=False,
                            create_resource_group=False, workspace_region=None, *,
                            subscription_id=None, resource_group=None, workspace_name=None):
    """Set configuration file for AutomatedML actions with the EconML library. If
    ``create_workspace`` is set true, a new workspace is created
    for the user. If ``create_workspace`` is set true, a new workspace is
    created for the user.

    Parameters
    ----------

    create_workspace: Boolean, optional, default False
       If set to true, a new workspace will be created if the specified
       workspace does not exist.

    create_resource_group: Boolean, optional, default False
       If set to true, a new resource_group will be created if the specified
       resource_group does not exist.

    workspace_region: String, optional
       Region of workspace, only necessary if create_new is set to true and a
       new workspace is being created.

    subscription_id: String, required
       Definition of a class that will serve as the parent class of the
       AutomatedMLMixin. This class must inherit from _BaseDMLCateEstimator.

    resource_group: String, required
       Name of resource group of workspace to be created or set.

    workspace_name: String, required
       Name of workspace of workspace to be created or set.
    """
    try:
        ws = Workspace(subscription_id=subscription_id, resource_group=resource_group, workspace_name=workspace_name)
        # write the details of the workspace to a configuration file to the notebook library
        ws.write_config()
        print("Workspace configuration has succeeded.")
    except:
        if(create_new):
            print("Workspace not accessible. Creating a new workspace and \
            resource group.")
            ws = Workspace.create(name=workspace_name,
                                  subscription_id=subscription_id,
                                  resource_group=resource_group,
                                  location=workspace_region,
                                  create_resource_group=create_resource_group,
                                  sku='basic',
                                  exist_ok=True)
            ws.get_details()
        else:
            print("Workspace not accessible. Set create_workspace = True and \
            create_resource_group = True and run again to create a new \
            workspace and resource group.")


def addAutomatedML(baseClass):
    """
    Add AutomatedMLMixin to specified base class.

    In addition to a mixin that creates automatedML models from AutoMLConfig
    objects, another mixin is added specifying modifications to the
    AutoMLConfig. The business logic that this mixin runs is detailed below:

    If the baseClass is of type NonParamDMLCateEstimator
    the automatedML configuration must support sample weights for
    correctness, and addAutomatedML will add functionality inject the
    following models to the ``model_final_automl_config`` blacklist.

    1. LightGBM
    2. GradientBoostingRegressor
    3. RandomForestRegressor
    4. ExtraTreesRegressor
    5. DecisionTreeRegressor
    6. KNeighborsRegressor
    7. DNNRegressor
    8. SGDRegressor
    9. XGBoost

    If the baseClass is not of type NonParamDMLCateEstimator
    the automatedML configuration must be a linear model for correctness,
    and addAutomatedML will add functionality inject the following
    models to the ``model_final_automl_config`` blacklist.

    1. GradientBoostingRegressor
    2. SGDRegressor
    3. RandomForestRegressor
    4. ExtraTreesRegressor
    5. DNNRegressor
    6. LinearRegressor
    7. FastLinearRegressor

    Parameters
    ----------

    baseClass: Class, required
       Definition of a class that will serve as the parent class of the
       AutomatedMLMixin.

    Returns
    ----------

    automatedMLClass: Class
      A modified version of ``baseClass`` that accepts the parameters of the
      AutomatedML Mixin rather in addition to the original class objects.

   """

    class AutomatedMLClass(AutomatedMLMixin, baseClass):
        pass
    return AutomatedMLClass


class AutomatedMLModel():
    def __init__(self, automl_config, workspace, experiment_name_prefix="aml_experiment"):
        """
        scikit-learn style model fitted and specified with automatedML.

        automatedML uses AzureML's Automated Machine Learning library
        to automatically preprocess data, specify features, and
        selects a model given a pair of training data and labels.

        Parameters
        ----------

        automl_config: azureml.train.automl.automlconfig.AutoMLConfig, required
           Configuration for submitting an Automated Machine Learning experiment in Azure Machine Learning.
           This configuration object contains and persists the parameters for configuring the experiment run parameters,
           as well as the training data to be used at run time. For guidance on selecting your settings, you may refer
           to https://docs.microsoft.com/azure/machine-learning/service/how-to-configure-auto-train.

        workspace: azureml.core.experiment.Experiment, optional
            The main experiment to associated with the automatedML runs for

        experiment_name_prefix: String, optional
            Prefix of experiment name for generated by SciKitAutoMLModel. The full name of
            the experiment will be {EXPERIMENT_NAME_PREFIX}_{INITIALIZE_EXPERIMENT_TIMESTAMP}.
            Must be comprised of alphanumeric characters, hyphens, underscores and have at most 18 characters.
        """
        self._innerModel = _InnerAutomatedMLModel(
            automl_config, workspace, experiment_name_prefix=experiment_name_prefix)

    def fit(self, X, y, sample_weight=None):
        """
        Select and fit model.

        Parameters
        ----------

        X: numpy.ndarray or pandas.DataFrame, required
           The training features to use when fitting pipelines during AutoML experiment.

        y: numpy.ndarray or pandas.DataFrame, required
           Training labels to use when fitting pipelines during AutoML experiment.

        sample_weight: numpy.ndarray or pandas.DataFrame, optional
            The weight to give to each training sample when running fitting pipelines,
            each row should correspond to a row in X and y data.

        experiment_name_prefix: String, optional
            Prefix of experiment name for generated by SciKitAutoMLModel. The full name of
            the experiment will be {EXPERIMENT_NAME_PREFIX}_{INITIALIZE_EXPERIMENT_TIMESTAMP}.
            Must be comprised of alphanumeric characters, hyphens, underscores and have at most 18 characters.
        """
        # if y is a multioutput model
        if y.ndim > 1:
            # Make sure second dimension has 1 or more item
            if y.shape[1] > 1:
                # switch _inner Model to a MultiOutputRegressor
                self._innerModel = MultiOutputRegressor(self._innerModel)
                self._innerModel.fit(X, y, sample_weight=sample_weight)
                return
        self._innerModel.fit(X, y, sample_weight=sample_weight)

    def predict(self, X):
        """
        Predict using selected and fitted model.

        X: numpy.ndarray or pandas.DataFrame, required
           The training features to use for predicting labels
        """
        return self._innerModel.predict(X)

    def predict_proba(self, X):
        """
        Predict using selected and fitted model.

        X: numpy.ndarray or pandas.DataFrame, required
           The training features to use for predicting label probabilities.
        """
        return self._innerModel.predict_proba(X)


class _InnerAutomatedMLModel():
    # Inner single model to be passed that wrapper can use to pass into MultiOutputRegressor
    def __init__(self, automl_config, workspace,
                 experiment_name_prefix="aml_experiment"):
        self._show_output = automl_config._show_output
        self._workspace = workspace
        self._automl_config = automl_config
        self._experiment_name_prefix = experiment_name_prefix

    def get_params(self, deep=True):
        # Must be implemented for MultiOutputRegressor to view _InnerAutomatedMLModel
        # as an sklearn estimator
        return {
            'workspace': self._workspace,
            'automl_config': self._automl_config,
            'experiment_name_prefix': self._experiment_name_prefix
        }

    def fit(self, X, y, sample_weight=None):
        # fit implementation for a single output model.
        # Create experiment for specified workspace
        automl_config = copy.deepcopy(self._automl_config)
        current_time = time.localtime()
        current_time_string = time.strftime('%y_%m_%d-%H_%M_%S', current_time)
        experiment_name = self._experiment_name_prefix + "_" + current_time_string
        self._experiment = Experiment(self._workspace, experiment_name)
        # Configure automl_config with training set information.
        automl_config.user_settings['X'] = X
        automl_config.user_settings['y'] = y
        automl_config.user_settings['sample_weight'] = sample_weight
        # Wait for remote run to complete, the set the model
        print("Experiment " + experiment_name + " has started.")
        local_run = self._experiment.submit(automl_config, show_output=self._show_output)
        print("Experiment " + experiment_name + " completed.")
        _, self._model = local_run.get_output()

    def predict(self, X):
        return self._model.predict(X)

    def predict_proba(self, X):
        return self._model.predict_proba(X)


class AutomatedMLMixin():
    def __init__(self, **kwargs):
        """
        Mixin enabling users to leverage automatedML as their model of choice in
        Double Machine Learners and Doubly Robust Learners. It instantiates
        AutomatedMLModels for each automl_config provided and pass them as
        parameters into its parent class.

        Parameters
        ----------

        kwargs: AutoMLConfig, optional
           kwargs that are passed in order to initiate the final automatedML run.
           Any kwarg, that is an AutoMLConfig, will be converted into as
           AutomatedMLModel.
       """
        # Loop through the kwargs if any of them is an AutoMLConfig file, pass them
        # create model and pass model into final.
        for key in kwargs:
            kwarg = kwargs[key]
            if isinstance(kwarg, EconAutoMLConfig):
                kwargs[key] = self._get_automated_ml_model(kwarg, key)
        super().__init__(**kwargs)

    def _get_automated_ml_model(self, automl_config, prefix):
        # takes in either automated_ml config and instantiates
        # an AutomatedMLModel
        # The prefix can only be 18 characters long
        # because prefixes come from kwarg_names, we must ensure they are
        # short enough.
        prefix = prefix[:18]
        # Get workspace from config file.
        workspace = Workspace.from_config()
        return AutomatedMLModel(automl_config, workspace,
                                experiment_name_prefix=prefix)


class EconAutoMLConfig(AutoMLConfig):

    def __init__(self, sample_weights_required=False, linear_model_required=False, show_output=False, **kwargs):
        """
        Azure AutoMLConfig object with added guards to ensure correctness when used
        with EconML

        Parameters
        ----------

        sample_weights_required: Boolean, optional, default False
           If set true, only models that require sample weights will be selected during
           AutomatedML.

        linear_model_required: Boolean, optional, default False
           If set to true, only linear models will be selected during AutomatedML.

        show_output: Boolean, optional, default False
            If set to true, outputs for the corresponding AutomatedMLModel
            will be shown when it is fitted.

        kwargs: list, optional
            List of kwargs to be passed to a correspodning AutoML Config object.
            To view the full documentation of the kwargs, you may refer to
            https://docs.microsoft.com/en-us/python/api/azureml-train-automl-client
            /azureml.train.automl.automlconfig.automlconfig?view=azure-ml-py

        """
        if(linear_model_required):
            kwargs["blacklist"] = ["LightGBM",
                                   "GradientBoostingRegressor",
                                   "RandomForestRegressor",
                                   "RandomForestRegressor",
                                   "ExtraTreesRegressor",
                                   "DecisionTreeRegressor",
                                   "KNeighborsRegressor",
                                   "DNNRegressor",
                                   "SGDRegressor",
                                   "XGBoost"]
        if(sample_weights_required):
            kwargs["blacklist"] = ["GradientBoostingRegressor",
                                   "SGDRegressor",
                                   "RandomForestRegressor",
                                   "ExtraTreesRegressor",
                                   "DNNRegressor",
                                   "LinearRegressor",
                                   "FastLinearRegressor"]

        # show output is not stored in the config in AutomatedML, so we need to make it a field.
        self._show_output = show_output

        super().__init__(**kwargs)
