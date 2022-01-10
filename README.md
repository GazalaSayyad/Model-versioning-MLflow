# Model-versioning-MLflow
Model monitoring and versioning with Mlflow

Machine Learning is an iterative and continouse process. Multiple times training are done to optimize a model's performance. 

Without the right methods, it is easy to lose track of experimentations with training datasets, hyperparameters, evaluation metrics, and model artifacts. This might in the long run be problematic when you need to reproduce an experiment. 

Model versioning -
Model versioning in a way involves tracking the changes made to an ML model that has been previously built.
It helps ML engineer to keep multiple versions of model.
In model versioning, aimplementation code,dataset and model changes or new algorithm trained on same dataset..all things need to be versioned, to help us keep track of important changes.

MLFLOW -
MLflow is an open-source platform for managing the end-to-end machine learning lifecycle. MLflow Models manages and deploys models from a variety of ML libraries to a variety of model serving and inference platforms.
MLFlow allows you to version data and models, repackage code for reproducible runs.

MLFlow offers 4 keys. These include: 

1. MLFlow Tracking: Track experiments by parameters, metrics, versions of code and output files.
2. MLFlow Projects:implementing code structure.
3. MLFlow Models: Providing  model with endpoint.
4. Model Registry:Version ML models. 