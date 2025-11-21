"""Sous-package d'exemples et wrappers utilitaires."""

from . import generate as generate
from . import train as train
from . import predict as predict
from . import tensorflow_alternative as tensorflow_alternative

# Exposer fonctions courantes au niveau du package
generate_sample_data = generate.generate_sample_data
generate_demo_models = generate.generate_demo_models
train_model_from_csv = train.train_model_from_csv
train_with_custom_data = train.train_with_custom_data
predict_from_csv = predict.predict_from_csv
train_with_tensorflow_example = tensorflow_alternative.train_with_tensorflow_example

__all__ = [
    'generate_sample_data', 'generate_demo_models',
    'train_model_from_csv', 'train_with_custom_data',
    'predict_from_csv', 'train_with_tensorflow_example'
]
