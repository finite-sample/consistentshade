"""Experiment scripts for BCR research."""

from .datasets import (
    create_challenging_datasets,
    prepare_adult_income,
    prepare_bank_marketing,
    prepare_california_housing,
    prepare_credit_card_default,
    prepare_diabetes,
    prepare_german_credit,
    prepare_synthetic_regression,
    prepare_wine_quality,
)

__all__ = [
    "prepare_synthetic_regression",
    "prepare_california_housing",
    "prepare_adult_income",
    "prepare_german_credit",
    "prepare_diabetes",
    "prepare_wine_quality",
    "prepare_bank_marketing",
    "prepare_credit_card_default",
    "create_challenging_datasets",
]
