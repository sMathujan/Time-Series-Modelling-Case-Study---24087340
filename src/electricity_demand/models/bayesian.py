"""Optional Bayesian forecasting model.

Not implemented in this study: the assignment lists the Bayesian model as
optional and it was not part of the submitted analysis (Parts 1-6 cover
benchmarks, SARIMA, SARIMAX, a feature-based model and an LSTM).

A natural implementation would be a Bayesian linear/structural regression with
seasonal (Fourier), temperature and holiday covariates, e.g. via PyMC or
statsmodels' UnobservedComponents, reporting posterior predictive intervals and
coefficient posteriors. Left as a documented extension point.
"""

from __future__ import annotations


def fit_bayesian(*args, **kwargs):
    raise NotImplementedError(
        "Bayesian model is an optional extension and is not implemented. "
        "See module docstring for a suggested approach."
    )


def forecast_bayesian(*args, **kwargs):
    raise NotImplementedError(
        "Bayesian model is an optional extension and is not implemented."
    )
