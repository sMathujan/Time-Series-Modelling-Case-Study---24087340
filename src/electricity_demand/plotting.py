"""Plotting helpers for forecasts and diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_forecasts(train, test, forecasts: dict, path=None):
    """Overlay every forecast on the train/test series."""
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(train.index, train, color="tab:gray", lw=1, label="Train")
    ax.plot(test.index, test, color="black", lw=2, label="Test (actual)")
    for name, pred in forecasts.items():
        ax.plot(pred.index, pred.reindex(test.index), lw=1.3, label=name, alpha=0.9)
    ax.axvline(test.index[0], color="k", lw=0.8, alpha=0.4)
    ax.set_title("Forecast comparison")
    ax.set_ylabel("Load (GW)")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def plot_forecast_with_ci(train, test, mean, ci, label, color="tab:red", path=None):
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(train.index, train, color="tab:gray", lw=1, label="Train")
    ax.plot(test.index, test, color="black", lw=2, label="Test (actual)")
    ax.plot(mean.index, mean, color=color, lw=1.8, label=label)
    if ci is not None:
        ax.fill_between(ci.index, ci["lower"], ci["upper"], color=color,
                        alpha=0.15, label="95% CI")
    ax.axvline(test.index[0], color="k", lw=0.8, alpha=0.4)
    ax.set_ylabel("Load (GW)")
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def plot_error_diagnostics(test, forecasts: dict, path=None):
    """Absolute error over time for each model (highlights the 2020 break)."""
    fig, ax = plt.subplots(figsize=(13, 5))
    for name, pred in forecasts.items():
        e = (pred.reindex(test.index) - test).abs()
        ax.plot(e.index, e, lw=1.2, label=name, alpha=0.9)
    ax.set_title("Absolute forecast error over time")
    ax.set_ylabel("|error| (GW)")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig
