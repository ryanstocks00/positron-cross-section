"""Matplotlib importer."""

import matplotlib

matplotlib.use("agg")

import matplotlib.pyplot  # noqa: F401,E402 pylint: disable=unused-import,wrong-import-position

plt = matplotlib.pyplot
