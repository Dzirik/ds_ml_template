"""
Visualizer

Base class for all time series plotly visualisations.

Customize markers: https://plotly.com/python/marker-style/
"""

from typing import List, Any, Dict, Optional

import plotly.graph_objs as go
from pandas import Series

from src.visualisations.plotly_base import PlotlyBase
from src.visualisations.visualisation_functions import hex_to_rgb

BASE_SELECTORS: List[Dict[str, Any]] = [
    {"count": 12, "label": "12h", "step": "hour", "stepmode": "backward"},
    {"count": 3, "label": "3d", "step": "day", "stepmode": "backward"},
    {"count": 7, "label": "1w", "step": "day", "stepmode": "backward"},
    {"count": 1, "label": "1m", "step": "month", "stepmode": "backward"},
    {"step": "all"}
]


# Selectors examples:
# {"count": 5, "label": "5min", "step": "minute", "stepmode": "backward"},
# {"count": 15, "label": "15min", "step": "minute", "stepmode": "backward"},
# {"count": 30, "label": "30min", "step": "minute", "stepmode": "backward"},
# {"count": 1, "label": "hour", "step": "hour", "stepmode": "backward"},
# {"count": 1, "label": "day", "step": "day", "stepmode": "backward"},
# {"count": 7, "label": "week", "step": "day", "stepmode": "backward"},
# {"count": 1, "label": "1m", "step": "month", "stepmode": "backward"},
# {"step": "all"}


class PlotlyTimeSeriesBase(PlotlyBase):  # type:ignore
    """
    Base class for all time series plotly visualisations.
    """

    def __init__(self, opacity: float) -> None:
        PlotlyBase.__init__(self)
        self._opacity = opacity
        self._selectors: List[Dict[str, Any]] = BASE_SELECTORS
        self._markers = {"size": 11, "line": {"width": 1.5, "color": "black"}}

    def set_selectors(self, selectors: List[Dict[str, Any]]) -> None:
        """
        Sets time selectors for time series.
        Example: [{"count": 3, "label": "3d", "step": "day", "stepmode": "backward"}].
        :param selectors: List[Dict[str, Any]].
        """
        self._selectors = selectors

    def _create_time_series_traces(self, series: List[Series], series_names: Optional[List[str]] = None,
                                   series_obs_names: Optional[List[List[str]]] = None,
                                   fill_areas: bool = False) -> List[Any]:
        """
        Creates list of classical continuous time series traces for plotting. Optionally fills the areas between the ts.
        :param series: List[Series]. List of pandas series.
        :param series_names: Optional[List[str]]. List of names for len(series) time series. If None, automatic ones
                             are generated.
        :param series_obs_names: Optional[List[List[str]]]. List of lists of names of observations for each
                                             time series.
        :param fill_areas: bool. If True if will fill the areas between time series.
        :return: List[Any]. List of traces.
        """
        if series_names is None:
            series_names = self._create_captions(len(series), "Time Series")

        if series_obs_names is None:
            series_obs_names = []
            for i, ts in enumerate(series):
                series_obs_names.append(self._create_captions(len(ts), "Observation"))

        traces = []
        for i, ts in enumerate(series):
            fill = None
            if fill_areas and i != 0:
                fill = "tonexty"
            trace = go.Scatter(x=ts.index,
                               y=ts.values,
                               name=series_names[i],
                               text=series_obs_names[i],
                               line={"color": self._colors["fill"][i % len(self._colors["fill"])]},
                               opacity=self._opacity,
                               mode="lines",
                               fillcolor=hex_to_rgb(self._colors["line"][i % len(self._colors["line"])], 0.1),
                               fill=fill)
            traces.append(trace)

        return traces

    def _create_event_traces(self, events: List[Series], event_names: Optional[List[str]] = None,
                             event_obs_names: Optional[List[List[str]]] = None) -> List[Any]:
        """
        Creates list of discrete (dots) event time series traces for plotting.
        :param events: List[Series]. List of pandas series.
        :param event_names: Optional[List[str]]. List of names for len(series) time series. If None, automatic ones
                             are generated.
        :param event_obs_names: Optional[List[List[str]]]. List of lists of names of observations for each
                                             time series.
        :return: List[Any]. List of traces.
        """
        if event_names is None:
            event_names = self._create_captions(len(events), "Event Series")

        if event_obs_names is None:
            event_obs_names = []
            for i, ts in enumerate(events):
                event_obs_names.append(self._create_captions(len(ts), "Event"))

        traces = []
        for i, ts in enumerate(events):
            trace = go.Scatter(x=ts.index,
                               y=ts.values,
                               name=event_names[i],
                               text=event_obs_names[i],
                               opacity=0.5,
                               line={"color": self._colors["dot"][i % len(self._colors["dot"])]},
                               mode="markers",
                               marker=self._markers)
            traces.append(trace)

        return traces

    def _create_layout(self, plot_title: str = "Title", y_title: str = "Y Axis Title") -> Any:
        """
        Creates layout for the plot.
        :param plot_title: str.
        :param y_title: str.
        :return: Any. Layout.
        """
        return {
            "yaxis_title": y_title,
            "xaxis_title": "Date & Time",
            "xaxis": {
                "rangeselector": {"buttons": self._selectors},
                "rangeslider": {"visible": True},
                "type": "date"},
            "title": plot_title,
            "paper_bgcolor": hex_to_rgb(self._colors["paper_background"]["color"],
                                        self._colors["paper_background"]["opacity"]),
            "plot_bgcolor": hex_to_rgb(self._colors["grid_background"]["color"],
                                       self._colors["grid_background"]["opacity"])
        }

    def _create_anomalies_from_index_trace(self, ts: Series, anomalies: List[int], anomalies_obs_names: \
            Optional[List[str]] = None) -> List[Any]:
        """
        Creates anomalies from ts based on indices anomalies.
        :param ts: Series. Time series to be used for anomalies.
        :param anomalies: Optional[List[int]]. List of indices of anomalies to be plotted from the time series.
        :param anomalies_obs_names: Optional[List[str]]. Captions for anomalies.
        :return: List[Any]. List with one trace. It is list because of the compatibility with traces handling.
        """
        if anomalies_obs_names is None:
            anomalies_obs_names = self._create_captions(len(ts), "Anomaly")

        trace = go.Scatter(x=ts.iloc[anomalies].index,
                           y=ts.iloc[anomalies].values,
                           name="Anomaly",
                           text=anomalies_obs_names,
                           line={"color": self._colors["error"][0]},
                           mode="markers",
                           marker=self._markers)
        return [trace]
