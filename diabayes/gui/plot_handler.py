import numpy as np
from bokeh.embed import components
from bokeh.plotting import figure
from flask import Flask


class PlotHandler:

    app: Flask | None = None

    def __init__(self) -> None:
        pass

    def init_app(self, app: Flask) -> None:
        app.extensions["plot_handler"] = self
        self.app = app

    def make_bokeh_doc(self, doc):

        p = figure(
            tools="pan,wheel_zoom,reset",
            sizing_mode="stretch_both",
        )

        doc._mode = None

        def set_marker_mode(attr, old, new):
            doc._mode = new

        doc.add_root(p)
        return doc

    def plot(self, data):
        pass
