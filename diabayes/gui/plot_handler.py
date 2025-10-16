import numpy as np
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, RangeTool
from bokeh.plotting import figure
from bokeh.server.server import Server
from flask import Flask


class PlotHandler:

    app: Flask | None = None
    server: Server | None = None

    def __init__(self) -> None:
        pass

    def init_app(self, app: Flask) -> None:
        app.extensions["plot_handler"] = self
        self.app = app

    def make_bokeh_doc(self, doc):

        source = ColumnDataSource(dict(x=[], y=[]))
        source2 = ColumnDataSource(dict(x=[], y=[]))

        p = figure(
            height=300,
            tools="hover,pan,box_zoom,undo,redo,reset",
            sizing_mode="stretch_width",
            tooltips=[
                ("index", "$index"),
                ("time", "$snap_x"),
                ("friction", "$snap_y"),
            ],
        )
        p.line("x", "y", source=source)
        p.xaxis.axis_label = "Time [s]"
        p.yaxis.axis_label = "Friction [-]"
        p.toolbar.logo = None

        q = figure(
            height=300,
            x_range=p.x_range,
            y_axis_type="log",
            tools="hover,pan,box_zoom,undo,redo,reset",
            sizing_mode="stretch_width",
            tooltips=[
                ("index", "$index"),
                ("time", "$snap_x"),
                ("slip rate", "$snap_y"),
            ],
        )
        q.line("x", "y", source=source2)
        q.xaxis.axis_label = "Time [s]"
        q.yaxis.axis_label = "Slip rate [m/s]"
        q.toolbar.logo = None

        select = figure(
            height=200,
            y_axis_type=None,
            y_range=p.y_range,
            tools="",
            toolbar_location=None,
            sizing_mode="stretch_width",
        )
        # select.x_range.range_padding = 0.1
        select.x_range.bounds = "auto"  # type: ignore

        range_tool = RangeTool(
            x_range=p.x_range, y_range=p.y_range, start_gesture="pan"
        )
        range_tool.overlay.fill_color = "green"
        range_tool.overlay.fill_alpha = 0.3

        select.line("x", "y", source=source)
        # select.ygrid.grid_line_color = None
        select.add_tools(range_tool)

        doc._mode = None

        def set_marker_mode(attr, old, new):
            doc._mode = new

        doc.add_root(column([select, q, p], sizing_mode="stretch_width"))
        doc.add_next_tick_callback(lambda: print("Bokeh doc ready"))
        doc._source = source
        doc._source2 = source2
        pass

    def plot(self, data):
        app = self.app
        assert app is not None

        if not self.server:
            app.logger.debug("Server not initialised")
            return

        app.logger.debug("Attempting to get plotting session")

        for ctx in self.server.get_sessions("/bkapp"):
            doc = ctx.document
            app.logger.debug("Server not initialised")

            if hasattr(doc, "_source"):
                doc.add_next_tick_callback(
                    lambda: doc._source.data.update(x=data[0], y=data[1])
                )
                app.logger.debug("Friction data updated")

            if hasattr(doc, "_source2"):
                doc.add_next_tick_callback(
                    lambda: doc._source2.data.update(x=data[0], y=data[2])
                )
                app.logger.debug("Slip rate data updated")

        app.logger.debug("Plot update done")
        pass

    def clear_plot(self):
        self.plot([[], [], []])
        assert self.app is not None
        self.app.logger.debug("Plot cleared")
