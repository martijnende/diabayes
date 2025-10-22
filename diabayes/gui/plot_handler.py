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
        line_colour = "#3cb371"
        overlay_colour = "#f5deb3"

        p = figure(
            height=300,
            tools="hover,pan,box_zoom,xwheel_zoom,ywheel_zoom,undo,redo,reset",
            sizing_mode="stretch_width",
            tooltips=[
                ("index", "$index"),
                ("time", "$snap_x"),
                ("friction", "$snap_y"),
            ],
        )
        p.line("x", "y", line_color=line_colour, source=source)
        p.yaxis.axis_label = "Friction [-]"
        p.toolbar.logo = None
        p.toolbar.active_drag = None

        q = figure(
            height=300,
            x_range=p.x_range,
            y_axis_type="log",
            tools="hover,pan,box_zoom,xwheel_zoom,ywheel_zoom,undo,redo,reset",
            sizing_mode="stretch_width",
            tooltips=[
                ("index", "$index"),
                ("time", "$snap_x"),
                ("velocity", "$snap_y"),
            ],
        )
        q.line("x", "y", line_color=line_colour, source=source2)
        q.yaxis.axis_label = "Velocty [m/s]"
        q.toolbar.logo = None
        q.toolbar.active_drag = None

        select = figure(
            height=100,
            y_axis_type=None,
            y_range=p.y_range,
            tools="",
            toolbar_location=None,
            sizing_mode="stretch_width",
        )
        # select.x_range.range_padding = 0.1
        select.x_range.bounds = "auto"  # type: ignore
        select.xaxis.axis_label = "Time [s]"

        range_tool = RangeTool(
            x_range=p.x_range, y_range=p.y_range, start_gesture="pan"
        )
        range_tool.overlay.fill_color = overlay_colour
        range_tool.overlay.fill_alpha = 0.3

        select.line("x", "y", line_color=line_colour, line_width=3, source=source)
        # select.ygrid.grid_line_color = None
        select.add_tools(range_tool)

        doc._mode = None

        def set_marker_mode(attr, old, new):
            doc._mode = new

        doc.add_root(column([q, p, select], sizing_mode="stretch_width"))
        doc.add_next_tick_callback(lambda: print("Bokeh doc ready"))
        doc._figures = {"friction": p, "velocity": q}
        doc._renderers = {"friction": {}, "velocity": {}}
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

            if hasattr(doc, "_source"):
                doc.add_next_tick_callback(
                    lambda: doc._source.data.update(x=data[0], y=data[1])  # type: ignore
                )
                app.logger.debug("Friction data updated")

            if hasattr(doc, "_source2"):
                doc.add_next_tick_callback(
                    lambda: doc._source2.data.update(x=data[0], y=data[2])  # type: ignore
                )
                app.logger.debug("Slip rate data updated")

        app.logger.debug("Plot update done")
        pass

    def add_friction(self, id, data):
        app = self.app
        assert app is not None

        if not self.server:
            app.logger.debug("Server not initialised")
            return

        def add_curve(doc, id):
            for fig, y in zip(("friction", "velocity"), (data["mu"], data["v"])):
                p = doc._figures.get(fig)
                renderers = doc._renderers.get(fig)
                if p is not None:

                    # If a curve with this ID already
                    # exists, remove it first
                    if id in renderers:
                        renderer = renderers.pop(id)
                        p.renderers.remove(renderer)

                    source = ColumnDataSource(dict(x=data["t"], y=y))
                    renderer = p.line(
                        "x", "y", line_color="orange", source=source, line_width=2
                    )
                    doc._renderers[fig][id] = renderer

        for ctx in self.server.get_sessions("/bkapp"):
            doc = ctx.document
            if hasattr(doc, "_figures"):
                app.logger.debug(f"Creating callback for {id}")
                doc.add_next_tick_callback(lambda: add_curve(doc, id))

        app.logger.debug(f"Added friction curve {id}")

        pass

    def del_friction(self, id):
        app = self.app
        assert app is not None

        if not self.server:
            app.logger.debug("Server not initialised")
            return

        def remove_curve(doc, id):
            for fig in ("friction", "velocity"):
                p = doc._figures.get(fig)
                renderer = doc._renderers[fig].pop(id, None)
                if renderer is not None:
                    p.renderers.remove(renderer)

        for ctx in self.server.get_sessions("/bkapp"):
            doc = ctx.document
            if hasattr(doc, "_figures"):
                doc.add_next_tick_callback(lambda: remove_curve(doc, id))

        app.logger.debug(f"Removed friction curve {id}")

        pass

    def clear_plot(self):
        self.plot([[], [], []])
        assert self.app is not None
        self.app.logger.debug("Plot cleared")
