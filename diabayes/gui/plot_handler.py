from bokeh.io import curdoc
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, RangeTool
from bokeh.plotting import figure
from bokeh.server.server import Server
from flask import Flask
from tornado.ioloop import IOLoop

bokeh_url = "http://127.0.0.1:5006/"


class PlotHandler:

    app: Flask | None = None
    server: Server | None = None
    source = ColumnDataSource(dict(x=[], y=[]))
    source2 = ColumnDataSource(dict(x=[], y=[]))

    def __init__(self) -> None:
        pass

    def init_app(self, app: Flask) -> None:
        app.extensions["plot_handler"] = self
        self.app = app

    def make_bokeh_doc(self, doc):

        line_colour = "#3cb371"
        overlay_colour = "#f5deb3"

        # Figure showing friction data
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
        p.line("x", "y", line_color=line_colour, source=self.source)
        p.yaxis.axis_label = "Friction [-]"
        p.toolbar.logo = None
        p.toolbar.active_drag = None

        # Figure showing velocity data
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
        q.line("x", "y", line_color=line_colour, source=self.source2)
        q.yaxis.axis_label = "Velocty [m/s]"
        q.toolbar.logo = None
        q.toolbar.active_drag = None

        # Overview panel (friction only)
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

        select.line("x", "y", line_color=line_colour, line_width=3, source=self.source)
        # select.ygrid.grid_line_color = None
        select.add_tools(range_tool)

        doc.add_root(column([q, p, select], sizing_mode="stretch_width"))
        doc.context = {
            "figures": {"friction": p, "velocity": q},
            "renderers": {"friction": {}, "velocity": {}},
        }

        pass

    def _get_doc(self):
        assert self.server is not None, "Server not initialised"

        # Grab the current sessions (there should be at least 1)
        current_sessions = self.server.get_sessions("/")
        assert len(current_sessions) > 0, "Session not initialised"
        session = current_sessions[0]
        # Get the session document
        doc = session.document
        return doc

    def plot(self, data):

        # Get the session document
        doc = self._get_doc()

        # Update data sources
        def update_friction():
            self.source.data = dict(x=data[0], y=data[1])

        def update_velocity():
            self.source2.data = dict(x=data[0], y=data[2])

        # Add callback
        doc.add_next_tick_callback(update_friction)
        doc.add_next_tick_callback(update_velocity)
        assert self.server is not None
        self.server.io_loop.add_callback(lambda: None)  # Trick to "wake up" thread
        pass

    def add_friction(self, id, data):

        # Get the session document
        doc = self._get_doc()

        def add_curve(doc, id):
            for fig, y in zip(("friction", "velocity"), (data["mu"], data["v"])):
                p = doc.context["figures"].get(fig)
                renderers = doc.context["renderers"].get(fig)
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
                    doc.context["renderers"][fig][id] = renderer

        doc.add_next_tick_callback(lambda: add_curve(doc, id))
        assert self.server is not None
        self.server.io_loop.add_callback(lambda: None)  # Trick to "wake up" thread

        assert self.app is not None
        self.app.logger.debug(f"Added friction curve {id}")

        pass

    def del_friction(self, id):

        doc = self._get_doc()

        def remove_curve(doc, id):
            for fig in ("friction", "velocity"):
                p = doc.context["figures"].get(fig)
                renderer = doc.context["renderers"][fig].pop(id, None)
                if renderer is not None:
                    p.renderers.remove(renderer)

        doc.add_next_tick_callback(lambda: remove_curve(doc, id))
        assert self.server is not None
        self.server.io_loop.add_callback(lambda: None)  # Trick to "wake up" thread

        assert self.app is not None
        self.app.logger.debug(f"Removed friction curve {id}")

        pass

    def clear_plot(self):
        assert self.app is not None
        if not self.server:
            self.app.logger.debug("Server not initialised")
            return

        self.plot([[], [], []])

        doc = self._get_doc()
        context = getattr(doc, "context", None)
        if context and "renderers" in context:
            ids = list(context["renderers"]["friction"].keys())
            for renderer_id in ids:
                self.del_friction(renderer_id)

        self.app.logger.debug("Plot cleared")
