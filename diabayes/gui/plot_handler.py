from collections import namedtuple
import numpy as np
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, RangeTool
from bokeh.plotting import figure
from bokeh.server.server import Server
from flask import Flask

bokeh_url = "http://127.0.0.1:5006/"


class PlotHandler:

    app: Flask | None = None
    server: Server | None = None

    def __init__(self) -> None:
        pass

    def init_app(self, app: Flask) -> None:
        app.extensions["plot_handler"] = self
        self.app = app

    def make_bokeh_doc(self, doc):

        line_colour = "#3cb371"
        overlay_colour = "#f5deb3"
        line_width = 3.0

        empty = np.array([], dtype=float)
        source = ColumnDataSource(dict(x=empty.copy(), y=empty.copy()))
        source2 = ColumnDataSource(dict(x=empty.copy(), y=empty.copy()))

        # Figure showing friction data
        p = figure(
            height=300,
            output_backend="webgl",
            tools="hover,pan,box_zoom,xwheel_zoom,ywheel_zoom,undo,redo,reset",
            sizing_mode="stretch_width",
            tooltips=[
                ("index", "$index"),
                ("displacement", "$snap_x"),
                ("friction", "$snap_y"),
            ],
        )
        p.line("x", "y", line_color=line_colour, source=source, line_width=line_width)
        p.x_range.range_padding = 0.0  # type: ignore
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
                ("displacement", "$snap_x"),
                ("velocity", "$snap_y"),
            ],
        )
        q.line("x", "y", line_color=line_colour, source=source2, line_width=line_width)
        q.x_range.range_padding = 0.0  # type: ignore
        q.yaxis.axis_label = "Velocity [m/s]"
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
        select.x_range.range_padding = 0.0  # type: ignore
        select.x_range.bounds = "auto"  # type: ignore
        select.xaxis.axis_label = "Load-point displacement [mm]"

        range_tool = RangeTool(
            x_range=p.x_range, y_range=p.y_range, start_gesture="pan"
        )
        range_tool.overlay.fill_color = overlay_colour
        range_tool.overlay.fill_alpha = 0.3

        select.line("x", "y", line_color=line_colour, line_width=3, source=source)
        # select.ygrid.grid_line_color = None
        select.add_tools(range_tool)

        doc.add_root(column([q, p, select], sizing_mode="stretch_width"))
        doc.context = {
            "figures": {"friction": p, "velocity": q},
            "renderers": {"friction": {}, "velocity": {}},
            "sources": {"friction": source, "velocity": source2},
        }

        pass

    def plot(self, data):

        assert self.server is not None, "Server not initialised"
        current_sessions = self.server.get_sessions("/")
        assert len(current_sessions) > 0, "Session not initialised"

        for session in current_sessions:

            doc = session.document
            ctx = getattr(doc, "context")
            if ctx is None:
                continue

            sources = ctx.get("sources")
            if sources is None:
                continue

            source = sources["friction"]
            source2 = sources["velocity"]

            def create_callbacks(src1, src2):

                # Update data sources
                def update_friction():
                    src1.data = dict(x=np.array(data.t), y=np.array(data.mu))

                def update_velocity():
                    src2.data = dict(x=np.array(data.t), y=np.array(data.v_lp))

                return update_friction, update_velocity

            # Add callbacks
            update_fric, update_vel = create_callbacks(source, source2)
            doc.add_next_tick_callback(update_fric)
            doc.add_next_tick_callback(update_vel)

        self.server.io_loop.add_callback(lambda: None)  # Trick to "wake up" thread

    def add_friction(self, id, data):

        assert self.server is not None, "Server not initialised"
        current_sessions = self.server.get_sessions("/")
        assert len(current_sessions) > 0, "Session not initialised"

        for session in current_sessions:

            doc = session.document
            ctx = getattr(doc, "context")
            if ctx is None:
                continue

            def create_add_callback(current_doc, curve_id):
                def add_curve():
                    for fig, y in zip(
                        ("friction", "velocity"), (data["mu"], data["v"])
                    ):
                        p = current_doc.context["figures"].get(fig)
                        renderers = current_doc.context["renderers"].get(fig)
                        if p is not None:
                            # If a curve with this ID already
                            # exists, remove it first
                            if curve_id in renderers:
                                renderer = renderers.pop(curve_id)
                                p.renderers.remove(renderer)

                            source = ColumnDataSource(dict(x=data["x"], y=y))
                            renderer = p.line(
                                "x",
                                "y",
                                line_color="orange",
                                source=source,
                                line_width=3,
                            )
                            current_doc.context["renderers"][fig][curve_id] = renderer

                return add_curve

            callback = create_add_callback(doc, id)
            doc.add_next_tick_callback(callback)

        self.server.io_loop.add_callback(lambda: None)  # Trick to "wake up" thread

        assert self.app is not None
        self.app.logger.debug(f"Added friction curve {id}")

        pass

    def del_friction(self, id):

        assert self.server is not None, "Server not initialised"
        current_sessions = self.server.get_sessions("/")
        assert len(current_sessions) > 0, "Session not initialised"

        for session in current_sessions:

            doc = session.document
            ctx = getattr(doc, "context")
            if ctx is None:
                continue

            def create_remove_callback(current_doc, curve_id):
                def remove_curve():
                    for fig in ("friction", "velocity"):
                        p = current_doc.context["figures"].get(fig)
                        renderer = current_doc.context["renderers"][fig].pop(
                            curve_id, None
                        )
                        if renderer is not None:
                            p.renderers.remove(renderer)

                return remove_curve

            callback = create_remove_callback(doc, id)
            doc.add_next_tick_callback(callback)

        self.server.io_loop.add_callback(lambda: None)  # Trick to "wake up" thread

        assert self.app is not None
        self.app.logger.debug(f"Removed friction curve {id}")

        pass

    def clear_plot(self):
        assert self.app is not None
        if not self.server:
            self.app.logger.debug("Server not initialised")
            return

        current_sessions = self.server.get_sessions("/")
        assert len(current_sessions) > 0, "Session not initialised"

        empty_data = namedtuple("empty", ("t", "mu", "v_lp"))
        empty = np.array([], dtype=float)
        self.plot(empty_data(mu=empty.copy(), t=empty.copy(), v_lp=empty.copy()))

        curve_ids = set()

        for session in current_sessions:
            doc = session.document
            ctx = getattr(doc, "context")
            if ctx is None:
                continue
            curves = ctx.get("renderers")
            if curves is not None:
                friction_curves = curves.get("friction", {})
                for curve_id in friction_curves:
                    curve_ids.add(curve_id)

        for curve_id in curve_ids:
            self.del_friction(curve_id)

        self.app.logger.debug("Plot cleared")
