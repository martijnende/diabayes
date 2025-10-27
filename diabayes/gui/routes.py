from bokeh.client import pull_session
from bokeh.embed import server_session
from flask import Blueprint, current_app, jsonify, render_template, request

from .models import LogEntry, StepEvent, db
from .physics import data_are_valid, run_forward

bp = Blueprint("main", __name__)

BOKEH_URL = "http://127.0.0.1:5006/bkapp"


@bp.route("/")
def index():

    fhandler = current_app.extensions["file_handler"]
    phandler = current_app.extensions["plot_handler"]

    current_sessions = phandler.server.get_sessions("/bkapp")
    if len(current_sessions) == 0:
        bokeh_session = pull_session(url=BOKEH_URL)
    else:
        if len(current_sessions) > 1:
            current_app.logger.warning(
                "Found more than one Bokeh session, which is not expected"
            )
        bokeh_session = current_sessions[0]

    data_ready = fhandler.check_data_exists()
    if data_ready:
        data = fhandler.load_data()
        current_app.logger.debug("Reloaded data")
        phandler.plot(data)
        current_app.logger.debug("Reloaded plots")

    # Get velocity steps
    vsteps = StepEvent.query.order_by(StepEvent.start.asc()).all()
    current_app.logger.debug(f"Got {len(vsteps)} events")

    # Get Bokeh canvas
    bokeh_script = server_session(session_id=bokeh_session.id, url=BOKEH_URL)
    current_app.logger.debug(f"Bokeh canvas loaded")

    return render_template(
        "index.html", bokeh_script=bokeh_script, vsteps=vsteps, data_ready=data_ready
    )


@bp.route("/update-step", methods=["POST"])
def update_step():

    # Get the form data and do some light validation
    # Everything defaults to None if no value is found
    # or if it fails to validate
    action = request.form.get("action", type=str)
    id = request.form.get("id", type=int)
    start = request.form.get("start", type=int)
    stop = request.form.get("stop", type=int)

    # Instead of manually manipulating each quantity,
    # loop over a dictionary instead
    field_names = ("v0", "v1", "mu0", "k", "a", "b", "Dc")
    fields = {key: request.form.get(key, type=float) for key in field_names}

    # Extract/calculate theta0
    # Compute k/kc

    app = current_app

    # Action 1: add a new v-step
    if action == "add":
        try:
            step = StepEvent(start=start, stop=stop, **fields)  # type: ignore
            db.session.add(step)
            db.session.commit()
            id = step.id
            app.logger.debug(f"Added v-step {id} {start} -> {stop}")
        except Exception as e:
            db.session.rollback()
            app.logger.error("Failed to insert v-step entry")
            app.logger.error(e)
            return jsonify({"status": "error", "message": e}), 500

    # Action 2: update an existing v-step
    elif action == "update":

        # Get the step based on the provided ID
        # Will return None if id cannot be found
        step = db.session.get(StepEvent, id)

        # Check that the key exists in DB (otherwise return error)
        if step == None:
            app.logger.error(f"Cannot find ID {id} in database")
            return jsonify({"status": "error", "message": ""}), 500

        # Update entry
        try:
            step.start = start
            step.stop = stop
            for key, val in fields.items():
                setattr(step, key, val)
            db.session.commit()
            app.logger.debug(f"Updated v-step {id}")
        except Exception as e:
            db.session.rollback()
            app.logger.error(f"Failed to update v-step {id}")
            app.logger.error(e)
            return jsonify({"status": "error", "message": e}), 500

    # Action 3: delete a v-step
    elif action == "delete":

        # Get the step based on the provided ID
        # Will return None if id cannot be found
        step = db.session.get(StepEvent, id)

        # Check that the key exists in DB (otherwise return error)
        if step == None:
            app.logger.error(f"Cannot find ID {id} in database")
            return jsonify({"status": "error", "message": ""}), 500

        # Delete entry from DB
        try:
            StepEvent.query.filter_by(id=id).delete()
            db.session.commit()
            # Remove any plot elements associated with id
            app.extensions["plot_handler"].del_friction(id)
            app.logger.debug(f"Deleted v-step {id}")
        except Exception as e:
            db.session.rollback()
            app.logger.error("Failed to delete v-step entry")
            app.logger.error(e)

    # Get all the current steps
    steps = StepEvent.query.all()
    # Render the HTML template
    html = render_template("steps.html", vsteps=steps)

    # At this point, id cannot be None; either it was
    # provided (update/delete), or it was created (add)
    assert id is not None

    # Check if data has been loaded
    data = getattr(app.extensions["file_handler"], "data", None)

    # Update model curves
    if (action != "delete") and (data is not None):

        assert len(data) > 0
        fields["t"] = app.extensions["file_handler"].data[0]

        # Check that all data are valid
        if data_are_valid(start, stop, fields):
            # Run forward model
            friction, v = run_forward(start, stop, fields)
            # Plot friction curves
            plot_fields = {
                "t": fields["t"][start:stop],
                "mu": friction,
                "v": v,
            }
            app.extensions["plot_handler"].add_friction(id, plot_fields)
        # If any data are invalid: remove curves
        else:
            app.logger.debug(f"Validation for step {id} failed")
            app.extensions["plot_handler"].del_friction(id)

    return jsonify({"status": "ok", "message": "", "html": html}), 200


@bp.route("/upload", methods=["POST"])
def upload_file():

    fhandler = current_app.extensions["file_handler"]
    phandler = current_app.extensions["plot_handler"]

    # Get data file from request
    file = request.files["data_file"]
    # Attempt to save the file
    # This will perform checks to ensure readability
    if not fhandler.save(file):
        # An exception occurred, raise status 500
        return jsonify({"status": "error", "message": f"{Exception}"}), 500

    # Load the data (already checked in previous step)
    data = fhandler.load_data()

    # Pass data to plotter
    phandler.plot(data)

    # All good (status 200)
    return jsonify({"status": "ok", "message": ""}), 200


@bp.route("/clear_logs", methods=["POST"])
def clear_logs():
    try:
        rows_deleted = db.session.query(LogEntry).delete()
        db.session.commit()
        current_app.logger.info(f"{rows_deleted} entries deleted")
        return jsonify({"status": "ok", "message": ""}), 200
    except Exception as e:
        db.session.rollback()
        current_app.logger.error("Failed to clear log entries")
        return jsonify({"status": "error", "message": f"{e}"}), 500


@bp.route("/clear_data", methods=["POST"])
def clear_data():
    try:
        current_app.extensions["plot_handler"].clear_plot()
        current_app.extensions["file_handler"].clear_all()
        return jsonify({"status": "ok", "message": ""}), 200
    except Exception as e:
        print(f"{e}")
        current_app.logger.error("Failed to clear data")
        return jsonify({"status": "error", "message": f"{e}"}), 500


@bp.route("/logs/all")
def check_db():
    entries = LogEntry.query.order_by(LogEntry.timestamp.desc()).all()
    return jsonify(
        [
            {
                "id": e.id,
                "timestamp": e.timestamp.astimezone().strftime("%Y-%d-%m %H:%M:%S"),
                "level": e.level.lower(),
                "msg": e.msg,
            }
            for e in entries
        ]
    )
