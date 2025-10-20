from bokeh.embed import server_document
from flask import Blueprint, current_app, jsonify, render_template, request

from .models import LogEntry, StepEvent, db

bp = Blueprint("main", __name__)


@bp.route("/")
def index():

    # Get Bokeh canvas
    bokeh_url = "http://127.0.0.1:5006/bkapp"
    bokeh_script = server_document(bokeh_url)
    current_app.logger.debug(f"Bokeh canvas loaded")

    # Get velocity steps
    vsteps = StepEvent.query.order_by(StepEvent.start.asc()).all()
    current_app.logger.debug(f"Got {len(vsteps)} events")

    # If no steps are found, create a "default" step with
    # ID -1 (will get replaced by assigned ID later)
    if len(vsteps) == 0:
        vstep_dict = {
            "-1": {
                "start": 0,
                "stop": 100,
                "v0": 1e-6,
                "v1": 1e-5,
            }
        }
    else:
        # Return dict buffer
        vstep_dict = {}
        # Loop over steps
        for step in vsteps:
            # Create dict entry per step
            vstep_dict[str(step.id)] = {
                "start": step.start,
                "stop": step.stop,
                "v0": step.v0,
                "v1": step.v1,
            }

    current_app.logger.debug(f"Velocity steps loaded")

    return render_template("index.html", bokeh_script=bokeh_script, vsteps=vstep_dict)


@bp.route("/update-step", methods=["POST"])
def update_step():

    action = request.form.get("action")
    id = request.form.get("step_id", type=int)

    app = current_app

    if action == "update":
        if id == -1:
            # New entry: insert into DB
            # Debug log entry
            ...
        else:
            # Attempt to update existing entry

            # Check that the key exists in DB (otherwise return error)
            if db.session.get(StepEvent, id) == None:
                app.logger.error(f"Cannot find ID {id} in database")
                return jsonify({"status": "error", "message": ""}), 500

            # Update entry
            # Debug log entry
            ...

    if action == "delete":
        # Skip blank row
        if id == -1:
            return jsonify({"status": "ok", "message": ""}), 200

        # Assert that ID exists
        # Delete entry from DB

    # Run forward model (eventually)
    # Return updated values to back-populate HTML

    # Make sure to update ID when a new entry is created!
    # Also automatically add new row in case of new entry

    # In the case of deletion, AJAX delete #form_vstep_{{ id }}
    # except if id == -1

    return jsonify({"status": "ok", "message": ""}), 200


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
    except Exception:
        db.session.rollback()
        current_app.logger.error("Failed to clear log entries")
        return jsonify({"status": "error", "message": f"{Exception}"}), 500


@bp.route("/clear_data", methods=["POST"])
def clear_data():
    try:
        current_app.extensions["plot_handler"].clear_plot()
        current_app.extensions["file_handler"].clear_all()
        return jsonify({"status": "ok", "message": ""}), 200
    except Exception:
        current_app.logger.error("Failed to clear data")
        return jsonify({"status": "error", "message": f"{Exception}"}), 500


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
