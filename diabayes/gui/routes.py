from bokeh.embed import server_document
from flask import Blueprint, current_app, jsonify, render_template, request

from .models import LogEntry, db

bp = Blueprint("main", __name__)


@bp.route("/")
def index():
    bokeh_url = "http://127.0.0.1:5006/bkapp"
    bokeh_script = server_document(bokeh_url)
    return render_template("index.html", bokeh_script=bokeh_script)


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


@bp.route("/clear_data", methods=["POST"])
def clear_data():
    try:
        rows_deleted = db.session.query(LogEntry).delete()
        db.session.commit()
        current_app.logger.info(f"{rows_deleted} entries deleted")
        current_app.extensions["plot_handler"].clear_plot()
        return jsonify({"status": "ok", "message": ""}), 200
    except Exception:
        db.session.rollback()
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
