import logging
import traceback
from datetime import datetime

from .app import db, socketio
from .database import Log

# From https://matthewmoisen.com/blog/how-to-log-to-a-database-with-flask/


def format(entry):
    time = datetime.strftime(entry.datetime, "%H:%M:%S")
    level = f'<span class="log-level-{entry.level.lower()}">{entry.level}</span>'
    # TODO: add formatting for stack trace
    return f"{time} [{level}] {entry.msg}"


def get_logs():
    log_entries = Log.query.order_by(Log.datetime.desc()).all()
    return [format(entry) for entry in log_entries]


class SQlAlchemyHandler(logging.Handler):

    def emit(self, record):
        print(record)
        trace = None
        exc = record.__dict__["exc_info"]
        if exc:
            trace = traceback.format_exc(exc)

        log = Log(
            level=record.__dict__["levelname"],
            msg=record.__dict__["msg"],
            trace=trace,
        )
        db.session.add(log)
        db.session.commit()
        socketio.emit("new_log", format(log))
