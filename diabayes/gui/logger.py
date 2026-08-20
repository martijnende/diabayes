import logging
import traceback
from datetime import datetime

from .models import LogEntry, db

# From https://matthewmoisen.com/blog/how-to-log-to-a-database-with-flask/


def format(entry):
    time = datetime.strftime(entry.timestamp, "%H:%M:%S")
    level = f'<span class="log-level-{entry.level.lower()}">{entry.level}</span>'
    # TODO: add formatting for stack trace
    return f"{time} [{level}] {entry.msg}"


class SQLiteHandler(logging.Handler):

    def __init__(self, socketio):
        super().__init__()
        self.socketio = socketio

    def emit(self, record):
        trace = None
        exc = record.__dict__["exc_info"]
        if exc:
            trace = traceback.format_exc(exc)

        try:
            entry = LogEntry(
                level=record.__dict__["levelname"],  # type: ignore
                msg=record.__dict__["msg"],  # type: ignore
                trace=trace,  # type: ignore
            )
            db.session.add(entry)
            # Potentially DANGEROUS caveat:
            # When using this logger to log exceptions,
            # make sure to first session.rollback() before
            # calling emit(), for otherwise emit() will
            # do db.session.commit() before the rollback!
            db.session.commit()

            # Websocket broadcast
            self.socketio.emit(
                "new_log",
                {
                    "timestamp": entry.timestamp.strftime("%Y-%d-%m %H:%M:%S"),
                    "level": entry.level.lower(),
                    "msg": entry.msg,
                },
                namespace="/logs",
            )
        except Exception:
            db.session.rollback()
            logging.getLogger("sqlite_handler").exception("Failed to log to DB")
