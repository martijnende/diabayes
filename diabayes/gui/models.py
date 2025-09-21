from datetime import datetime

from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class LogEntry(db.Model):
    __tablename__ = "logs"

    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    level = db.Column(db.String(10), nullable=False)
    msg = db.Column(db.Text, nullable=False)
    trace = db.Column(db.Text, nullable=True)

    def __repr__(self):
        return f"<LogEntry {self.timestamp} {self.level}: {self.msg}>"
