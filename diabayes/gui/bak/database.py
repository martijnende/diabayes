from datetime import datetime

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    pass


db = SQLAlchemy(model_class=Base)


class Log(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    datetime = db.Column(db.DateTime, nullable=False, server_default=func.now())
    level = db.Column(db.String(100), nullable=False)
    msg = db.Column(db.Text, nullable=False)
    trace = db.Column(db.Text, nullable=True)

    __table_args__ = ({"sqlite_autoincrement": True},)

    def __repr__(self):
        return (
            f"{datetime.strftime(self.datetime, '%H:%M:%S')} [{self.level}] {self.msg}"
        )
