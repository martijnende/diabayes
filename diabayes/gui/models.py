from datetime import datetime
from typing import Optional

from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import DateTime, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, MappedAsDataclass, mapped_column

"""
NOTE: pyright is broken as fuck with SQLAlchemy's v2.0 ORM declarations.
Following online guides on using MappedAsDataclass etc. don't work, so
the only solution is to put `# type: ignore` everywhere.
Thanks Obama...
"""


class Base(MappedAsDataclass, DeclarativeBase):
    pass


db = SQLAlchemy(model_class=Base)


class LogEntry(db.Model):
    __tablename__ = "logs"

    id: Mapped[int] = mapped_column(init=False, primary_key=True)
    level: Mapped[str] = mapped_column(String(10), nullable=False)
    msg: Mapped[str] = mapped_column(Text, nullable=False)
    trace: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=func.now(), nullable=False
    )

    def __repr__(self):
        return f"<LogEntry {self.timestamp} {self.level}: {self.msg}>"


class StepEvent(db.Model):
    __tablename__ = "vsteps"

    id: Mapped[int] = mapped_column(init=False, primary_key=True)
    # Columns are `nullable` by default, so no need to
    # explicitly specify `mapped_column(nullable=True)`
    start: Mapped[Optional[int]]
    stop: Mapped[Optional[int]]
    v0: Mapped[Optional[float]]
    v1: Mapped[Optional[float]]
    mu0: Mapped[Optional[float]]
    theta0: Mapped[Optional[float]]
    k: Mapped[Optional[float]]
    a: Mapped[Optional[float]]
    b: Mapped[Optional[float]]
    Dc: Mapped[Optional[float]]
