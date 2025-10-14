from datetime import datetime, timezone
from typing import Optional

from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


db = SQLAlchemy(model_class=Base)


class LogEntry(db.Model):
    __tablename__ = "logs"

    id: Mapped[int] = mapped_column(primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(
        default=datetime.now(timezone.utc), nullable=False
    )
    level: Mapped[str] = mapped_column(String(10), nullable=False)
    msg: Mapped[str] = mapped_column(Text, nullable=False)
    trace: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    def __repr__(self):
        return f"<LogEntry {self.timestamp} {self.level}: {self.msg}>"


class StepEvent(db.Model):
    __tablename__ = "vsteps"

    id: Mapped[int] = mapped_column(primary_key=True)
    # Columns are `nullable` by default, so no need to
    # explicitly specify `mapped_column(nullable=True)`
    start: Mapped[Optional[int]]
    stop: Mapped[Optional[int]]
    v0: Mapped[Optional[float]]
    v1: Mapped[Optional[float]]
