import os
basedir = os.path.abspath(os.path.dirname(__file__))

class Config:
    # Flask
    SECRET_KEY = os.environ.get("SECRET_KEY") or "secret_key"

    # Database (SQLite in instance/ directory)
    SQLALCHEMY_DATABASE_URI = (
        os.environ.get("DATABASE_URL")
        or f"sqlite:///{os.path.join(basedir, 'instance', 'app.db')}"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # File uploads
    UPLOAD_FOLDER = os.path.join(basedir, "instance", "uploads")
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB upload limit

    # Logging (basic setup for file logging, can expand)
    LOG_FILE = os.path.join(basedir, "instance", "app.log")

