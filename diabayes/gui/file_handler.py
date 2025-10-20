import os
from pathlib import Path

import numpy as np
from flask import Flask


class FileHandler:

    data_file: Path | None = None
    app: Flask | None = None

    def __init__(self) -> None:
        pass

    def init_app(self, app: Flask) -> None:
        app.extensions["file_handler"] = self
        self.app = app

    def save(self, file) -> bool:

        app = self.app
        assert app is not None, "Flask app is not initialised"

        try:
            # Get the upload folder from config
            upload_folder = app.config["UPLOAD_FOLDER"]
            app.logger.debug(f"Received {file.filename}")
            # Define absolute path
            path = os.path.join(upload_folder, file.filename)
            # Save file to disk
            file.save(path)
            app.logger.debug("File saved")
            self.data_file = path
            # Check that file contents are consistent
            if not self._check_before_save():
                app.logger.error("Uploaded file does not meet criteria")
                raise IOError
            # Thumbs up in log
            app.logger.info("File upload successful")

        except Exception:
            # Thumbs down in log
            app.logger.error("File upload failed")
            # Remove uploaded file
            self.clear_all()
            return False

        return True

    def load_data(self):
        assert self.app is not None
        assert self.data_file is not None
        self.app.logger.debug("Loading data...")
        return np.load(self.data_file)

    def clear_all(self):
        assert self.app is not None
        assert self.data_file is not None
        # Check that a file path is set and that the file exists
        if self.data_file and os.path.isfile(self.data_file):
            os.remove(self.data_file)
            self.app.logger.debug(f"Deleted {self.data_file}")
        self.data_file = None

    def _check_before_save(self) -> bool:
        # TODO: expand initial checks
        try:
            data = self.load_data()
            assert isinstance(data, np.ndarray)
            return True
        except Exception:
            return False
        return False
