import os
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask


class FileHandler:

    data_file: Path | None = None
    data: pd.DataFrame | None = None
    app: Flask | None = None

    def __init__(self) -> None:
        pass

    def init_app(self, app: Flask) -> None:
        app.extensions["file_handler"] = self
        self.app = app
        upload_dir = app.config["UPLOAD_FOLDER"]
        assert (upload_dir is not None) and len(upload_dir) > 0
        try:
            data_file = Path(upload_dir) / "data.csv"
            if data_file.is_file():
                self.data_file = data_file
        except Exception as e:
            # Unfortunately, the logger is not yet initialised at this point...
            print(e)
            pass

    def save(self, file) -> bool:

        app = self.app
        assert app is not None, "Flask app is not initialised"

        try:
            app.logger.debug(f"Received {file.filename}")
            # Save file to disk
            file.save(self.data_file)
            app.logger.debug("File saved")
            # Check that file contents are consistent
            if not self._check_before_save():
                app.logger.error("Uploaded file does not meet criteria")
                raise IOError
            # Thumbs up in log
            app.logger.info("File upload successful")

        except Exception as e:
            # Thumbs down in log
            print(e)
            app.logger.error("File upload failed")
            # Remove uploaded file
            self.clear_all()
            return False

        return True

    def check_data_exists(self):
        if self.data_file == None:
            return False
        return self.data_file.is_file()

    def load_data(self):
        assert self.app is not None
        assert self.data_file is not None
        self.app.logger.debug("Loading data...")
        self.data = pd.read_csv(self.data_file)
        return self.data

    def clear_all(self):
        data_file = self.data_file
        assert self.app is not None
        assert data_file is not None
        # Check that a file path is set and that the file exists
        if data_file and data_file.is_file():
            os.remove(data_file)
            self.app.logger.debug(f"Deleted {data_file}")

    def _check_before_save(self) -> bool:
        # TODO: expand initial checks
        try:
            data = self.load_data()
            assert isinstance(data, pd.DataFrame)
            return True
        except Exception as e:
            print(e)
            return False
        return False
