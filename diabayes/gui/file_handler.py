import os

import numpy as np
from flask import current_app


class FileHandler:

    data_file = None

    def __init__(self) -> None:
        pass

    def save(self, file) -> bool:
        try:
            # Get the upload folder from config
            upload_folder = current_app.config["UPLOAD_FOLDER"]
            # Define absolute path
            path = os.path.join(upload_folder, file.filename)
            # Save file to disk
            file.save(path)
            self.data_file = path
            # Check that file contents are consistent
            if not self._check_before_save():
                current_app.logger.error("Uploaded file does not meet criteria")
                raise IOError
            # Thumbs up in log
            current_app.logger.info("File upload successful")

        except Exception:
            # Thumbs down in log
            current_app.logger.error("File upload failed")
            # Remove uploaded file
            self.clear_all()
            return False

        return True

    def load_data(self):
        return np.load(self.data_file)

    def clear_all(self):
        # Check that a file path is set and that the file exists
        if self.data_file and os.path.isfile(self.data_file):
            os.remove(self.data_file)
        self.data_file = None

    def _check_before_save(self) -> bool:
        try:
            data = self.load_data()
            assert isinstance(data, np.ndarray)
            return True
        except Exception:
            return False
        return False
