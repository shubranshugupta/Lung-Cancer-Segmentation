class FolderNotFoundError(Exception):
    def __init__(self, message):
        self.message = message
    def __str__(self):
        return self.message

class AWSCredentialError(Exception):
    def __init__(self, message):
        self.message = message
    def __str__(self):
        return self.message

class DownloadDataError(Exception):
    def __init__(self, message):
        self.message = message
    def __str__(self):
        return self.message

class MyNoCredentialError(Exception):
    def __init__(self, message):
        self.message = message
    def __str__(self):
        return self.message