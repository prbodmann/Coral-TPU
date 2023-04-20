from enum import IntEnum

class Logger:
    class Level(IntEnum):
        ERROR = 0
        INFO = 1
        TIMING = 2
        DEBUG = 3

    level = Level.INFO

    @staticmethod
    def setLevel(level: Level):
        Logger.level = level

    @staticmethod
    def error(msg: str, ex: Exception = None):
        if Logger.level >= Logger.Level.ERROR:
            if ex:
                print(f"ERROR: {msg}", ex)
            else:
                print(f"ERROR: {msg}")

    @staticmethod
    def info(msg: str, ex: Exception = None):
        if Logger.level >= Logger.Level.INFO:
            if ex:
                print(f"INFO: {msg}", ex)
            else:
                print(f"INFO: {msg}")
            
    @staticmethod
    def timing(msg: str, seconds: float):
        if Logger.level >= Logger.Level.TIMING:
            print(f"TIMING: {msg}: {seconds} s")

    @staticmethod
    def debug(msg: str):
        if Logger.level >= Logger.Level.DEBUG:
            print(f"DEBUG: {msg}")