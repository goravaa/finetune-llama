import logging, sys, time, os

_EMOJI_LEVEL = {
    logging.DEBUG: "🔍",
    logging.INFO:  "ℹ️ ",
    logging.WARNING: "⚠️ ",
    logging.ERROR: "💥",
    logging.CRITICAL: "🛑",
}

class EmojiFormatter(logging.Formatter):
    def format(self, record):
        emoji = _EMOJI_LEVEL.get(record.levelno, "❔")
        ts = time.strftime("%H:%M:%S")
        record.msg = f"{emoji}  [{ts}] {record.msg}"
        return super().format(record)

def setup_logger(name: str = "llama-trainer", level=logging.INFO):
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(EmojiFormatter())
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger
