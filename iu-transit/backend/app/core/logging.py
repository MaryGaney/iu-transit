"""
app/core/logging.py
───────────────────
Structured logging setup. Import `logger` from here everywhere.

Only iu_transit logger shows DEBUG when DEBUG=true.
All third-party libraries (SQLAlchemy, aiosqlite, httpcore, httpx) stay at WARNING.
"""

import logging
import sys


def setup_logging(debug: bool = False) -> None:
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

    # Root logger at WARNING — catches anything not explicitly configured
    logging.basicConfig(
        level=logging.WARNING,
        format=fmt,
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Our app logger — INFO normally, DEBUG when debug=True
    app_logger = logging.getLogger("iu_transit")
    app_logger.setLevel(logging.DEBUG if debug else logging.INFO)
    app_logger.propagate = False
    if not app_logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter(fmt))
        app_logger.addHandler(h)

    # Explicitly silence noisy third-party loggers regardless of root level
    noisy = [
        "sqlalchemy",
        "sqlalchemy.engine",
        "sqlalchemy.engine.Engine",
        "aiosqlite",
        "apscheduler",
        "apscheduler.scheduler",
        "apscheduler.executors",
        "httpx",
        "httpcore",
        "httpcore.connection",
        "httpcore.http11",
        "uvicorn.access",
    ]
    for name in noisy:
        logging.getLogger(name).setLevel(logging.WARNING)


logger = logging.getLogger("iu_transit")
