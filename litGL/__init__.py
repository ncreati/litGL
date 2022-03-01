""" LitGL package initialization file.

    Author:
        - 2020-2021 Nicola Creati

        - 2020-2021 Roberto Vidmar

    Copyright:
        2020-2021 Nicola Creati <ncreati@inogs.it>

        2020-2021 Roberto Vidmar <rvidmar@inogs.it>

    License:
        MIT/X11 License (see
        :download:`license.txt <../../../license.txt>`)
"""
import logging
import warnings
try:
    from rich.traceback import install
except ImportError:
    pass
else:
    install()

# -----------------------------------------------------------------------------
def namedLogger(name, cls=None):
    """ Return a new logger instance with the given name.
        Set also the NullHandler to the instance.

        Args:
            name (str): logger name
            cls (class): logger class (if any)

        Returns:
            :class:`logging.Logger`: logger instance
    """
    if cls:
        logger = logging.getLogger(name + '.' + cls.__name__)
    else:
        logger = logging.getLogger(name)
    logger.addHandler(logging.NullHandler())
    return logger

# -----------------------------------------------------------------------------
def custom_formatwarning(msg, *args, **kargs):
    """ Custom warning formatter.

        Returns:
            str: warning message
    """
    return "UserWarning: " + str(msg) + '\n'

warnings.formatwarning = custom_formatwarning
