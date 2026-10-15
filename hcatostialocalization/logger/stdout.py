import os
from termcolor import colored

from hcatostialocalization.config.config import config

def log(text, out=False):
    text = colored(text, tuple(config.logging.colors.log))
    if out:
        return text
    else:
        print(text)

def debug(text, out=False):
    text = colored(text, tuple(config.logging.colors.debug))
    if out:
        return text
    else:
        print(text)

def error(text, out=False):
    text = colored(text, tuple(config.logging.colors.error), attrs=["bold"])
    if out:
        return text
    else:
        print(text)

def printdbg(*args, **kwargs):
    if int(os.environ.get('DEBUG_MODE', 0)):
        print(debug("DEBUG_MODE:", out=True), *args, **kwargs)

if __name__ == '__main__':
    log("This is a log")
    debug("this is some debug")
    error("This is some error")
    raise RuntimeError(error("This is an error inside a 'raise' statement.", True))
