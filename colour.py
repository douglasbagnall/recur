
C_NORMAL    = "\033[00m"
DARK_RED    = "\033[00;31m"
RED         = "\033[01;31m"
DARK_GREEN  = "\033[00;32m"
GREEN       = "\033[01;32m"
YELLOW      = "\033[01;33m"
DARK_YELLOW = "\033[00;33m"
DARK_BLUE   = "\033[00;34m"
BLUE        = "\033[01;34m"
PURPLE      = "\033[00;35m"
MAGENTA     = "\033[01;35m"
DARK_CYAN   = "\033[00;36m"
CYAN        = "\033[01;36m"
GREY        = "\033[00;37m"
WHITE       = "\033[01;37m"

REV_RED     = "\033[01;41m"

_FOREGROUND = "\033[38;5;%sm"
_BACKGROUND = "\033[48;5;%sm"

_spectrum = (range(160, 196, 6) +
             range(226, 190, -6) +
             range(124, 128, 1) +
             range(128, 164, 6) +
             range(122, 90, -6) +
             range(91, 88, -1) +
             range(161, 166, 1) +
             range(201, 196, -1) +
             range(201, 196, -1) +
             range(130, 160, 6) +
             range(118, 88, -6))

SPECTRUM = [_FOREGROUND % x for x in _spectrum]
BACKGROUND_SPECTRUM = [_BACKGROUND % x for x in _spectrum]


def get_namespace(use_colour='auto'):
    import sys
    class EmptyThing(object):
        pass
    ns = EmptyThing()

    if use_colour.lower() in ('yes', 'true', 'y', 'always'):
        c = True
    elif use_colour.lower() in ('no', 'false', 'n', 'never'):
        c = False
    else:
        c = sys.stdout.isatty()

    for k, v in globals().items():
        if k.isupper():
            setattr(ns, k, (v if c else ''))

    return ns
