_FOREGROUND = "\033[38;5;%sm"
_BACKGROUND = "\033[48;5;%sm"

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

def combo(foreground, background):
    return _BACKGROUND % background + _FOREGROUND % foreground

COLOURS = {
    "Z": C_NORMAL,
    "g": GREEN,
    "G": DARK_GREEN,
    "r": RED,
    "R": DARK_RED,
    "M": MAGENTA,
    "P": PURPLE,
    "C": CYAN,
    "Y": YELLOW,
    "W": WHITE,
}

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

SCALE_30 = [_BACKGROUND % '16' + _FOREGROUND % x
            for x in
            (17, 17, 18, 18, 19, 19,
             57, 56, 55, 54, 53, 52,
             90, 89, 88, 160, 196, 202,
             208, 214, 220, 226, 190, 154,
             118, 82, 46, 48, 49, 51)]

SCALE_12 = [COLOURS[x] for x in 'PPrRYYGGgCCW']
SCALE_11 = SCALE_12[:-1]


def colouriser(colour_scale):
    c_scale = len(colour_scale) * 0.9999
    c_max = int(c_scale)
    def colourise(val):
        i = min(int(val * c_scale), c_max)
        return colour_scale[max(i, 0)]
    return colourise
