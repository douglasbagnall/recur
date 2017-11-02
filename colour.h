/*Colour strings for ANSI compatible terminals.*/
#ifndef __COLOUR_H__
#define __COLOUR_H__

#include <stdbool.h>

#define C_NORMAL  "\033[00m"
#define BG_NORMAL  "\043[00m"
#define C_DARK_RED  "\033[00;31m"
#define C_RED "\033[01;31m"
#define C_DARK_GREEN  "\033[00;32m"
#define C_GREEN  "\033[01;32m"
#define C_YELLOW  "\033[01;33m"
#define C_DARK_YELLOW  "\033[00;33m"
#define C_DARK_BLUE  "\033[00;34m"
#define C_BLUE  "\033[01;34m"
#define C_PURPLE  "\033[00;35m"
#define C_MAGENTA  "\033[01;35m"
#define C_DARK_CYAN  "\033[00;36m"
#define C_CYAN  "\033[01;36m"
#define C_GREY  "\033[00;37m"
#define C_WHITE  "\033[01;37m"

#define C_REV_RED "\033[01;41m"

#define C_ITALIC "\x1B[3m"
#define C_STANDARD "\x1B[23m"


const char * colourise_float01(float f, bool rev);

#endif
