all::

BASENAME = recur

GDB_ALWAYS_FLAGS = -ggdb -O3
WARNINGS = -Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers -ftree-vectorizer-verbose=0 -fno-inline
# -Wunsafe-loop-optimizations  -ftree-vectorizer-verbose=2  -fno-inline

GST_VERSION = 1.0

ARCH = $(shell arch)
ifeq "$(ARCH)" "x86_64"
ARCH_CFLAGS = -fPIC -DPIC -m64
else
ARCH_CFLAGS = -m32 -msse2
endif

LIB_ARCH_DIR = /usr/lib/$(ARCH)-linux-gnu
INC_ARCH_DIR = /usr/include/$(ARCH)-linux-gnu
INC_DIR = /usr/include

#CC = nccgen -ncgcc -ncld -ncfabs
#CC = /usr/bin/clang
#CC = /usr/local/bin/clang
#CC = /usr/local/bin/clang  -Weverything -Wno-documentation -Wno-system-headers -Wno-sign-conversion -Wno-conversion -Wno-gnu -Wno-variadic-macros -Wno-vla
#CLANG_FLAGS = -fslp-vectorize-aggressive
#CLANG_FLAGS =  -fplugin=/usr/lib/gcc/x86_64-linux-gnu/4.7/plugin/dragonegg.so

ALL_CFLAGS = -march=native -pthread $(VECTOR_FLAGS) $(WARNINGS) -pipe  -D_GNU_SOURCE -std=gnu1x $(INCLUDES) $(ARCH_CFLAGS) $(CFLAGS) $(GDB_ALWAYS_FLAGS) -ffast-math -funsafe-loop-optimizations $(CLANG_FLAGS) -std=gnu11
ALL_LDFLAGS = $(LDFLAGS)

$(BASENAME)_SRC = gst$(BASENAME).c

export GST_DEBUG = $(BASENAME):4

GST_INCLUDES =  -isystem $(INC_DIR)/gstreamer-$(GST_VERSION)\
	 -isystem $(INC_DIR)/glib-2.0\
	 -isystem $(LIB_ARCH_DIR)/glib-2.0/include\
	 -isystem $(INC_DIR)/glib-2.0/include\
	 -isystem /usr/include/libxml2

GTK_INCLUDES = -isystem /usr/include/gtk-3.0 \
	       -isystem /usr/include/pango-1.0/\
	       -isystem /usr/include/cairo/ \
	       -isystem /usr/lib/gdk \
	       -isystem /usr/include/gdk-pixbuf-2.0/ \
	       -isystem /usr/include/atk-1.0/


INCLUDES = -I. $(GST_INCLUDES)

LINKS = -L/usr/local/lib -lgstbase-$(GST_VERSION) -lgstreamer-$(GST_VERSION) \
	 -lgobject-2.0 -lglib-2.0 -lgstvideo-$(GST_VERSION) -lm -pthread -lrt \
	-lgmodule-2.0 -lgthread-2.0  -lgstfft-$(GST_VERSION) -lgstaudio-$(GST_VERSION) \
	 -lblas -lcdb

GTK_LINKS =  -lgtk-3 -lgdk-3


SOURCES =  gst$(BASENAME)_manager.c gst$(BASENAME)_audio.c gst$(BASENAME)_video.c \
	recur-context.c rescale.c recur-nn.c recur-nn-io.c context-recurse.c mfcc.c
OBJECTS := $(patsubst %.c,%.o,$(SOURCES))
#PLUGINS := $(patsubst %.c,lib%.so,$(SOURCES))


all:: libgstrecur.so

clean:
	rm -f *.so *.o *.a *.d *.s

pgm-clean:
	#find images -maxdepth 1 -name '*.p?m' | xargs rm -f
	rm -r images
	mkdir images

.c.o:
	$(CC)  -c -MMD $(ALL_CFLAGS) $(CPPFLAGS) -o $@ $<

#ASM_OPTS= -fverbose-asm
%.s:	%.c
	$(CC)  -S $(ASM_OPTS)  $(ALL_CFLAGS) $(CPPFLAGS) -o $@ $<

%.S:	%.c
	$(CC)  -S $(ASM_OPTS)  $(ALL_CFLAGS) $(CPPFLAGS) -fverbose-asm -o $@ $<

%.i:	%.c
	$(CC)  -E  $(ALL_CFLAGS) $(CPPFLAGS) -o $@ $<

%.c: %.h

NN_SPECIAL_FLAGS =  -fprefetch-loop-arrays

recur-nn.o: recur-nn.c
	$(CC)  -c -MMD $(ALL_CFLAGS) $(CPPFLAGS) $(NN_SPECIAL_FLAGS) -o $@ $<

libgstrecur.so: $(OBJECTS)
	$(CC) -shared -Wl,-O1 $+  $(INCLUDES) $(DEFINES)  $(LINKS) -Wl,-soname -Wl,$@ \
	  -o $@

libgstparrot.so: recur-nn.o recur-nn-io.o  mdct.o window.o gstparrot.o mfcc.o
	$(CC) -shared -Wl,-O1 $+  $(INCLUDES) $(DEFINES)  $(LINKS) -Wl,-soname -Wl,$@ \
	  -o $@

libgstclassify.so: recur-nn.o recur-nn-io.o gstclassify.o mfcc.o
	$(CC) -shared -Wl,-O1 $+  $(INCLUDES) $(DEFINES)  $(LINKS) -Wl,-soname -Wl,$@ \
	  -o $@

test_mfcc_table: %: recur-context.o recur-nn.o recur-nn-io.o rescale.o %.o
	$(CC) -Wl,-O1 $^   $(INCLUDES) $(DEFINES)  $(LINKS)  -o $@

test_%: test_%.o
	$(CC) -Wl,-O1 $^   $(INCLUDES) $(DEFINES)  $(LINKS)   -o $@

test_window_functions test_dct: %: mfcc.o %.o
	$(CC) -Wl,-O1 $^   $(INCLUDES) $(DEFINES)  $(LINKS)  -o $@

test_simple_rescale test_rescale: %: rescale.o %.o
	$(CC) -Wl,-O1 $^   $(INCLUDES) $(DEFINES)  $(LINKS)  -o $@

test_backprop test_fb_backprop: %: recur-nn.o recur-nn-io.o %.o
	$(CC) -Wl,-O1 $^   $(INCLUDES) $(DEFINES)  $(LINKS)  -o $@

test_mdct: %: recur-nn.o mdct.o window.o %.o
	$(CC) -Wl,-O1 $^   $(INCLUDES) $(DEFINES)  $(LINKS)  -o $@



gtk-recur.o: gtk-recur.c
	$(CC) -c  -MMD $(ALL_CFLAGS) $(CPPFLAGS)  $(INCLUDES)  $(GTK_INCLUDES) -o $@ $<


gtk-recur: gtk-recur.o
	$(CC) -Wl,-O1 $^   $(INCLUDES)  $(GTK_INCLUDES) $(DEFINES)  $(LINKS) $(GTK_LINKS)   -o $@


.PHONY: all test-pipeline clean pgm-clean


VIDEO_DIR = $(CURDIR)/test-video
VID_URI_1=file://$(VIDEO_DIR)/small/2004-08-08.avi
VID_URI_2=file://$(VIDEO_DIR)/F30275.mov
VID_URI_3=file://$(VIDEO_DIR)/rochester-pal.avi
VID_URI_4=file://$(VIDEO_DIR)/DEC.flv
VID_URI_5=file://$(VIDEO_DIR)/alowhum/vts_16_1.vob.avi
VID_URI_6=file://$(VIDEO_DIR)/movies/louis-theroux-lagos/louis.theroux.law.and.disorder.in.lagos.ws.pdtv.xvid-waters.avi
VID_URI_7=file://$(VIDEO_DIR)/movies/InBruges.avi
VID_URI_8=file://$(VIDEO_DIR)/movies/louis-theroux-zionists/Louis.Theroux.Ultra.Zionists.WS.PDTV.XviD-PVR.avi

VID_W=800
VID_H=600
VID_SPECS = video/x-raw, format=I420, width=$(VID_W), height=$(VID_H)

VID_TEST_SRC_1 = videotestsrc pattern=14 kt=2 kxt=1 kyt=3  kxy=3 !\
        $(VID_SPECS), framerate=\(fraction\)25/1

VID_LINE=videoscale method=nearest-neighbour ! videoconvert ! $(VID_SPECS)
AUD_LINE=audioconvert ! audioresample

VID_FILE_SRC_1 = uridecodebin name=src uri=$(VID_URI_1) ! $(VID_LINE)
VID_FILE_SRC_2 = uridecodebin name=src uri=$(VID_URI_2) ! $(VID_LINE)
VID_FILE_SRC_3 = uridecodebin name=src uri=$(VID_URI_3) ! $(VID_LINE)
VID_FILE_SRC_4 = uridecodebin name=src uri=$(VID_URI_4) ! $(VID_LINE)
VID_FILE_SRC_5 = uridecodebin name=src uri=$(VID_URI_5) ! $(VID_LINE)
VID_FILE_SRC_6 = uridecodebin name=src uri=$(VID_URI_6) ! $(VID_LINE)
VID_FILE_SRC_7 = uridecodebin name=src uri=$(VID_URI_7) ! $(VID_LINE)
VID_FILE_SRC_8 = uridecodebin name=src uri=$(VID_URI_8) ! $(VID_LINE)

#GST_DEBUG=uridecodebin:7
#GST_DEBUG=recur*:5
TIMER =
#TIMER = time -f '\nused %P CPU\n' timeout 10
GDB =
#GDB = gdb --args
VALGRIND = valgrind --tool=memcheck --log-file=valgrind.log --trace-children=yes --suppressions=valgrind-python.supp  --leak-check=full --show-reachable=yes


TEST_PIPELINE_CORE = gst-launch-1.0  \
	  --gst-plugin-path=$(CURDIR) \
	$(VID_FILE_SRC_3) ! recur_manager name=recur osdebug=0 ! videoconvert \
	! xvimagesink force-aspect-ratio=false \
	recur. ! autoaudiosink \
	src. ! $(AUD_LINE) ! recur.



test-pipeline: all
	GST_DEBUG=$(GST_DEBUG) $(TIMER) $(GDB) $(TEST_PIPELINE_CORE)  2> gst.log

test-pipeline-valgrind: all
	$(VALGRIND) $(TEST_PIPELINE_CORE)

test-pipeline-kcachegrind: all
	kcachegrind &
	valgrind --tool=callgrind $(TEST_PIPELINE_CORE) 2>callgrind.log

train-pipeline:
	gst-launch-1.0  \
	  --gst-plugin-path=$(CURDIR) \
	$(VID_FILE_SRC_6) ! recur_manager name=recur ! videoconvert ! fakesink \
	recur. ! fakesink \
	src. ! $(AUD_LINE) ! recur.
	gst-launch-1.0  \
	  --gst-plugin-path=$(CURDIR) \
	$(VID_FILE_SRC_6) ! recur_manager name=recur ! videoconvert ! fakesink \
	recur. ! fakesink \
	src. ! $(AUD_LINE) ! recur.


MP3_DIR = /home/douglas/media/audio/mp3

AUD_URI_1 = $VID_URI_3
AUD_URI_2 = file://$(MP3_DIR)/Bach/Johann\ Sebastian\ Bach\ -\ Sonatas\ \&\ Partitas\ for\ Violin\ solo/cd1/02\ Sonata\ I\ in\ G\ minor\,\ BWV\ 1001\ -\ Fuga.\ Allegro.mp3
AUD_URI_3 = file://$(MP3_DIR)/John_White/John\ White--Balloon\ Adventure--08\ End\ of\ the\ Road.mp3
AUD_URI_4 = file://$(MP3_DIR)/Sam\ Hunt\ _\ Mammal/Beware\ The\ Man/Sam\ Hunt\ _\ Mammal--Beware\ The\ Man--08\ Lyn.mp3
AUD_URI_5 = file://$(MP3_DIR)/Kraftwerk/Kraftwerk\ -\ Autobahn.mp3
AUD_URI_6 = file://$(MP3_DIR)/spoken/sat-20110813-0910-john_kendrick_bird_watching-00.ogg
AUD_URI_CALE = file://$(MP3_DIR)/John_Cale/03\ Hedda\ Gabbler.mp3
AUD_URI_REED = file://$(MP3_DIR)/misc/Lou\ Reed-10\ Sad\ Song.mp3
AUD_URI_CALE_CAT = file://$(CURDIR)/test-audio/03\ Hedda\ Gabbler.mp3
AUD_URI_REED_CAT = file://$(CURDIR)/test-audio/Lou\ Reed-10\ Sad\ Song.mp3

TEST_PARROT_CORE = gst-launch-1.0  \
	  --gst-plugin-path=$(CURDIR) \
	uridecodebin name=src uri=$(AUD_URI_6)  ! $(AUD_LINE) \
	 ! parrot name=parrot

PARROT_DEBUG=parrot*:5
#PARROT_DEBUG=5
test-parrot: libgstparrot.so
	GST_DEBUG=$(PARROT_DEBUG) $(TIMER) $(GDB) $(TEST_PARROT_CORE) ! autoaudiosink  2> gst.log

train-parrot: libgstparrot.so
	$(TEST_PARROT_CORE) ! fakesink
	$(TEST_PARROT_CORE) ! fakesink

train-parrot-torben: libgstparrot.so
	gst-launch-1.0 --gst-plugin-path=$(CURDIR) \
		uridecodebin uri=$(AUD_URI_REED)  ! $(AUD_LINE) ! parrot ! fakesink
	gst-launch-1.0 --gst-plugin-path=$(CURDIR) \
		uridecodebin uri=$(AUD_URI_CALE)  ! $(AUD_LINE) ! parrot ! fakesink

test-parrot-torben: libgstparrot.so
	GST_DEBUG=recur:3 gst-launch-1.0 --gst-plugin-path=$(CURDIR) \
		uridecodebin uri=$(AUD_URI_REED)  ! $(AUD_LINE) ! parrot ! autoaudiosink
	GST_DEBUG=recur:3 gst-launch-1.0 --gst-plugin-path=$(CURDIR) \
		uridecodebin uri=$(AUD_URI_CALE)  ! $(AUD_LINE) ! parrot ! autoaudiosink

PARROT_CAPS = "audio/x-raw,channels=1,rate=22050,format=S16LE"

test-parrot-duo: libgstparrot.so
	GST_DEBUG=2 gst-launch-1.0 --gst-plugin-path=$(CURDIR) \
		uridecodebin uri=$(AUD_URI_REED)  ! audioconvert ! audioresample \
		! $(PARROT_CAPS) ! interleave name=il \
		! parrot ! autoaudiosink \
		uridecodebin uri=$(AUD_URI_CALE) ! audioconvert ! audioresample \
		! $(PARROT_CAPS) ! il.

train-parrot-duo: libgstparrot.so
	GST_DEBUG=parrot:5 gst-launch-1.0 --gst-plugin-path=$(CURDIR) \
		uridecodebin uri=$(AUD_URI_REED_CAT)  ! audioconvert ! audioresample \
		! $(PARROT_CAPS) ! interleave name=il \
		! parrot ! fakesink \
		uridecodebin uri=$(AUD_URI_CALE_CAT) ! audioconvert ! audioresample \
		! $(PARROT_CAPS) ! il.

.PHONY: classify-test

classify-test: libgstclassify.so
	mv classify*.net nets || echo no net to move
	rm classify.log || echo no log to nuke
	time ./classify-train > log.log
	python classify.py test

valgrind-classify:
	mv classify*.net nets || echo no net to move
	rm classify.log || echo no log to nuke
	$(VALGRIND) --leak-check=full --show-reachable=yes python classify.py train

include $(wildcard *.d)

flawfinder:
	flawfinder *.[ch]

pscan:
	pscan *.[ch]

rats:
	rats *.[ch]

splint:
	splint $(INCLUDES)  *.[ch]

frama-c:
	frama-c -cpp-extra-args="$(INCLUDES)" *.[c]

cppcheck:
	cppcheck $(INCLUDES) -UGLIB_STATIC_COMPILATION -UGST_DISABLE_GST_DEBUG \
	 -UGST_DISABLE_DEPRECATED -UG_PLATFORM_WIN64 -UDLL_EXPORT -UG_PLATFORM_WIN32 \
	-UG_OS_WIN32  *.[ch]


perf-stat:
	sudo perf stat -e cpu-cycles,instructions,cache-references,cache-misses ./test_backprop
	#sudo perf stat -e cpu-cycles,instructions,cache-references,cache-misses,branch-instructions,branch-misses,bus-cycles,ref-cycles,page-faults,minor-faults,major-faults,cpu-migrations,L1-dcache-loads,L1-dcache-load-misses,L1-dcache-stores,L1-dcache-store-misses,L1-dcache-prefetches,L1-dcache-prefetch-misses,L1-icache-loads,L1-icache-load-misses,L1-icache-prefetches,L1-icache-prefetch-misses,LLC-loads,LLC-load-misses,LLC-stores,LLC-store-misses,LLC-prefetches,LLC-prefetch-misses,dTLB-loads,dTLB-load-misses,dTLB-stores,dTLB-store-misses,dTLB-prefetches,dTLB-prefetch-misses,iTLB-loads,iTLB-load-misses,branch-loads,branch-load-misses ./test_backprop
