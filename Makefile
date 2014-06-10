all::

local.mak:
	@if [ ! -e $@ ] ; then \
	  echo " You need a './local.mak'. For a start either symlink or"; \
	  echo " copy and modify './local.mak.example.x86_64'."; \
	  false; \
	fi

include local.mak

WARNINGS = -Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers

LIB_ARCH_DIR = /usr/lib/$(ARCH)-linux-gnu
INC_ARCH_DIR = /usr/include/$(ARCH)-linux-gnu
INC_DIR = /usr/include

### Alternative compilers
#CC = nccgen -ncgcc -ncld -ncfabs
#CC = /usr/bin/clang
#CC = /usr/bin/clang -Weverything -Wno-variadic-macros -Wno-gnu -Wno-vla
#CC = /usr/bin/clang  -Weverything -Wno-documentation -Wno-system-headers -Wno-sign-conversion -Wno-conversion -Wno-gnu -Wno-variadic-macros -Wno-vla -Wno-disabled-macro-expansion -Wno-cast-align
#CLANG_FLAGS = -fslp-vectorize-aggressive
#CLANG_FLAGS =  -fplugin=/usr/lib/gcc/x86_64-linux-gnu/4.7/plugin/dragonegg.so
#CC = clang -Xclang -analyze -Xclang -analyzer-checker=debug.ViewCallGraph

ifdef USE_CBLAS
BLAS_LINK = -lblas
BLAS_FLAGS = -DUSE_CBLAS
endif


ALL_CFLAGS = -march=native -pthread $(WARNINGS) -pipe  -D_GNU_SOURCE $(INCLUDES) $(ARCH_CFLAGS) $(CFLAGS) $(DEV_CFLAGS) -ffast-math -funsafe-loop-optimizations $(CLANG_FLAGS) -std=gnu11 $(CFLAGS) $(BLAS_CFLAGS) $(LOCAL_FLAGS)
ALL_LDFLAGS = $(LDFLAGS)

GST_INCLUDES =  -isystem $(INC_DIR)/gstreamer-1.0\
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

COMMON_LINKS = -L/usr/local/lib  -lm -pthread -lrt \
		 $(BLAS_LINK) -lcdb

GST_LINKS = -lgstbase-1.0 -lgstreamer-1.0 \
	 -lgobject-2.0 -lglib-2.0 -lgstvideo-1.0  \
	-lgmodule-2.0 -lgthread-2.0  -lgstfft-1.0 -lgstaudio-1.0 \

LINKS = $(COMMON_LINKS) $(GST_LINKS)

GTK_LINKS =  -lgtk-3 -lgdk-3

OPT_OBJECTS = ccan/opt/opt.o ccan/opt/parse.o ccan/opt/helpers.o ccan/opt/usage.o

subdirs = images nets test-video
$(subdirs):
	mkdir -p $@

all:: libgstclassify.so $(subdirs)


ELF_EXECUTABLES = test_backprop convert-saved-net rnnca-player gtk-recur

clean:
	rm -f *.so *.o *.a *.d *.s *.pyc
	rm -f path.h config.h
	rm -f ccan/*/*.[oad]
	rm -f $(ELF_EXECUTABLES)

pgm-clean:
	#find images -maxdepth 1 -name '*.p?m' | xargs rm -f
	rm -r images
	mkdir images

# Ensure we don't end up with empty file if configurator fails!
config.h: ccan/tools/configurator/configurator
	ccan/tools/configurator/configurator $(CC) $(ALL_CFLAGS) > $@.tmp && mv $@.tmp $@

.c.o:
	$(CC)  -c -MMD $(ALL_CFLAGS) $(CPPFLAGS) -o $@ $<

%.s:	%.c
	$(CC)  -S $(ASM_OPTS)  $(ALL_CFLAGS) $(CPPFLAGS) -o $@ $<

%.S:	%.c
	$(CC)  -S $(ASM_OPTS)  $(ALL_CFLAGS) $(CPPFLAGS) -fverbose-asm -o $@ $<

%.i:	%.c
	$(CC)  -E  $(ALL_CFLAGS) $(CPPFLAGS) -o $@ $<

%.c: %.h

%.dot: %.c
	clang -S  $(ALL_CFLAGS) -O0  -emit-llvm $^ -o - | opt-3.4 -analyze -dot-callgraph
	mv callgraph.dot $@

recur-nn.o: recur-nn.c
	$(CC)  -c -MMD $(ALL_CFLAGS) $(CPPFLAGS) $(NN_SPECIAL_FLAGS) -o $@ $<

RNN_OBJECTS =  recur-nn.o recur-nn-io.o recur-nn-init.o

RECUR_OBJECTS = gstrecur_manager.o gstrecur_audio.o gstrecur_video.o \
	recur-context.o context-recurse.o

libgstrecur.so: $(RECUR_OBJECTS) $(RNN_OBJECTS) rescale.o mfcc.o | nets images
	$(CC) -shared -Wl,-O1 $+ $(INCLUDES) $(DEFINES) $(LINKS) -Wl,-soname -Wl,$@ \
	  -o $@

libgstrnnca.so: $(RNN_OBJECTS) gstrnnca.o rescale.o | nets images
	$(CC) -shared -Wl,-O1 $+ $(INCLUDES) $(DEFINES) $(LINKS) -Wl,-soname -Wl,$@ \
	  -o $@

libgstparrot.so: $(RNN_OBJECTS)  mdct.o gstparrot.o mfcc.o | nets images
	$(CC) -shared -Wl,-O1 $+ $(INCLUDES) $(DEFINES) $(LINKS) -Wl,-soname -Wl,$@ \
	  -o $@

libgstclassify.so: $(RNN_OBJECTS) gstclassify.o mfcc.o | nets images
	$(CC) -shared -Wl,-O1 $+ $(INCLUDES) $(DEFINES) $(LINKS) -Wl,-soname -Wl,$@ \
	  -o $@

test/test_mfcc_table: %:  mfcc.o rescale.o %.o
	$(CC) -Wl,-O1 $^ $(INCLUDES) $(DEFINES) $(LINKS)   -o $@

test/test_mfcc_bins: %: mfcc.o %.o
	$(CC) -Wl,-O1 $^ $(INCLUDES) $(DEFINES) $(LINKS)  -o $@

test_backprop.o: config.h

#actually there are more path.h dependers.
test_backprop.o test/test_fb_backprop.o test/test_rescale.o :path.h

test/test_%: test/test_%.o
	$(CC) -Wl,-O1 $^  -I. $(DEFINES)  $(COMMON_LINKS)   -o $@

test/test_window_functions test/test_dct: %: mfcc.o %.o
	$(CC) -Wl,-O1 $^ $(INCLUDES) $(DEFINES) $(LINKS)  -o $@

test/test_simple_rescale test/test_rescale: %: rescale.o %.o
	$(CC) -Wl,-O1 $^   -I. $(DEFINES)  $(COMMON_LINKS)  -o $@

test_backprop test/test_fb_backprop: %: $(RNN_OBJECTS) %.o $(OPT_OBJECTS)  | nets images
	$(CC) -Iccan/opt/ -Wl,-O1 $(filter %.o,$^)   -I. $(DEFINES)  $(COMMON_LINKS)  -o $@

convert-saved-net: %: $(RNN_OBJECTS) %.o
	$(CC) -Iccan/opt/ -Wl,-O1 $(filter %.o,$^)   -I. $(DEFINES)  $(COMMON_LINKS)  -o $@

test/test_mdct: %: recur-nn.o mdct.o  %.o
	$(CC) -Wl,-O1 $^   -I. $(DEFINES)  $(LINKS)  -o $@

test/test_window: %: mfcc.o mdct.o %.o
	$(CC) -Wl,-O1 $^   -I. $(DEFINES)  $(LINKS)  -o $@

path.h:
	@echo generating path.h
	@echo '/* generated by make */' > $@
	@echo "#ifndef HAVE_RECUR_PATH_H"                 >>$@
	@echo "#define HAVE_RECUR_PATH_H"                 >>$@
	@echo "#define BASE_PATH \"$(CURDIR)\""                 >>$@
	@echo "#define TEST_DATA_DIR \"$(CURDIR)/test-images\"" >>$@
	@echo "#define TEST_VIDEO_DIR \"$(CURDIR)/test-video\"" >>$@
	@echo "#define TEST_AUDIO_DIR \"$(CURDIR)/test-audio\"" >>$@
	@echo "#define DEBUG_IMAGE_DIR \"$(CURDIR)/images\""    >>$@
	@echo "#endif"                                    >>$@


gtk-recur.o rnnca-player.o: %.o: %.c
	$(CC) -c  -MMD $(ALL_CFLAGS) $(CPPFLAGS) $(INCLUDES)  $(GTK_INCLUDES) -o $@ $<

rnnca-player gtk-recur: %: %.o
	$(CC) -Wl,-O1 $^ $(INCLUDES) $(GTK_INCLUDES) $(DEFINES) $(LINKS) $(GTK_LINKS)   -o $@

startup/%.desktop: startup/%.desktop.template
	sed "s|{recur-root}|$(CURDIR)|g" < $< > $@
	chmod a+x $@

.PHONY: all test-pipeline clean pgm-clean

#base urls for fetching test video.
IA_URL = https://archive.org/download
NOAA_16_URL = http://docs.lib.noaa.gov/noaa_documents/video/NOAA_16mm

robbers-under-water_URL = $(IA_URL)/0330_Robbers_Under_Water_E01201_08_45_30_00/0330_Robbers_Under_Water_E01201_08_45_30_00.ogv
rochester_URL = $(IA_URL)/Rochester_A_City_of_Quality/0103_Rochester_A_City_of_Quality_M05344_15_00_42_00_3mb.ogv
white-pelican_URL = $(IA_URL)/0812_White_Pelican_00_43_03_00/0812_White_Pelican_00_43_03_00_3mb.ogv
LA-colour_URL = $(IA_URL)/PET0981_R-2_LA_color/PET0981_R-2_LA_color.ogv
filmcollectief-09-208_URL = $(IA_URL)/filmcollectief-09-208/filmcollectief-09-208.ogv
filmcollectief-09-542_URL = $(IA_URL)/filmcollectief-09-542/filmcollectief-09-542.ogv

camille_URL = $(NOAA_16_URL)/QC9452C3L331969.mov
shrimp-tanks_URL = $(NOAA_16_URL)/QL444M33R631972pt1.mov
QC875M671960_URL = $(NOAA_16_URL)/QC875M671960.mov
plane-left-window_URL = $(NOAA_16_URL)/QC9452D3H871958film1.mov
arctic_URL = $(NOAA_16_URL)/QB2812G461949.mov
QC819O51958_URL = $(NOAA_16_URL)/QC819O51958.mov


.PRECIOUS: test-video/%.ogv test-video/%.mov

test-video/%.ogv test-video/%.mov: |test-video
	wget "$($*_URL)" -O $@

VID_W=288
VID_H=192
VID_FPS=20

test-video/%-$(VID_W)x$(VID_H)-$(VID_FPS)fps.avi: test-video/%.ogv
	scripts/reduce-video.sh $< $@ $(VID_FPS) $(VID_W) $(VID_H)

test-video/%-$(VID_W)x$(VID_H)-$(VID_FPS)fps.avi: test-video/%.mov
	scripts/reduce-video.sh $< $@ $(VID_FPS) $(VID_W) $(VID_H) 1

VID_SPECS = video/x-raw, format=I420, width=$(VID_W), height=$(VID_H), framerate=20/1

VID_TEST_SRC_1 = videotestsrc pattern=14 kt=2 kxt=1 kyt=3  kxy=3 !\
        $(VID_SPECS), framerate=\(fraction\)25/1

VID_LINE=videoscale method=nearest-neighbour ! videoconvert ! $(VID_SPECS)
AUD_LINE=audioconvert ! audioresample

#GST_DEBUG=uridecodebin:7
#GST_DEBUG=recur*:5
#TIMER = time -f '\nused %P CPU\n' timeout 10
#GDB = gdb --args
#GDB=valgrind --tool=memcheck  --track-origins=yes
VALGRIND = valgrind --tool=memcheck --log-file=valgrind.log --trace-children=yes --suppressions=valgrind-python.supp  --leak-check=full --show-reachable=yes

DEFAULT_VIDEO = test-video/filmcollectief-09-542-288x192-20fps.avi
#DEFAULT_VIDEO = test-video/camille-288x192-20fps.avi
VID_FILE_SRC_DEFAULT = uridecodebin name=src uri=file://$(CURDIR)/$(DEFAULT_VIDEO) \
	! $(VID_LINE)

#RNNCA_DEBUG=GST_DEBUG=rnnca*:5,recur*:5

test-rnnca: libgstrnnca.so $(subdirs) $(DEFAULT_VIDEO)
	$(RNNCA_DEBUG)	$(GDB) 	gst-launch-1.0  \
	  --gst-plugin-path=$(CURDIR) \
	$(VID_FILE_SRC_DEFAULT) \
	! rnnca log-file=rnnca.log training=1 playing=1 edges=0  \
	! videoconvert ! xvimagesink force-aspect-ratio=0

train-rnnca: libgstrnnca.so $(subdirs) $(DEFAULT_VIDEO)
	$(RNNCA_DEBUG) $(GDB) 	gst-launch-1.0  \
		  --gst-plugin-path=$(CURDIR) \
		$(VID_FILE_SRC_DEFAULT) \
		! rnnca log-file=rnnca.log training=1 playing=0 \
		! fakesink ;\

PROPER_RNNCA_PROPERTIES = momentum-soft-start=3000 momentum=0.95 learn-rate=3e-6 \
	hidden-size=79 log-file=rnnca.log offsets=Y000111C000111

train-rnnca-properly: libgstrnnca.so $(subdirs) $(DEFAULT_VIDEO)
	$(RNNCA_DEBUG) $(GDB) 	gst-launch-1.0  \
		  --gst-plugin-path=$(CURDIR) \
		$(VID_FILE_SRC_DEFAULT) \
		! rnnca $(PROPER_RNNCA_PROPERTIES) \
		training=1 playing=0 \
		! fakesink ;\

test-rnnca-properly: libgstrnnca.so $(subdirs) $(DEFAULT_VIDEO)
	$(RNNCA_DEBUG) $(GDB) 	gst-launch-1.0  \
		  --gst-plugin-path=$(CURDIR) \
		$(VID_FILE_SRC_DEFAULT) \
		! rnnca $(PROPER_RNNCA_PROPERTIES) \
		training=1 playing=1 \
		!  xvimagesink force-aspect-ratio=0

play-rnnca-properly: libgstrnnca.so $(subdirs)
	$(RNNCA_DEBUG)	$(GDB) 	gst-launch-1.0  \
	  --gst-plugin-path=$(CURDIR) \
	videotestsrc pattern=black  ! $(VID_SPECS)  \
	! rnnca $(PROPER_RNNCA_PROPERTIES) \
	training=0 playing=1 edges=0 \
	! videoconvert ! xvimagesink force-aspect-ratio=0

#RNNCA_DEBUG=GST_DEBUG=5

record-rnnca-properly: libgstrnnca.so
	$(RNNCA_DEBUG)	$(GDB) 	gst-launch-1.0 	--gst-plugin-path=$(CURDIR) \
	avimux name=mux ! filesink location=rnnca2.avi \
	videotestsrc pattern=black  ! $(VID_SPECS) \
	! rnnca $(PROPER_RNNCA_PROPERTIES) \
	 training=0 playing=1 edges=0 \
	! videoconvert ! x264enc bitrate=512 ! mux.


play-rnnca: libgstrnnca.so $(subdirs)
	$(RNNCA_DEBUG)	$(GDB) 	gst-launch-1.0  \
	  --gst-plugin-path=$(CURDIR) \
	videotestsrc pattern=black  ! $(VID_SPECS) ! \
	rnnca training=0 playing=1 edges=0 \
	! videoconvert ! xvimagesink force-aspect-ratio=0

record-rnnca: libgstrnnca.so
	$(RNNCA_DEBUG)	$(GDB) 	gst-launch-1.0  \
	  --gst-plugin-path=$(CURDIR) \
	videotestsrc pattern=black  ! $(VID_SPECS), framerate=20/1 ! \
	 rnnca training=0 playing=1 edges=0 \
	! videoconvert ! vp8enc ! webmmux ! filesink location=rnnca.webm


TEST_PIPELINE_CORE = gst-launch-1.0  \
	  --gst-plugin-path=$(CURDIR) \
	$(VID_FILE_SRC_DEFAULT) ! recur_manager name=recur osdebug=0 ! videoconvert \
	! xvimagesink force-aspect-ratio=false \
	recur. ! autoaudiosink \
	src. ! $(AUD_LINE) ! recur.



test-pipeline: libgstrecur.so  $(DEFAULT_VIDEO)
	GST_DEBUG=$(GST_DEBUG) $(TIMER) $(GDB) $(TEST_PIPELINE_CORE)

%-recur.ogv: libgstrecur.so  $(DEFAULT_VIDEO)
	timeout 30 gst-launch-1.0  \
	  --gst-plugin-path=$(CURDIR) \
	$(VID_FILE_SRC_DEFAULT) ! recur_manager name=recur ! videoconvert ! queue ! theoraenc ! oggmux ! filesink location="$@" \
	recur. ! fakesink \
	src. ! $(AUD_LINE) ! recur.

test-pipeline-valgrind: libgstrecur.so
	$(VALGRIND) $(TEST_PIPELINE_CORE)

test-pipeline-kcachegrind: libgstrecur.so
	kcachegrind &
	valgrind --tool=callgrind $(TEST_PIPELINE_CORE) 2>callgrind.log

train-pipeline:  $(DEFAULT_VIDEO)
	gst-launch-1.0  \
	  --gst-plugin-path=$(CURDIR) \
	$(VID_FILE_SRC_DEFAULT) ! recur_manager name=recur ! videoconvert ! fakesink \
	recur. ! fakesink \
	src. ! $(AUD_LINE) ! recur.
	gst-launch-1.0  \
	  --gst-plugin-path=$(CURDIR) \
	$(VID_FILE_SRC_DEFAULT) ! recur_manager name=recur ! videoconvert ! fakesink \
	recur. ! fakesink \
	src. ! $(AUD_LINE) ! recur.


AUD_URI_CALE = file://$(MP3_DIR)/John_Cale/03\ Hedda\ Gabbler.mp3
AUD_URI_REED = file://$(MP3_DIR)/misc/Lou\ Reed-10\ Sad\ Song.mp3
AUD_URI_CALE_CAT = file://$(CURDIR)/test-audio/03\ Hedda\ Gabbler.mp3
AUD_URI_REED_CAT = file://$(CURDIR)/test-audio/Lou\ Reed-10\ Sad\ Song.mp3

PARROT_CAPS = "audio/x-raw,channels=1,format=S16LE"

TEST_PARROT_CORE = gst-launch-1.0  \
	  --gst-plugin-path=$(CURDIR) \
	uridecodebin name=src uri=$(AUD_URI_REED) ! audioconvert ! audioresample \
	! parrot name=parrot

PARROT_DEBUG=parrot*:5
#PARROT_DEBUG=5
PARROT_SIZE=399
#PARROT_GDB=gdb --args
#PARROT_GDB=valgrind --tool=memcheck  --track-origins=yes
test-parrot: libgstparrot.so
	GST_DEBUG=$(PARROT_DEBUG)  $(TIMER) $(PARROT_GDB) $(TEST_PARROT_CORE) \
	training=1 playing=1 log-file=parrot.log hidden-size=$(PARROT_SIZE) ! autoaudiosink  #2> gst.log

play-parrot: libgstparrot.so
	GST_DEBUG=$(PARROT_DEBUG)  $(TIMER) $(PARROT_GDB) $(TEST_PARROT_CORE) \
	training=0 playing=1 hidden-size=$(PARROT_SIZE) ! autoaudiosink  #2> gst.log

train-parrot: libgstparrot.so
	GST_DEBUG=$(PARROT_DEBUG)  $(TIMER) $(PARROT_GDB) $(TEST_PARROT_CORE) \
	training=1 playing=0 log-file=parrot.log hidden-size=$(PARROT_SIZE) ! fakesink

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

PARROT_CAPS = "audio/x-raw,channels=1,rate=16000,format=S16LE"

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

.PHONY: classify

classify: libgstclassify.so
	mv classify*.net nets || echo no net to move
	rm classify.log || echo no log to nuke
	time ./classify-train -q
	time ./classify-test -q

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
