/* Copyright 2015 Douglas Bagnall <douglas@halo.gen.nz> LGPL

Python bindings for text RNNs.
*/
#include <Python.h>
#include "charmodel.h"
#include "utf8.h"
#include "py-recur-text.h"
#include <math.h>
#include <string.h>
#include <fenv.h>

#define DEFAULT_ADAGRAD_BALLAST 100
#define DEFAULT_ADADELTA_BALLAST 100

typedef struct {
    PyObject_HEAD
    RnnCharAlphabet *alphabet;
    int *char_to_net;
} Alphabet;


static void
Alphabet_dealloc(Alphabet* self)
{
    if (self->alphabet){
        rnn_char_free_alphabet(self->alphabet);
    }
    if (self->char_to_net){
        free(self->char_to_net);
    }
    self->ob_type->tp_free((PyObject*)self);
}

static PyObject *
Alphabet_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    Alphabet *self = (Alphabet *)PyType_GenericNew(type, args, kwds);
    self->alphabet = NULL;
    return (PyObject *)self;
}

static int
Alphabet_init(Alphabet *self, PyObject *args, PyObject *kwds)
{
    char *text = NULL;
    Py_ssize_t text_len = 0;
    float threshold = 1e-5;
    float digit_adjust = 0.3;
    float alpha_adjust = 3;
    int ignore_case = 1;
    int utf8 = 1;
    int collapse_space = 1;
    const char *alphabet_chars = NULL;
    const char *collapse_chars = NULL;

    static char *kwlist[] = {"text", "threshold", "digit_adjust", "alpha_adjust",
                             "ignore_case", "utf8", "collapse_space",
                             "alphabet_chars", "collapse_chars", NULL};


    if (!PyArg_ParseTupleAndKeywords(args, kwds, "z#|fffiiizz",
            kwlist, &text, &text_len,
            &threshold, &digit_adjust, &alpha_adjust,
            &ignore_case, &utf8, &collapse_space,
            &alphabet_chars, &collapse_chars)){
	PyErr_PrintEx(1);
        return -1;
    }

    self->alphabet = rnn_char_new_alphabet();
    rnn_char_alphabet_set_flags(self->alphabet, ignore_case, utf8,
				collapse_space);


    if (text == NULL){
	if (alphabet_chars == NULL){
            PyErr_Format(PyExc_ValueError, "Neither text nor alphabet_chars"
			 " is set");
            return -1;
	}
	if (collapse_chars == NULL){
	    collapse_chars = "";
	}

	RnnCharAlphabet *a = self->alphabet;
	a->len = fill_codepoints_from_string(a->points,
					     256, alphabet_chars,
					     utf8);
	a->collapsed_len = fill_codepoints_from_string(a->collapsed_points,
						       256, collapse_chars,
						       utf8);
    }
    else {
	int r = rnn_char_find_alphabet_s(text, text_len, self->alphabet,
					 threshold, digit_adjust, alpha_adjust);
        if (r) {
            rnn_char_free_alphabet(self->alphabet);
            self->alphabet = NULL;
            PyErr_Format(PyExc_ValueError, "can't find an alphabet!");
            return -1;
        }
    }
    self->char_to_net = rnn_char_new_char_lut(self->alphabet);

    return 0;
}

static PyObject *
Alphabet_get_codepoint(Alphabet *self, PyObject *letter)
{
    const char *s = PyString_AsString(letter);
    int c = rnn_char_get_codepoint(self->alphabet, s);
    return PyInt_FromLong(c);
}

static PyObject *
Alphabet_encode_text(Alphabet *self, PyObject *orig_obj)
{
    const char *orig_str = PyString_AsString(orig_obj);
    int orig_len = PyString_Size(orig_obj);
    int encoded_len;

    if (orig_str == NULL || orig_len < 0){
        return PyErr_Format(PyExc_ValueError, "encode_text requires a string");
    }

    u8 *encoded_text = rnn_char_alloc_encoded_text(alphabet,
        orig_str, orig_len, &encoded_len, NULL, false);

    PyObject *final_obj = PyString_FromStringAndSize((char *)encoded_text,
        encoded_len);
    return final_obj;
}

static PyObject *
Alphabet_decode_text(Alphabet *self, PyObject *orig_obj)
{
    const char *orig_str = PyString_AsString(orig_obj);
    int orig_len = PyString_Size(orig_obj);
    int new_len;
    char *s = rnn_char_uncollapse_text(self->alphabet, (u8*)orig_str, orig_len,
                                       &new_len);

    PyObject *final_obj = PyString_FromStringAndSize(s, new_len);
    free(s);

    return final_obj;
}


static PyObject *
Alphabet_getalphabet(Alphabet *self, void *closure)
{
    RnnCharAlphabet *alphabet = self->alphabet;
    char *s;
    char which = *(char *)closure;
    int *points;
    int len;
    int utf8 = alphabet->flags & RNN_CHAR_FLAG_UTF8;
    if (which == 'a'){
        len = alphabet->len;
        points = alphabet->points;
    }
    else {
        len = alphabet->collapsed_len;
        points = alphabet->collapsed_points;
    }

    s = new_string_from_codepoints(points, len, utf8);
    return PyString_FromString(s);
}


static const u32 flag_ignore_case = RNN_CHAR_FLAG_CASE_INSENSITIVE;
static const u32 flag_utf8 = RNN_CHAR_FLAG_UTF8;
static const u32 flag_collapse_space = RNN_CHAR_FLAG_COLLAPSE_SPACE;

static PyObject *
Alphabet_getflag(Alphabet *self, void *closure)
{
    long f = self->alphabet->flags & *(u32*)closure;
    return PyBool_FromLong(f);
}

static int
Alphabet_setflag(Alphabet *self, PyObject *value, void *closure)
{
    int set = PyObject_IsTrue(value);
    u32 flag = *(u32*)closure;
    if (set){
        self->alphabet->flags |= flag;
    }
    else {
        self->alphabet->flags &= ~flag;
    }
     return 0;
}

static PyGetSetDef Alphabet_getsetters[] = {
    {"alphabet",
     (getter)Alphabet_getalphabet, NULL,
     NULL,
     "a"},
    {"collapsed_chars",
     (getter)Alphabet_getalphabet, NULL,
     NULL,
     "c"},
    {"ignore_case",
     (getter)Alphabet_getflag,
     (setter)Alphabet_setflag,
     "treat lowercase and capitals as the same letter",
     (u32*) &flag_ignore_case},
    {"utf8",
     (getter)Alphabet_getflag,
     (setter)Alphabet_setflag,
     "parse text as UTF-8, not bytes",
     (u32*) &flag_utf8},
    {"collapse_space",
     (getter)Alphabet_getflag,
     (setter)Alphabet_setflag,
     "most runs of whitespace become single spaces",
     (u32*) &flag_collapse_space},
    {NULL}  /* Sentinel */
};


static PyMemberDef Alphabet_members[] = {
    {NULL}
};

static PyMethodDef Alphabet_methods[] = {
    {"get_codepoint", (PyCFunction)Alphabet_get_codepoint, METH_O,
     "get the codepoint relating to the character"},
    {"encode_text", (PyCFunction)Alphabet_encode_text, METH_O,
     "encode the given string with the alphabet"},
    {"decode_text", (PyCFunction)Alphabet_decode_text, METH_O,
     "decode the given string with the alphabet"},

    {NULL}
};

static const char alphabet_doc[] =                                      \
    "Representation of the characters recognised by a Net.\n\n"         \
    "The Alphabet can be initialised with with a text string from "     \
    "which a reasonable set of characters will be deduced, or with "    \
    "explicit string of acceptable characters.\n\n"                     \
    "When the alphabet is being deduced from a text, the "              \
    "``threshold``,``digit_adjust``, and ``alpha_adjust`` parameters "  \
    "determine which characters are used. Characters less frequent "    \
    "than ``threshold`` (by default 1e-5) are collapsed together into " \
    "a single representative character. ``alpha_adjust`` and "          \
    "``digit_adjust`` alter the threshold for ascii letters and "       \
    "digits respectively. The threshold is divided by this "            \
    "number for members of the class. By default ``alpha_adjust`` is "  \
    "3 and ``digit_adjust`` is 0.3; thus with the default "             \
    "``threshold`` the effective threshold for letters is around "      \
    "3.3e-6, and that digits is 3.3e-5. Punctuation and other "         \
    "characters use 1e-5.\n\n"                                          \
    "There are 3 boolean flag parameters. The ``ignore_case`` causes "  \
    "uppercase and lowercase versions of a letter to be treated as "    \
    "one. The ``utf8`` flag causes the string the be treated as "       \
    "utf-8. If ``utf8`` is False the string is parsed as bytes. "       \
    "The ``collapse_space`` flag reduces all runs of whitespace to "    \
    "a single space.\n\n"                                               \
    "If ``text`` is ``None``, ``alphabet_chars`` must be a string "     \
    "of characters to use as the alphabet. ``collapse_chars can be "    \
    "used to specify characters that get mapped to the first "          \
    "character of ``alphabet_chars``. ``ignore_case``, ``utf8``, and "  \
    "``collapse_space`` still have meaning.";

static PyTypeObject AlphabetType = {
    PyObject_HEAD_INIT(NULL)
    0,                            /*ob_size*/
    "charmodel.Alphabet",         /*tp_name*/
    sizeof(Alphabet),             /*tp_basicsize*/
    0,                            /*tp_itemsize*/
    (destructor)Alphabet_dealloc, /*tp_dealloc*/
    0,                            /*tp_print*/
    0,                            /*tp_getattr*/
    0,                            /*tp_setattr*/
    0,                            /*tp_compare*/
    0,                            /*tp_repr*/
    0,                            /*tp_as_number*/
    0,                            /*tp_as_sequence*/
    0,                            /*tp_as_mapping*/
    0,                            /*tp_hash */
    0,                            /*tp_call*/
    0,                            /*tp_str*/
    0,                            /*tp_getattro*/
    0,                            /*tp_setattro*/
    0,                            /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /*tp_flags*/
    alphabet_doc,                 /* tp_doc */
    0,                            /* tp_traverse */
    0,                            /* tp_clear */
    0,                            /* tp_richcompare */
    0,                            /* tp_weaklistoffset */
    0,                            /* tp_iter */
    0,                            /* tp_iternext */
    Alphabet_methods,             /* tp_methods */
    Alphabet_members,             /* tp_members */
    Alphabet_getsetters,          /* tp_getset */
    0,                            /* tp_base */
    0,                            /* tp_dict */
    0,                            /* tp_descr_get */
    0,                            /* tp_descr_set */
    0,                            /* tp_dictoffset */
    (initproc)Alphabet_init,      /* tp_init */
    0,                            /* tp_alloc */
    Alphabet_new,                 /* tp_new */
};



/*net object */

typedef struct {
    PyObject_HEAD
    RecurNN *net;
    Alphabet *alphabet;
    PyObject *class_names;
    PyObject *class_name_lut;
    rnn_learning_method learning_method;
    RnnCharProgressReport *report;
    float momentum;
    int n_classes;
    int batch_size;
    RnnCharImageSettings images;
    int periodic_pgm_period;
    const char *filename;
} Net;

static void
Net_dealloc(Net* self)
{
    if (self->net){
        /* save first? */
        rnn_delete_net(self->net);
        self->net = NULL;
    }
    if (self->images.input_ppm){
        free(self->images.input_ppm);
        self->images.input_ppm = NULL;
    }
    if (self->images.error_ppm){
        free(self->images.error_ppm);
        self->images.error_ppm = NULL;
    }
    Py_CLEAR(self->alphabet);
    Py_CLEAR(self->class_names);
    Py_CLEAR(self->class_name_lut);
    self->ob_type->tp_free((PyObject*)self);
}

static PyObject *
Net_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    Net *self = (Net *)PyType_GenericNew(type, args, kwds);
    /*I believe PyType_GenericNew zeros the memory, so all pointers are NULL, etc */
    self->batch_size = 1;
    return (PyObject *)self;
}

static int
set_net_classnames(Net *self, PyObject *class_names)
{
    PyObject *class_name_lut;

    int n_classes = PySequence_Length(class_names);
    self->n_classes = n_classes;

    if (! PySequence_Check(class_names) || n_classes < 1){
	PyErr_Format(PyExc_ValueError,
		     "class_names should be a sequence of strings");
	return -1;
    }
    if (self->class_names){
        PyErr_Format(PyExc_AttributeError,
		     "net->class_names is already set!");
	return -1;
    }

    self->class_names = class_names;
    Py_INCREF(self->class_names);

    class_name_lut = PyDict_New();
    for (long i = 0; i < n_classes; i++){
        PyObject *k = PySequence_Fast_GET_ITEM(class_names, i);
        PyObject *v = PyInt_FromLong(i);
        PyDict_SetItem(class_name_lut, k, v);
        /* PyInt_FromLong does an incref. So does PyDict_SetItem.
           That's one two many. */
        Py_DECREF(v);
    }
    self->class_name_lut = class_name_lut;

    return n_classes;
}

static int
set_net_filename(Net *self, const char *filename, const char *basename,
		 char *metadata)
{
    char s[1000];
    RecurNN *net = self->net;
    if (filename){
        self->filename = strdup(filename);
    }
    else {
        u32 sig = rnn_hash32(metadata);
        int wrote = snprintf(s, sizeof(s), "%s-%0" PRIx32 "i%d-h%d-o%d.net",
            basename, sig, net->input_size, net->hidden_size, net->output_size);
        if (wrote >= sizeof(s)){
	    PyErr_Format(PyExc_ValueError,
			 "filename is trying to be too long!");
            return -1;
        }
        self->filename = strdup(s);
    }
    return 0;
}


static int
set_net_pgm_dump(Net *self, const char *basename, int temporal_pgm_dump,
		 char *periodic_pgm_dump, int periodic_pgm_period)
{
    char s[1000];
    RecurNN *net = self->net;
    self->images.temporal_pgm_dump = temporal_pgm_dump;
    if (temporal_pgm_dump){
        snprintf(s, sizeof(s), "%s-input_layer", basename);
        self->images.input_ppm = temporal_ppm_alloc(net->i_size, 300,
            s, 0, PGM_DUMP_COLOUR, NULL);
        snprintf(s, sizeof(s), "%s-output_error", basename);
        self->images.error_ppm = temporal_ppm_alloc(net->o_size, 300,
            s, 0, PGM_DUMP_COLOUR, NULL);
    }

    if (periodic_pgm_dump){
        self->images.periodic_pgm_dump_string = periodic_pgm_dump;
        self->periodic_pgm_period = periodic_pgm_period;
    }
    return 0;
}



static int
Net_init(Net *self, PyObject *args, PyObject *kwds)
{
    /* mandatory arguments */
    Alphabet *alphabet;
    PyObject *class_names;
    uint hidden_size;

    /* optional arguments */
    unsigned long long rng_seed = 2;
    const char *log_file = "multi-text.log";
    const char *filename = NULL;
    int bptt_depth = 50;
    float learn_rate = 0.001;
    float momentum = 0.95;
    float presynaptic_noise = 0.1;
    rnn_activation activation = RNN_RESQRT;
    int learning_method = RNN_ADAGRAD;
    int verbose = 0;
    int batch_size = 1;
    int init_method = RNN_INIT_FLAT;
    int temporal_pgm_dump = 0;
    char *periodic_pgm_dump = NULL;
    int periodic_pgm_period = 1000;
    const char *basename = NULL;
    char *metadata = NULL;

    /* other vars */
    int n_classes;
    uint output_size;
    RecurNN *net;
    u32 flags = RNN_NET_FLAG_STANDARD | RNN_NET_FLAG_BPTT_ADAPTIVE_MIN_ERROR;

    static char *kwlist[] = {"alphabet",             /* O! */
                             "classes",              /* O  */
                             "hidden_size",          /* i  |  */
                             "log_file",             /* z  */
                             "bptt_depth",           /* i  */
                             "learn_rate",           /* f  */
                             "filename",             /* z  */
                             "momentum",             /* f  */
                             "presynaptic_noise",    /* f  */
                             "rng_seed",             /* K  */
                             "metadata",             /* z  */
                             "activation",           /* i  */
                             "learning_method",      /* i  */
                             "basename",             /* z  */
                             "verbose",              /* i  */
                             "temporal_pgm_dump",    /* i  */
                             "periodic_pgm_dump",    /* z  */
                             "periodic_pgm_period",  /* i  */
                             "batch_size",           /* i  */
                             "init_method",          /* i  */
                             NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!Oi|zifzffKziiziiziii",
            kwlist, &AlphabetType,
            &alphabet,            /* O! */
            &class_names,         /* O  */
            &hidden_size,         /* i  |  */
            &log_file,            /* z  */
            &bptt_depth,          /* i  */
            &learn_rate,          /* f  */
            &filename,            /* z  */
            &momentum,            /* f  */
            &presynaptic_noise,   /* f  */
            &rng_seed,            /* K  */
            &metadata,            /* z  */
            &activation,          /* i  */
            &learning_method,     /* i  */
            &basename,            /* z  */
            &verbose,             /* i  */
            &temporal_pgm_dump,   /* i  */
            &periodic_pgm_dump,   /* z  */
            &periodic_pgm_period, /* i  */
            &batch_size,          /* i  */
            &init_method          /* i  */
        )){
        return -1;
    }
    if (activation >= RNN_ACTIVATION_LAST || activation < 1){
        PyErr_Format(PyExc_ValueError, "%d is not a valid activation", activation);
        return -1;
    }

    if (batch_size < 1){
        PyErr_Format(PyExc_ValueError, "batch_size %d won't work", batch_size);
        return -1;
    }

    if ((unsigned)learning_method >= RNN_LAST_LEARNING_METHOD){
        PyErr_Format(PyExc_ValueError, "%d is not a valid learning method",
            learning_method);
        return -1;
    }

    n_classes = set_net_classnames(self, class_names);
    if (n_classes < 1){
	return -1;
    }

    if (learning_method == RNN_ADADELTA || learning_method == RNN_RPROP){
        flags |= RNN_NET_FLAG_AUX_ARRAYS;
    }

    output_size = alphabet->alphabet->len * n_classes;

    net = rnn_new(
        alphabet->alphabet->len,
        hidden_size,
        output_size,
        flags,
        rng_seed,
        log_file,
        bptt_depth,
        learn_rate,
        momentum,
        presynaptic_noise,
        activation);

    self->net = net;
    self->momentum = momentum;
    self->report = verbose ? calloc(sizeof(*self->report), 1) : NULL;
    self->batch_size = batch_size;

    if (init_method < RNN_INIT_ZERO || init_method >= RNN_INIT_LAST){
        init_method = RNN_INIT_FLAT;
    }
    rnn_randomise_weights_simple(net, init_method);

    net->metadata = metadata;
    if (basename == NULL){
        basename = "multi-text";
    }

    set_net_filename(self, filename, basename, metadata);

    set_net_pgm_dump(self, basename, temporal_pgm_dump,
		     periodic_pgm_dump, periodic_pgm_period);

    self->learning_method = learning_method;

    Py_INCREF(alphabet);
    self->alphabet = alphabet;


    switch(learning_method){
    case RNN_ADAGRAD:
        rnn_set_momentum_values(self->net, DEFAULT_ADAGRAD_BALLAST);
        break;
    case RNN_ADADELTA:
        rnn_set_momentum_values(self->net, DEFAULT_ADADELTA_BALLAST);
        break;
    case RNN_RPROP:
        rnn_set_aux_values(self->net, 1);
        break;
    }
    return 0;
}

static PyObject *
Net_getfloat_rnn(Net *self, int *closure)
{
    void *addr = ((void *)self->net) + *closure;
    float f = *(float *)addr;
    return PyFloat_FromDouble((double)f);
}

static int
Net_setfloat_rnn(Net *self, PyObject *value, int *closure)
{
    PyObject *pyfloat = PyNumber_Float(value);
    if (pyfloat == NULL){
        return -1;
    }
    void *addr = ((void *)self->net) + *(int*)closure;
    float f = PyFloat_AS_DOUBLE(pyfloat);
    *(float *)addr = f;
    return 0;
}

static PyObject *
Net_getfloat_bptt(Net *self, void *closure)
{
    void *addr = ((void *)self->net->bptt) + *(int*)closure;
    float f = *(float *)addr;
    return PyFloat_FromDouble((double)f);
}

static int
Net_setfloat_bptt(Net *self, PyObject *value, void *closure)
{
    PyObject *pyfloat = PyNumber_Float(value);
    if (pyfloat == NULL){
        return -1;
    }
    void *addr = ((void *)self->net->bptt) + *(int*)closure;
    float f = PyFloat_AS_DOUBLE(pyfloat);
    *(float *)addr = f;
    return 0;
}

static const int rnn_offset_presynaptic_noise = offsetof(RecurNN, presynaptic_noise);

static const int bptt_offset_learn_rate = offsetof(RecurNNBPTT, learn_rate);
static const int bptt_offset_ih_scale = offsetof(RecurNNBPTT, ih_scale);
static const int bptt_offset_ho_scale = offsetof(RecurNNBPTT, ho_scale);
static const int bptt_offset_momentum_weight = offsetof(RecurNNBPTT, momentum_weight);

static PyGetSetDef Net_getsetters[] = {
    {"presynaptic_noise",
     (getter)Net_getfloat_rnn,
     (setter)Net_setfloat_rnn,
     "Scale of presynaptic noise",
     (void *)&rnn_offset_presynaptic_noise
    },
    {"learn_rate",
     (getter)Net_getfloat_bptt,
     (setter)Net_setfloat_bptt,
     "learning rate",
     (void *)&bptt_offset_learn_rate
    },
    {"ih_scale",
     (getter)Net_getfloat_bptt,
     (setter)Net_setfloat_bptt,
     "scale input to hidden learn-rate",
     (void *)&bptt_offset_ih_scale
    },
    {"ho_scale",
     (getter)Net_getfloat_bptt,
     (setter)Net_setfloat_bptt,
     "scale hidden to output learn-rate",
     (void *)&bptt_offset_ho_scale
    },
    {"momentum_weight",
     (getter)Net_getfloat_bptt,
     (setter)Net_setfloat_bptt,
     "give this much weight to momentum",
     (void *)&bptt_offset_momentum_weight
    },

    {NULL}  /* Sentinel */
};


static PyMemberDef Net_members[] = {
    {"alphabet", T_OBJECT_EX, offsetof(Net, alphabet), READONLY,
     "the net's Alphabet object"},
    {"class_names", T_OBJECT_EX, offsetof(Net, class_names), READONLY,
     "names of classes"},
    {"class_name_lut", T_OBJECT_EX, offsetof(Net, class_name_lut), READONLY,
     "mapping classes to indices"},
    {"n_classes", T_INT, offsetof(Net, n_classes), READONLY,
     "the same as len(Net.class_names)"},
    {"batch_size", T_INT, offsetof(Net, batch_size), 0,
     "generations per mini-batch"},
    {"learning_method", T_INT, offsetof(Net, learning_method), READONLY,
     "learing method: 0: weighted, 1: Nesterov, 2: simplified N.,"
     "3: classical, 4: adagrad, 5: adadelta, 6: rprop;"},
    {"momentum", T_FLOAT, offsetof(Net, momentum), 0,
     "momentum rate (if applicable)"},
    {"filename", T_STRING, offsetof(Net, filename), 0,
     "net will be saved/loaded here"},
    {NULL}
};

static PyObject *
Net_dump_parameters(Net *self, PyObject *args)
{
    RecurNN *net = self->net;
    RecurNNBPTT *bptt = net->bptt;
    printf("Net object\n");
    printf("learning method %d\n", self->learning_method);
    printf("n_classes %d (class_names length %d)\n",
        self->n_classes, (int)PySequence_Length(self->class_names));
    printf("momentum %.2f\n", self->momentum);
    printf("batch size %d\n", self->batch_size);
    printf("i_size %5d  input_size %5d\n", net->i_size, net->input_size);
    printf("h_size %5d hidden_size %5d\n", net->h_size, net->hidden_size);
    printf("o_size %5d output_size %5d\n", net->o_size, net->output_size);
    printf("flags %x\n", net->flags);
    printf("generation %d\n", net->generation);
    printf("presynaptic noise %f\n", net->presynaptic_noise);
    printf("activation %x\n", net->activation);
    printf("bptt depth %d\n", bptt->depth);
    printf("bptt index %d\n", bptt->index);
    printf("learn_rate %f\n", bptt->learn_rate);
    printf("ih_scale %f\n", bptt->ih_scale);
    printf("ho_scale %f\n", bptt->ho_scale);
    printf("bptt momentum %g\n", bptt->momentum);
    printf("bptt momentum_weight %g\n", bptt->momentum_weight);
    printf("bptt min_error_factor %g\n", bptt->min_error_factor);
    return Py_BuildValue("");
}



static PyObject *
Net_train(Net *self, PyObject *args, PyObject *kwds)
{
    const char *text;
    Py_ssize_t text_len = 0;
    PyObject *target_class;
    int target_index;
    float leakage = -1;
    uint ignore_start = 0;

    static char *kwlist[] = {"text",                 /* s# */
                             "target_class",         /* O | */
                             "leakage",              /* f  */
                             "ignore_start",         /* I  */
                             NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "s#O|fI", kwlist,
            &text,
            &text_len,
            &target_class,
            &leakage,
            &ignore_start
        )){
        return NULL;
    }
    if (text_len < 2 + ignore_start){
        return PyErr_Format(PyExc_ValueError, "The text is not long enough");
    }
    PyObject *target_py_int = PyDict_GetItem(self->class_name_lut, target_class);
    if (target_py_int == NULL){
        PyObject *r = PyObject_Repr(target_class);
        PyObject *e = PyErr_Format(PyExc_KeyError, "unknown class: %s",
                                   PyString_AS_STRING(r));
        Py_DECREF(r);
        return e;
    }
    target_index = PyInt_AsLong(target_py_int);

    if (leakage < 0){
        leakage = -leakage / self->n_classes;
    }
    if (ignore_start){
        rnn_char_multitext_spin(self->net, (u8*)text, ignore_start,
            self->images.input_ppm, self->images.error_ppm,
            self->images.periodic_pgm_dump_string, self->periodic_pgm_period);
        text += ignore_start;
        text_len -= ignore_start;
    }

    rnn_char_multitext_train(self->net, (u8*)text, text_len,
        self->alphabet->alphabet->len, target_index, leakage,
        self->report, self->learning_method, self->momentum,
        self->batch_size, self->images.input_ppm, self->images.error_ppm,
        self->images.periodic_pgm_dump_string, self->periodic_pgm_period);
    if (self->report){
        RnnCharProgressReport *r = self->report;
        char *s = PyString_AsString(target_class);
        printf("%8d t%.1f %d/s %s\n", self->net->generation,
            r->training_entropy, (int)r->per_second, s);
    }
    return Py_BuildValue("");
}


static PyObject *
Net_test(Net *self, PyObject *args, PyObject *kwds)
{
    const u8 *text;
    Py_ssize_t text_len = 0;
    uint ignore_start = 0;
    int as_list = 0;
    static char *kwlist[] = {"text",                 /* s# | */
                             "ignore_start",         /* I  */
                             "as_list",              /* i  */
                             NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "s#|Ii", kwlist,
            &text,
            &text_len,
            &ignore_start,
            &as_list
        )){
        return NULL;
    }

    if (text_len < 2 + ignore_start){
        return PyErr_Format(PyExc_ValueError,
            "The text is not long enough (%ld bytes)", text_len);
    }

    double *entropy = calloc(self->n_classes, sizeof(double));

    rnn_char_multi_cross_entropy(self->net, text, text_len,
        self->alphabet->alphabet->len, entropy, ignore_start);

    PyObject *py_entropy;
    if (as_list){
        py_entropy = PyList_New(self->n_classes);
        for (int i = 0; i < self->n_classes; i++){
            PyList_SET_ITEM(py_entropy, i, PyFloat_FromDouble(entropy[i]));
        }
    }
    else { /* as dict */
        py_entropy = PyDict_New();
        for (int i = 0; i < self->n_classes; i++){
            PyObject *k = PyList_GET_ITEM(self->class_names, i);
            PyDict_SetItem(py_entropy, k, PyFloat_FromDouble(entropy[i]));
        }
    }

    free(entropy);
    return py_entropy;
}


static PyObject *
Net_save(Net *self, PyObject *args, PyObject *kwds)
{
    RecurNN *net = self->net;
    const char *filename = NULL;
    int backup = 1;

    static char *kwlist[] = {"filename",             /* z */
                             "backup",               /* i */
                             NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|zi", kwlist,
            &filename,
            &backup
        )){
        return NULL;
    }
    if (filename == NULL){
        filename = self->filename;
    }
    int r = rnn_save_net(net, filename, backup);
    if (r){
        return PyErr_Format(PyExc_IOError, "could not save to %s",
            filename);
    }
    return Py_BuildValue("");
}


static PyObject*
Net_load(PyTypeObject *class, PyObject *args, PyObject *kwds)
{
    const char *filename;
    PyObject *parse_metadata;
    RecurNN *net;
    Net *self;

    static char *kwlist[] = {"filename",             /* s */
                             "parse_metadata",       /* O */
                             NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "sO", kwlist,
            &filename,            /* s  */
            &parse_metadata       /* O  */
        )){
        return NULL;
    }

    net = rnn_load_net(filename);
    if (net == NULL){
        return PyErr_Format(PyExc_IOError,
			    "I could not load file '%s'", filename);
    }

    /* parse the metadata using a passed-in python function into a python
       dictionary. The keys and the types of the values are proscribed. */
    PyObject *metadata_pystring = PyString_FromString(net->metadata);
    PyObject *metadata = PyObject_CallFunctionObjArgs(parse_metadata,
        metadata_pystring, NULL);
    if (metadata == NULL){
        /* an exception in parse_metadata() */
        Py_DECREF(metadata_pystring);
        return NULL;
    }

#define METADATA(x) PyDict_GetItemString(metadata, (x))

    PyObject *version_pyint = METADATA("version");
    int version = PyInt_AsLong(version_pyint);
    if (version != 1){
        Py_DECREF(metadata_pystring);
        Py_DECREF(metadata);
        return PyErr_Format(PyExc_ValueError,
            "I don't know metadata format version %d", version);
    }

    PyObject *alphabet_chars = METADATA("alphabet");
    PyObject *collapse_chars = METADATA("collapse_chars");
    PyObject *case_insensitive = METADATA("case_insensitive");
    PyObject *utf8 = METADATA("utf8");
    PyObject *collapse_space = METADATA("collapse_space");

    PyObject *classnames = METADATA("classnames");
    PyObject *batch_size = METADATA("batch_size");
    PyObject *verbose = METADATA("verbose");
    PyObject *momentum = METADATA("momentum");
    PyObject *learning_method = METADATA("learning_method");

    PyObject *temporal_pgm_dump = METADATA("temporal_pgm_dump");
    PyObject *periodic_pgm_dump = METADATA("periodic_pgm_dump");
    PyObject *periodic_pgm_period = METADATA("periodic_pgm_period");
    PyObject *basename = METADATA("basename");

#undef METADATA

    PyObject *alphabet = PyObject_CallFunction((PyObject *)&AlphabetType,
					       "zfffOOOOO", NULL,
					       0.0, 0.0, 0.0,
					       case_insensitive,
					       utf8,
					       collapse_space,
					       alphabet_chars,
					       collapse_chars);


    Py_INCREF(alphabet);
    self = (Net *)class->tp_new(class, args, kwds);
    self->net = net;
    self->alphabet = (Alphabet *)alphabet;

    Py_INCREF(classnames);
    self->n_classes = set_net_classnames(self, classnames);
    self->momentum = PyFloat_AsDouble(momentum);
    self->learning_method = PyInt_AsLong(learning_method);
    self->batch_size = PyInt_AsLong(batch_size);
    int verbose_flag = PyInt_AsLong(verbose);
    self->report = verbose_flag ? calloc(sizeof(*self->report), 1) : NULL;
    self->filename = strdup(filename);

    const char *basename_char;
    if (PyString_Check(basename)){
	basename_char = PyString_AsString(basename);
    }
    else {
	basename_char = "multi-text";
    }

    set_net_pgm_dump(self, basename_char,
		     PyInt_AsLong(temporal_pgm_dump),
		     PyString_AsString(periodic_pgm_dump),
		     PyInt_AsLong(periodic_pgm_period));

    Py_INCREF(self);
    Py_DECREF(metadata_pystring);
    Py_DECREF(metadata);
    return (PyObject *)self;
}


static PyMethodDef Net_methods[] = {
    {"train", (PyCFunction)Net_train, METH_VARARGS | METH_KEYWORDS,
     "train the net with a block of text"},
    {"test", (PyCFunction)Net_test, METH_VARARGS | METH_KEYWORDS,
     "calculate cross entropies for a block of text"},
    {"save", (PyCFunction)Net_save, METH_VARARGS | METH_KEYWORDS,
     "Save the net"},
    {"load", (PyCFunction)Net_load, METH_VARARGS | METH_KEYWORDS | METH_CLASS,
     "Load a net (class method)"},
    {"dump_parameters", (PyCFunction)Net_dump_parameters, METH_NOARGS,
     "print net parameters"},
    {NULL}
};

static const char Net_doc[] =                                           \
    "Multi-headed predictive recurrent neural network\n\n"              \
    "Besides the ``Net()`` constructor, you can load a saved "          \
    "with the class method ``Net.load()``.\n\n"                         \
    "Net.__init__ parameters:\n"                                        \
    ":param alphabet: an Alphabet object\n"                             \
    ":param classes: sequence of class names\n"                         \
    ":param hidden_size: number of hidden recurrent nodes\n"            \
    "Optional parameters:\n"                                            \
    ":param log_file: log statistics here (default None meaning "       \
    "no log)\n"                                                         \
    ":param bptt_depth: how fat to back-propagate through time\n"       \
    ":param learn_rate: \n"                                             \
    ":param filename: save net here\n"                                  \
    ":param momentum: \n"                                               \
    ":param presynaptic_noise: \n"                                      \
    ":param rng_seed: -1 for automatic seed\n"                          \
    ":param metadata: \n"                                               \
    ":param activation: \n"                                             \
    ":param learning_method: \n"                                        \
    ":param basename: \n"                                               \
    ":param verbose: \n"                                                \
    ":param temporal_pgm_dump: \n"                                      \
    ":param periodic_pgm_dump: \n"                                      \
    ":param periodic_pgm_period: \n"                                    \
    ":param batch_size: \n"                                             \
    ":param init_method: 0: zeros, 1: flat, 2: fan-in, 3: runs\n"       \
    ;


static PyTypeObject NetType = {
    PyObject_HEAD_INIT(NULL)
    0,                            /*ob_size*/
    "charmodel.Net",              /*tp_name*/
    sizeof(Net),                  /*tp_basicsize*/
    0,                            /*tp_itemsize*/
    (destructor)Net_dealloc,      /*tp_dealloc*/
    0,                            /*tp_print*/
    0,                            /*tp_getattr*/
    0,                            /*tp_setattr*/
    0,                            /*tp_compare*/
    0,                            /*tp_repr*/
    0,                            /*tp_as_number*/
    0,                            /*tp_as_sequence*/
    0,                            /*tp_as_mapping*/
    0,                            /*tp_hash */
    0,                            /*tp_call*/
    0,                            /*tp_str*/
    0,                            /*tp_getattro*/
    0,                            /*tp_setattro*/
    0,                            /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /*tp_flags*/
    Net_doc,                      /* tp_doc */
    0,                            /* tp_traverse */
    0,                            /* tp_clear */
    0,                            /* tp_richcompare */
    0,                            /* tp_weaklistoffset */
    0,                            /* tp_iter */
    0,                            /* tp_iternext */
    Net_methods,                  /* tp_methods */
    Net_members,                  /* tp_members */
    Net_getsetters,               /* tp_getset */
    0,                            /* tp_base */
    0,                            /* tp_dict */
    0,                            /* tp_descr_get */
    0,                            /* tp_descr_set */
    0,                            /* tp_dictoffset */
    (initproc)Net_init,           /* tp_init */
    0,                            /* tp_alloc */
    Net_new,                      /* tp_new */
};


/* top level functions */

static PyObject *
Function_enable_fp_exceptions(Net *self, PyObject *nothing)
{
    feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);
    return Py_BuildValue("");
}



/* bindings for top_level */

static PyMethodDef top_level_functions[] = {
    {"enable_fp_exceptions", (PyCFunction)Function_enable_fp_exceptions,
     METH_NOARGS, "turn on some floating point exceptions"},
    {NULL}
};


/* initialisation.    */

#ifndef PyMODINIT_FUNC
#define PyMODINIT_FUNC void
#endif
PyMODINIT_FUNC
initcharmodel(void)
{
    if (PyType_Ready(&AlphabetType) < 0 ||
        PyType_Ready(&NetType) < 0)
      return;

    PyObject* m = Py_InitModule3("charmodel", top_level_functions,
                                 "Wrapper around RNN and utilities for text prediction.");

    if (m == NULL){
        DEBUG("Could not initialise charmodel module!");
        return;
    }

    int r = 0;

#define ADD_INT_CONSTANT(x) (PyModule_AddIntConstant(m, QUOTE(x), (RNN_ ##x)))

    r = r || ADD_INT_CONSTANT(MOMENTUM_WEIGHTED);
    r = r || ADD_INT_CONSTANT(MOMENTUM_NESTEROV);
    r = r || ADD_INT_CONSTANT(MOMENTUM_SIMPLIFIED_NESTEROV);
    r = r || ADD_INT_CONSTANT(MOMENTUM_CLASSICAL);
    r = r || ADD_INT_CONSTANT(ADAGRAD);
    r = r || ADD_INT_CONSTANT(ADADELTA);
    r = r || ADD_INT_CONSTANT(RPROP);

    r = r || ADD_INT_CONSTANT(RELU);
    r = r || ADD_INT_CONSTANT(RESQRT);
    r = r || ADD_INT_CONSTANT(RELOG);
    r = r || ADD_INT_CONSTANT(RETANH);
    r = r || ADD_INT_CONSTANT(RECLIP20);

    r = r || ADD_INT_CONSTANT(INIT_ZERO);
    r = r || ADD_INT_CONSTANT(INIT_FLAT);
    r = r || ADD_INT_CONSTANT(INIT_FAN_IN);
    r = r || ADD_INT_CONSTANT(INIT_RUNS);

#undef ADD_INT_CONSTANT

    if (r < 0){
        DEBUG("can't add constants to module charmodel");
        return;
    }

    Py_INCREF(&AlphabetType);
    PyModule_AddObject(m, "Alphabet", (PyObject *)&AlphabetType);
    Py_INCREF(&NetType);
    PyModule_AddObject(m, "Net", (PyObject *)&NetType);
}
