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
} Alphabet;


static void
Alphabet_dealloc(Alphabet* self)
{
    if (self->alphabet){
        rnn_char_free_alphabet(self->alphabet);
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
    char *text;
    Py_ssize_t text_len = 0;
    float threshold = 1e-5;
    float digit_adjust = 0.3;
    float alpha_adjust = 3;
    int ignore_case = 1;
    int utf8 = 1;
    int collapse_space = 1;

    static char *kwlist[] = {"text", "threshold", "digit_adjust", "alpha_adjust",
                             "ignore_case", "utf8", "collapse_space", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "s#|fffiii", kwlist, &text, &text_len,
                                     &threshold, &digit_adjust, &alpha_adjust,
                                     &ignore_case, &utf8, &collapse_space)){
        return -1;
    }

    self->alphabet = rnn_char_new_alphabet();
    rnn_char_alphabet_set_flags(self->alphabet, ignore_case, utf8, collapse_space);

    int r = rnn_char_find_alphabet_s(text, text_len, self->alphabet,
                                     threshold, digit_adjust, alpha_adjust);
    if (r) {
        rnn_char_free_alphabet(self->alphabet);
        self->alphabet = NULL;
        PyErr_Format(PyExc_ValueError, "can't find an alphabet!");
        return -1;
    }
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
    int new_len;
    if (orig_str == NULL || orig_len < 0){
        return PyErr_Format(PyExc_ValueError, "encode_text requires a string");
    }
    u8 *s = malloc(orig_len + 2);
    memcpy(s, orig_str, orig_len);
    rnn_char_collapse_buffer(self->alphabet, s, orig_len, &new_len);
    PyObject *final_obj = PyString_FromStringAndSize((char *)s, new_len);
    free(s);

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

/*
void rnn_char_dump_collapsed_text(const u8 *text, int len, const char *name,
    const char *alphabet);
*/



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
    "Alphabet objects",           /* tp_doc */
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
Net_init(Net *self, PyObject *args, PyObject *kwds)
{
    /* mandatory arguments */
    Alphabet *alphabet;
    PyObject *class_names;
    uint hidden_size;

    /* optional arguments */
    unsigned long long rng_seed = 2;
    const char *log_file = "multi-text.log";
    int bptt_depth = 30;
    float learn_rate = 0.001;
    float momentum = 0.95;
    float presynaptic_noise = 0.1;
    rnn_activation activation = RNN_RESQRT;
    int learning_method = RNN_ADAGRAD;
    int verbose = 0;
    int temporal_pgm_dump = 0;
    char *periodic_pgm_dump = NULL;
    int periodic_pgm_period = 1000;
    const char *basename = "multi-text";
    char *metadata = NULL;

    /* other vars */
    PyObject *class_name_lut;
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
                             NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!Oi|zifffKziiziizi", kwlist,
            &AlphabetType,
            &alphabet,            /* O! */
            &class_names,         /* O  */
            &hidden_size,         /* i  |  */
            &log_file,            /* z  */
            &bptt_depth,          /* i  */
            &learn_rate,          /* f  */
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
            &periodic_pgm_period  /* i  */
        )){
        return -1;
    }
    if (activation >= RNN_ACTIVATION_LAST || activation < 1){
        PyErr_Format(PyExc_ValueError, "%d is not a valid activation", activation);
        return -1;
    }

    if ((unsigned)learning_method >= RNN_LAST_LEARNING_METHOD){
        PyErr_Format(PyExc_ValueError, "%d is not a valid learning method",
            learning_method);
        return -1;
    }
    if (learning_method == RNN_ADADELTA || learning_method == RNN_RPROP){
        flags |= RNN_NET_FLAG_AUX_ARRAYS;
    }

    n_classes = PySequence_Length(class_names);
    self->n_classes = n_classes;

    if (! PySequence_Check(class_names) || n_classes < 1){
        PyErr_Format(PyExc_ValueError, "class_names should be a sequence of strings");
        return -1;
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
    net->metadata = metadata;
    char s[500];
    u32 sig = rnn_hash32(metadata);
    int wrote = snprintf(s, sizeof(s), "%s-%0" PRIx32 "i%d-h%d-o%d.net",
        basename, sig, net->input_size, net->hidden_size, net->output_size);
    if (wrote >= sizeof(s)){
        DEBUG("basename is too long!");
        return -1;
    }
    self->filename = strdup(s);

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


    self->learning_method = learning_method;
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
Net_train(Net *self, PyObject *args, PyObject *kwds)
{
    char *text;
    Py_ssize_t text_len = 0;
    PyObject *target_class;
    int target_index;
    float leakage = -1;

    static char *kwlist[] = {"text",                 /* s# */
                             "target_class",         /* O | */
                             "leakage",              /* f  */
                             NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "s#O|f", kwlist,
            &text,
            &text_len,
            &target_class,
            &leakage
        )){
        return NULL;
    }
    if (text_len < 2){
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
static const int bptt_offset_momentum = offsetof(RecurNNBPTT, momentum);
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
    {"momentum",
     (getter)Net_getfloat_bptt,
     (setter)Net_setfloat_bptt,
     "momentum",
     (void *)&bptt_offset_momentum
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
    {"class_names", T_OBJECT_EX, offsetof(Net, class_names), READONLY,
     "names of classes"},
    {"class_name_lut", T_OBJECT_EX, offsetof(Net, class_name_lut), READONLY,
     "mapping classes to indices"},
    {"batch_size", T_INT, offsetof(Net, batch_size), 0,
     "generations per mini-batch"},
    {"learning_method", T_INT, offsetof(Net, learning_method), READONLY,
     "learing method: 0: weighted, 1: Nesterov, 2: simplified N.,"
     "3: classical, 4: adagrad, 5: adadelta, 6: rprop;"},
    {"momentum", T_FLOAT, offsetof(Net, momentum), 0,
     "momentum rate (if applicable)"},
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
Net_save(Net *self, PyObject *args, PyObject *kwds)
{
    RecurNN *net = self->net;
    const char *filename = NULL;
    int backup = 1;

    static char *kwlist[] = {"filename",             /* s */
                             "backup",               /* i */
                             NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|si", kwlist,
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


static PyMethodDef Net_methods[] = {
    {"train", (PyCFunction)Net_train, METH_VARARGS | METH_KEYWORDS,
     "train the net with a block of text"},
    {"save", (PyCFunction)Net_save, METH_VARARGS | METH_KEYWORDS,
     "Save the net"},
    {"dump_parameters", (PyCFunction)Net_dump_parameters, METH_NOARGS,
     "print net parameters"},
    {NULL}
};

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
    "Net objects",                /* tp_doc */
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




/*
void rnn_char_init_schedule(RnnCharSchedule *s, int recent_len,
    float learn_rate_min, float learn_rate_mul, int adjust_noise);

float rnn_char_calc_ventropy(RnnCharModel *model, RnnCharVentropy *v, int lap);
*/
/*
int rnn_char_confabulate(RecurNN *net, char *dest, int char_len,
    int byte_len, RnnCharAlphabet* a, float bias, int *prev_char,
    int start_point, int stop_point);
*/
/*
void rnn_char_init_ventropy(RnnCharVentropy *v, RecurNN *net, const u8 *text,
    const int len, const int lap);
*/
/*
int rnn_char_epoch(RnnCharModel *model, RecurNN *confab_net, RnnCharVentropy *v,
    const u8 *text, const int len,
    const int start, const int stop,
    float confab_bias, int confab_size, int confab_line_end,
    int quietness);
*/
/*
char *rnn_char_construct_metadata(const struct RnnCharMetadata *m);
int rnn_char_load_metadata(const char *metadata, struct RnnCharMetadata *m);

void rnn_char_free_metadata_items(struct RnnCharMetadata *m);

char* rnn_char_construct_net_filename(struct RnnCharMetadata *m,
    const char *basename, int input_size, int bottom_size, int hidden_size,
    int output_size);

int rnn_char_check_metadata(RecurNN *net, struct RnnCharMetadata *m,
    bool trust_file_metadata, bool force_metadata);

void rnn_char_copy_metadata_items(struct RnnCharMetadata *src,\
    struct RnnCharMetadata *dest);
*/
/*
RnnCharAlphabet *rnn_char_new_alphabet_from_net(RecurNN *net);
*/
/*
int rnn_char_prime(RecurNN *net, RnnCharAlphabet *alphabet,
    const u8 *text, const int len);
*/
/*
double rnn_char_cross_entropy(RecurNN *net, RnnCharAlphabet *alphabet,
    const u8 *text, const int len, const int ignore_first,
    const u8 *prefix_text, const int prefix_len);

*/

static PyObject *
Function_enable_fp_exceptions(Net *self, PyObject *nothing)
{
    feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);
    return Py_BuildValue("");
}


/**********************************************************************/
/* method binding structs                                             */
/**********************************************************************/
/* bindings for top_level */
static PyMethodDef top_level_functions[] = {
    {"enable_fp_exceptions", (PyCFunction)Function_enable_fp_exceptions,
     METH_NOARGS, "turn on some floating point exceptions"},
    {NULL}
};




/**********************************************************************/
/* initialisation.                                                    */
/**********************************************************************/


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
