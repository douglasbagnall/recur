/* Copyright 2015 Douglas Bagnall <douglas@halo.gen.nz> LGPL

Python bindings for text RNNs.
*/
#include <Python.h>
#include "charmodel.h"
#include "utf8.h"
#include "py-recur-text.h"
#include <math.h>
#include <string.h>

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
    PyObject *class_names;
    PyObject *class_name_lut;
    rnn_learning_method learning_method;
} Net;


static void
Net_dealloc(Net* self)
{
    if (self->net){
        /* save first? */
        rnn_delete_net(self->net);
    }
    self->ob_type->tp_free((PyObject*)self);
}

static PyObject *
Net_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    Net *self = (Net *)PyType_GenericNew(type, args, kwds);
    self->net = NULL;
    self->class_names = NULL;
    self->class_name_lut = NULL;
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
    const char *log_file = "muli-text.log";
    int bptt_depth = 30;
    float learn_rate = 0.001;
    float momentum = 0.95;
    float presynaptic_noise = 0.1;
    rnn_activation activation = RNN_RESQRT;
    int learning_method = RNN_ADAGRAD;

    /* other vars */
    PyObject *class_name_lut;
    int n_classes;
    uint output_size;
    u32 flags = RNN_NET_FLAG_STANDARD | RNN_NET_FLAG_BPTT_ADAPTIVE_MIN_ERROR;

    static char *kwlist[] = {"alphabet",             /* O! */
                             "classes",              /* O  */
                             "hidden_size",          /* i  |  */
                             "log_file",             /* s  */
                             "bptt_depth",           /* i  */
                             "learn_rate",           /* f  */
                             "momentum",             /* f  */
                             "presynaptic_noise",    /* f  */
                             "rng_seed",             /* K  */
                             "activation",           /* i  */
                             "learning_method",      /* i  */
                             NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!Oi|sifffKii", kwlist,
            &AlphabetType,
            &alphabet,          /* O! */
            &class_names,       /* O  */
            &hidden_size,       /* i  |  */
            &log_file,          /* s  */
            &bptt_depth,        /* i  */
            &learn_rate,        /* f  */
            &momentum,          /* f  */
            &presynaptic_noise, /* f  */
            &rng_seed,          /* K  */
            &activation,        /* i  */
            &learning_method    /* i  */
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

    if (! PySequence_Check(class_names) || n_classes < 1){
        PyErr_Format(PyExc_ValueError, "class_names should be a sequence of strings");
        return -1;
    }

    output_size = alphabet->alphabet->len * n_classes;

    self->net = rnn_new(
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

    self->learning_method = learning_method;
    self->class_names = class_names;
    class_name_lut = PyDict_New();
    for (long i = 0; i < n_classes; i++){
        PyObject *k = PySequence_GetItem(class_names, i);
        PyObject *v = PyInt_FromLong(i);
        PyDict_SetItem(class_name_lut, k, v);
    }
    self->class_name_lut = class_name_lut;

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

static PyGetSetDef Net_getsetters[] = {
    {NULL}  /* Sentinel */
};


static PyMemberDef Net_members[] = {
    {"class_names", T_OBJECT_EX, offsetof(Net, class_names), READONLY,
     "names of classes"},
    {"class_name_lut", T_OBJECT_EX, offsetof(Net, class_name_lut), READONLY,
     "mapping classes to indices"},
    {NULL}
};

static PyMethodDef Net_methods[] = {
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



/**********************************************************************/
/* method binding structs                                             */
/**********************************************************************/
/* bindings for top_level */
static PyMethodDef top_level_functions[] = {
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
