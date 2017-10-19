/* Copyright 2017 Catalyst IT Ltd.  LGPL.
 *
 * Written by Douglas Bagnall <douglas.bagnall@catalyst.net.nz>
 *
 * Read in numpy arrays for input, output, and target values.
*/
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include "structmember.h"
#include "structseq.h"
#include "py-recur-helpers.h"
#include <numpy/arrayobject.h>
#include "recur-nn.h"
#include "badmaths.h"
#include <fenv.h>
#include <stdbool.h>
#include "colour.h"

#define DEFAULT_ADAGRAD_BALLAST 100
#define DEFAULT_ADADELTA_BALLAST 100

#define BATCH_SIZE 80

/*net object. see py-recur-helpers.h */
typedef struct {
    BaseNet_HEAD
    uint *seen_counts;
    uint seen_sum;
    uint used_sum;
} Net;

static PyObject *PyExc_RecurError;


static void
Net_dealloc(Net* self)
{
    if (self->net) {
        /* save first? */
        rnn_delete_net(self->net);
        self->net = NULL;
    }
    self->ob_type->tp_free((PyObject*)self);
}

static PyObject *
Net_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    Net *self = (Net *)PyType_GenericNew(type, args, kwds);
    self->batch_size = 1;
    return (PyObject *)self;
}


static PyArrayObject *
get_2d_array_check_size(PyObject *arg, uint w, uint h,
                        const char *name)
{
    PyArrayObject *array = NULL;
    npy_intp *shape;
    array = (PyArrayObject *)PyArray_FROM_OTF(arg, NPY_FLOAT32,
                                              NPY_ARRAY_IN_ARRAY);
    if (array == NULL) {
        return NULL;
    }
    if (PyArray_NDIM(array) != 2) {
        PyErr_Format(PyExc_ValueError,
                     "%s array should be 2 dimension, not %d",
                     name, PyArray_NDIM(array));
        goto error;
    }
    shape = PyArray_SHAPE(array);

    /* zero h, w means we don't care/know about that dimension */
    if (h != 0 && shape[0] != h) {
        PyErr_Format(PyExc_ValueError,
                     "%s array has %ld rows, expected %u",
                     name, shape[0], h);
        goto error;
    }
    if (w != 0 && shape[1] != w) {
        PyErr_Format(PyExc_ValueError,
                     "%s array has %ld columns, expected %u",
                     name, shape[1], w);
        goto error;
    }

    return array;
 error:
    Py_DECREF(array);
    return NULL;
}

static int
Net_init(Net *self, PyObject *args, PyObject *kwds)
{
    /* mandatory arguments */
    uint input_size;
    uint hidden_size;
    uint output_size;

    /* optional arguments */
    unsigned long long rng_seed = 2;
    const char *log_file = NULL;
    const char *filename = NULL;
    int bptt_depth = 30;
    float learn_rate = 0.1;
    float momentum = 0.95;
    float presynaptic_noise = 0.0;
    rnn_activation activation = RNN_RELU;
    int learning_method = RNN_ADAGRAD;
    int verbose = 0;
    int batch_size = BATCH_SIZE;
    int init_method = RNN_INIT_FLAT;
    int temporal_pgm_dump = 0;
    char *periodic_pgm_dump = NULL;
    int periodic_pgm_period = 1000;
    const char *basename = NULL;
    char *metadata = NULL;
    float ballast = -1;

    /* other vars */
    RecurNN *net;
    u32 flags = RNN_NET_FLAG_STANDARD | RNN_NET_FLAG_BPTT_ADAPTIVE_MIN_ERROR;

    static char *kwlist[] = {"input_size",           /* i  */
                             "hidden_size",          /* i  */
                             "output_size",          /* i  |  */
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
                             "ballast",     /* f  */
                             NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "iii|zifzffKziiziiziiif",
				     kwlist,
				     &input_size,          /* i  */
				     &hidden_size,         /* i  */
				     &output_size,         /* i  |  */
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
				     &init_method,         /* i  */
                                     &ballast              /* f  */
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

    if (learning_method == RNN_ADADELTA || learning_method == RNN_RPROP){
        flags |= RNN_NET_FLAG_AUX_ARRAYS;
    }

    net = rnn_new(
        input_size,
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
    self->batch_size = batch_size;

    if (init_method < RNN_INIT_ZERO || init_method >= RNN_INIT_LAST){
        init_method = RNN_INIT_FLAT;
    }
    rnn_randomise_weights_simple(net, init_method);

    net->metadata = metadata;
    if (basename != NULL){
	set_net_filename(RNNPY_BASE_NET(self), filename, basename, metadata);
    }

    self->learning_method = learning_method;
    self->seen_counts = calloc(self->net->output_size, sizeof(uint));
    self->seen_sum = 1;
    self->used_sum = 1;
    if (self->seen_counts == NULL) {
        return -1;
    }

    switch(learning_method){
    case RNN_ADAGRAD:
        if (ballast <= 0) {
            ballast = DEFAULT_ADAGRAD_BALLAST;
        }
        rnn_set_momentum_values(self->net, ballast);
        break;
    case RNN_ADADELTA:
        if (ballast <= 0) {
            ballast = DEFAULT_ADADELTA_BALLAST;
        }
        rnn_set_momentum_values(self->net, ballast);
        break;
    case RNN_RPROP:
        rnn_set_aux_values(self->net, 1);
        break;
    }
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




static PyObject *
Net_train(Net *self, PyObject *args, PyObject *kwds)
{
    uint epoch, i, j;
    PyObject *input_arg = NULL;
    PyArrayObject *inputs = NULL;
    PyObject *target_arg = NULL;
    PyArrayObject *targets = NULL;
    PyObject *mask_arg = NULL;
    PyArrayObject *mask = NULL;
    float learn_rate = -1;
    float balance = 0;
    npy_intp *shape;
    RecurNN *net = self->net;
    uint n_epochs = 0;

    if (net == NULL) {
        PyErr_Format(PyExc_RecurError, "Not ready: we have no net");
        return NULL;
    }
    RecurNNBPTT *bptt = net->bptt;
    if (bptt == NULL) {
        PyErr_Format(PyExc_RecurError,
                     "This net is not set up for training (no bptt)");
        return NULL;
    }

    static char *kwlist[] = {"features",             /* O  */
                             "targets",              /* O  */
                             "n_epochs",             /* I  */
                             "mask",                 /* O  */
                             "learn_rate",           /* f  */
                             "balance",              /* f  */
                             NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OOI|Off",
				     kwlist,
                                     &input_arg,           /* O  */
                                     &target_arg,          /* O  */
                                     &n_epochs,            /* I  */
                                     &mask_arg,            /* O  */
                                     &learn_rate,          /* f  */
                                     &balance,             /* f  */
                                     NULL)) {
        return NULL;
    }

    inputs = get_2d_array_check_size(input_arg,
                                     net->input_size, 0,
                                     "input");
    if (inputs == NULL) {
        return NULL;
    }

    /* targets need same height as inputs */
    shape = PyArray_SHAPE(inputs);

    targets = get_2d_array_check_size(target_arg,
                                      net->output_size, shape[0],
                                      "target");
    if (targets == NULL) {
	Py_DECREF(inputs);
        return NULL;
    }

    if (mask_arg != NULL) {
        mask = (PyArrayObject *)PyArray_FROM_OTF(mask_arg, NPY_BOOL,
                                                 NPY_ARRAY_IN_ARRAY);
        if (mask == NULL) {
            goto error;
        }
        if (PyArray_NDIM(mask) != 1) {
            PyErr_Format(PyExc_ValueError, "mask should be 1 dimensional");
            goto error;
        }
        npy_intp *shape2 = PyArray_SHAPE(mask);

        if (shape[0] != shape2[0]) {
            PyErr_Format(PyExc_ValueError, "mask is %ld long; should be %ld",
                         shape2[0], shape[0]);
            goto error;
        }
    }
    float *restrict idata = PyArray_DATA(inputs);
    float *restrict tdata = PyArray_DATA(targets);
    float *restrict error = bptt->o_error;
    bool *restrict mdata = NULL;
    if (mask != NULL) {
        mdata = PyArray_DATA(mask);
    }

    if (learn_rate > 0) {
        bptt->learn_rate = learn_rate;
    }

    DEBUG("balance is %f; learn-rate %f", balance, bptt->learn_rate);
    float rolling_accuracy = 0.5;
    for (epoch = 1; epoch <= n_epochs; epoch++) {
        uint countdown = self->batch_size;
        float epoch_accuracy = 0;
        float epoch_error = 0;
        uint epoch_count = 0;
	for (i = 0; i < shape[0]; i++) {
	    rnn_bptt_advance(net);
	    float *restrict irow = idata + i * net->input_size;
	    float *restrict trow = tdata + i * net->output_size;
	    float *restrict answer = rnn_opinion(net, irow,
						 net->presynaptic_noise);
            if (mdata && mdata[i] == false) {
                continue;
            }
            if (balance != 0) {
                uint target = 0;
                for (j = 0; j < net->output_size; j++) {
                    if (trow[j] > trow[target]) {
                        target = j;
                    }
                }
                self->seen_counts[target]++;
                self->seen_sum++;
                float p = 1.0f - (self->seen_counts[target] /
                                  (float)self->seen_sum);
                float training_p = powf(p, balance);

                if (training_p < rand_float(&self->net->rng)) {
                    /* no training for this one. */
                    continue;
                }
                self->used_sum++;
                rnn_log_float(net, "use_ratio",
                              (float)self->used_sum / self->seen_sum);
            }

	    softmax_best_guess(error, answer, net->output_size);
            float error_t = 0;
            rolling_accuracy *= 255.0;
            float accuracy = 0;
	    for (j = 0; j < net->output_size; j++) {
		error[j] += trow[j];
                if (trow[j]) {
                    error_t += error[j];
                    accuracy += fabsf(error[j]) < 0.5;
                }
	    }
            epoch_count++;
            rolling_accuracy += accuracy;
            epoch_accuracy += accuracy;
            epoch_error += error_t;
            rolling_accuracy /= 256.0;
            rnn_log_float(net, "error_1", error[1]);
            rnn_log_float(net, "rolling_accuracy", rolling_accuracy);
            rnn_log_float(net, "t1", trow[1]);
            rnn_log_float(net, "error_t", error_t);
            countdown--;
	    if (countdown == 0) {
		rnn_apply_learning(net, self->learning_method, self->momentum);
		rnn_bptt_calc_deltas(net, 0, NULL);
                countdown = self->batch_size;
	    } else {
		rnn_bptt_calc_deltas(net, 1, NULL);
            }
        }
        if (epoch_count != 0) {
	    epoch_accuracy /= epoch_count;
	    epoch_error /= epoch_count;
	    DEBUG("epoch %3u trained on %5u; alleged"
		  " accuracy %s%.2f" C_NORMAL
		  " error %s%.2f" C_NORMAL,
		  epoch,
		  epoch_count,
		  colourise_float01(epoch_accuracy, true),
		  epoch_accuracy,
		  colourise_float01(1.0f - (epoch_error * epoch_error), false),
		  epoch_error);
	} else {
		DEBUG("epoch %u trained on zero examples!", epoch);
	}
        if (PyErr_CheckSignals() == -1) {
            /* this will allow a control-C to interrupt. */
                DEBUG("interrupted");
            break;
        }
    }

    Py_DECREF(inputs);
    Py_DECREF(targets);
    return Py_BuildValue("");
 error:
    if (inputs != NULL) {
        Py_DECREF(inputs);
    }
    if (targets != NULL) {
        Py_DECREF(targets);
    }
    if (mask != NULL) {
        Py_DECREF(mask);
    }
    return NULL;
}


static PyObject *
Net_classify(Net *self, PyObject *args)
{
    PyObject *array_arg = NULL;
    PyArrayObject *array = NULL;
    PyObject *results = NULL;
    RecurNN *net = self->net;
    npy_intp *in_shape;
    npy_intp out_shape[2];
    uint i;

    if (net == NULL) {
        PyErr_Format(PyExc_RecurError, "Not ready: we have no net");
        return NULL;
    }

    if (!PyArg_ParseTuple(args, "O", &array_arg)) {
        return NULL;
    }

    array = get_2d_array_check_size(array_arg,
                                    net->input_size, 0,
                                    "input");
    if (array == NULL) {
        return NULL;
    }


    /* targets need same height as inputs */
    in_shape = PyArray_SHAPE(array);
    out_shape[0] = in_shape[0];
    out_shape[1] = net->output_size;

    results = PyArray_ZEROS(2, out_shape, NPY_FLOAT32, 0);

    float *restrict idata = PyArray_DATA(array);
    float *restrict rdata = PyArray_DATA((PyArrayObject *)results);

    for (i = 0; i < out_shape[0]; i++) {
	float *restrict irow = idata + i * net->input_size;
	float *restrict rrow = rdata + i * net->output_size;
	float *restrict answer = rnn_opinion(net, irow,
					     net->presynaptic_noise);
	softmax(rrow, answer, net->output_size);
    }

    Py_DECREF(array);
    return results;
}


static const char Net_doc[] =                                           \
    "Numpy interface to Recur recurrent neural network\n\n"             \
    "Besides the ``Net()`` constructor, you can load a saved "          \
    "with the class method ``Net.load()``.\n\n"                         \
    ;


static PyMethodDef Net_methods[] = {
    {"train", (PyCFunction)Net_train, METH_VARARGS | METH_KEYWORDS,
     "train the net with inpuit and target arrays"},
    {"classify", (PyCFunction)Net_classify,
     METH_VARARGS | METH_KEYWORDS,
     "generate answers for an input array"},
    {NULL}
};

static PyMemberDef Net_members[] = {
    {NULL}
};


static PyTypeObject NetType = {
    PyObject_HEAD_INIT(NULL)
    0,                            /*ob_size*/
    "rnnumpy.Net",                /*tp_name*/
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


#ifndef PyMODINIT_FUNC
#define PyMODINIT_FUNC void
#endif
PyMODINIT_FUNC
initrnnumpy(void)
{
    if (PyType_Ready(&NetType) < 0) {
        return;
    }

    PyObject* m = Py_InitModule3("rnnumpy", top_level_functions,
                                 "Wrapper around RNN for numpy arrays.");

    if (m == NULL) {
        DEBUG("Could not initialise numpy rnn module!");
        return;
    }

    int r = add_module_constants(m);
    if (r < 0){
        DEBUG("can't add constants to module charmodel");
        return;
    }

    PyExc_RecurError = PyErr_NewException((char *)"rnnumpy.RecurError",
					  NULL, NULL);
    PyModule_AddObject(m, "RecurError", PyExc_RecurError);

    import_array();
    Py_INCREF(&NetType);
    PyModule_AddObject(m, "Net", (PyObject *)&NetType);
}
