#ifndef HAVE_PY_RECUR_HELPERS_H
#define HAVE_PY_RECUR_HELPERS_H
#include <Python.h>
#include "structmember.h"
#include "structseq.h"
#include "recur-nn.h"

#define BaseNet_HEAD				\
	PyObject_HEAD				\
	RecurNN *net;				\
	rnn_learning_method learning_method;	\
	float momentum;				\
	int batch_size;				\
	const char *filename;			\

typedef struct {
	BaseNet_HEAD
} BaseNet;


#define RNNPY_BASE_NET(x) ((BaseNet *)(x))

static int
set_net_filename(BaseNet *self, const char *filename, const char *basename,
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

/* Net_{get,set}float_{rnn,bptt}. These access float attributes that are
   pointed to via an integer offset into the struct.

   Without this we'd need separate access functions for each attribute.
 */

static UNUSED PyObject *
Net_getfloat_rnn(BaseNet *self, int *offset)
{
    void *addr = ((void *)self->net) + *offset;
    float f = *(float *)addr;
    return PyFloat_FromDouble((double)f);
}

static UNUSED int
Net_setfloat_rnn(BaseNet *self, PyObject *value, int *offset)
{
    PyObject *pyfloat = PyNumber_Float(value);
    if (pyfloat == NULL){
        return -1;
    }
    void *addr = ((void *)self->net) + *offset;
    float f = PyFloat_AS_DOUBLE(pyfloat);
    *(float *)addr = f;
    return 0;
}

static UNUSED PyObject *
Net_getfloat_bptt(BaseNet *self, int *offset)
{
    void *addr = ((void *)self->net->bptt) + *offset;
    float f = *(float *)addr;
    return PyFloat_FromDouble((double)f);
}

static UNUSED int
Net_setfloat_bptt(BaseNet *self, PyObject *value, int *offset)
{
    PyObject *pyfloat = PyNumber_Float(value);
    if (pyfloat == NULL){
        return -1;
    }
    void *addr = ((void *)self->net->bptt) + *offset;
    float f = PyFloat_AS_DOUBLE(pyfloat);
    *(float *)addr = f;
    return 0;
}


static int add_module_constants(PyObject* m)
{
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
    r = r || ADD_INT_CONSTANT(RECLIP20);

    r = r || ADD_INT_CONSTANT(INIT_ZERO);
    r = r || ADD_INT_CONSTANT(INIT_FLAT);
    r = r || ADD_INT_CONSTANT(INIT_FAN_IN);
    r = r || ADD_INT_CONSTANT(INIT_RUNS);

#undef ADD_INT_CONSTANT

    return r;
}


#endif
