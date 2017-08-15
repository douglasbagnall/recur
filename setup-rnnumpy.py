from distutils.core import setup
from distutils.extension import Extension
import numpy as np

setup(
    name = "rnnumpy",
    ext_modules = [
        Extension(
            name="rnnumpy",
            library_dirs=['.'],
            libraries=['cdb'],
            sources=["py-recur-numpy.c", "recur-nn.c",
                     "recur-nn-io.c", "recur-nn-init.c"],
            depends=["pgm_dump.h", "badmaths.h",
                     "path.h", #"py-recur-numpy.h",
                     "recur-common.h", "recur-nn.h"],
            include_dirs = [".", np.get_include()],
            define_macros = [('_GNU_SOURCE', None),
                             ('VECTOR', None)],
            extra_compile_args = ['-march=native', '-ggdb', '-std=gnu11',
                                  '-I.', '-Wall', '-O3', '-ffast-math',
                                  '-fno-inline', '-DVECTOR', '-DPIC',
            ],
            language="c"
        )
    ]
)
