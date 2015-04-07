from distutils.core import setup
from distutils.extension import Extension

setup(
    name = "charmodel",
    ext_modules = [
        Extension(
            name="charmodel",
            library_dirs=['.'],
            libraries=['cdb'],
            sources=["py-recur-text.c", "charmodel-multi-predict.c",
                     "charmodel-predict.c", "charmodel-init.c", "recur-nn.c",
                     "recur-nn-io.c", "recur-nn-init.c"],
            depends=["charmodel.h", "pgm_dump.h", "badmaths.h",
                     "charmodel-helpers.h", "path.h", "py-recur-text.h",
                     "recur-common.h", "recur-nn.h"],  # ..., etc.
            include_dirs = ["."],
            extra_compile_args=['-ggdb', '-std=gnu11', '-I.', '-Wall',
                                '-D_GNU_SOURCE'],
            language="c"
        )
    ]
)
