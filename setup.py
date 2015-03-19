from distutils.core import setup
from distutils.extension import Extension

setup(
    name = "charmodel",
    ext_modules = [
        Extension(
            name="charmodel",
            library_dirs=['.'],
            libraries=['cdb', 'charmodel'],
            sources=["py-recur-text.c"],
        #, "charmodel-predict.c", "charmodel-init.c",
        #"recur-nn.c", "recur-nn-io.c", "recur-nn-init.c"],
            include_dirs = ["."],
            extra_compile_args=['-ggdb', '-std=gnu11', '-I.', '-Wall'],
            language="c"
        )
    ]
)
