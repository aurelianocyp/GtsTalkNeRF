import os
import torch
from packaging import version as pver
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

_src_path = os.path.dirname(os.path.abspath(__file__))
stdversion = 'c++14' if pver.parse(torch.__version__) < pver.parse('2.0') else 'c++17'

nvcc_flags = [
    '-I' + os.path.join(_src_path, "glm"),
    '-O3', '-std='+stdversion,
    '-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__', '-U__CUDA_NO_HALF2_OPERATORS__',
    '-use_fast_math'
]

if os.name == "posix":
    c_flags = ['-O3', '-std='+stdversion]
elif os.name == "nt":
    c_flags = ['/O2', '/std:'+stdversion]

    # find cl.exe
    def find_cl_path():
        import glob
        for edition in ["Enterprise", "Professional", "BuildTools", "Community"]:
            paths = sorted(glob.glob(r"C:\\Program Files (x86)\\Microsoft Visual Studio\\*\\%s\\VC\\Tools\\MSVC\\*\\bin\\Hostx64\\x64" % edition), reverse=True)
            if paths:
                return paths[0]

    # If cl.exe is not on path, try to find it.
    if os.system("where cl.exe >nul 2>nul") != 0:
        cl_path = find_cl_path()
        if cl_path is None:
            raise RuntimeError("Could not locate a supported Microsoft Visual C++ installation")
        os.environ["PATH"] += ";" + cl_path

setup(
    name='gsencoder', # package name, import this to use python API
    ext_modules=[
        CUDAExtension(
            name='_gsencoder', # extension name, import this to use CUDA API
            sources=[os.path.join(_src_path, 'src', f) for f in [
                'freqencoder.cu',
                'compute_cov3d.cu',
                'compute_gauss.cu',
                'bindings.cpp',
            ]],
            extra_compile_args={
                'cxx': c_flags,
                'nvcc': nvcc_flags,
            }
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension,
    }
)