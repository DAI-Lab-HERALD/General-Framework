import os
#########################################################################################
# The following code is used to run the cpp extensions                                  #
#########################################################################################

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

if CUDA_HOME is not None:
    os.environ['CUDA_HOME'] = CUDA_HOME
    print("CUDA_HOME is set to:", CUDA_HOME)
else:
    print("CUDA_HOME is not found. Make sure CUDA is installed correctly.")

# Helper function to create CUDAExtension
def make_cuda_ext(name, module, sources):
    full_name = '%s.%s' % (module, name)
    source_names = [os.path.join(*module.split('.'), src) for src in sources]
    return CUDAExtension(name=full_name, sources=source_names)


# Define your CUDA modules
CUDA_modules = [
    make_cuda_ext(
        name='knn_cuda',
        module='Models.MTR_unitraj.model.ops.knn',
        sources=[
            'src/knn.cpp',
            'src/knn_gpu.cu',
            'src/knn_api.cpp',
        ],
    ),
    make_cuda_ext(
        name='attention_cuda',
        module='Models.MTR_unitraj.model.ops.attention',
        sources=[
            'src/attention_api.cpp',
            'src/attention_func_v2.cpp',
            'src/attention_func.cpp',
            'src/attention_value_computation_kernel_v2.cu',
            'src/attention_value_computation_kernel.cu',
            'src/attention_weight_computation_kernel_v2.cu',
            'src/attention_weight_computation_kernel.cu',
        ],
    ),
]

# Setup function
setup(
    name='MTR_unitraj_cuda',
    ext_modules=CUDA_modules,
    cmdclass={'build_ext': BuildExtension},  # Use PyTorch's BuildExtension
    script_args=['build_ext', '--inplace'],
)

