import os
import sys
import re
import platform

import lit.util
import lit.formats

import lit.llvm

config.llvm_tools_dir = os.environ.get('LLVM_TOOLS_DIR')
config.lit_tools_dir = '' # Intentionally empty
lit.llvm.initialize(lit_config, config)

from lit.llvm import llvm_config

config.name = 'GPUCompiler'
config.suffixes = ['.ll', '.jl']
config.test_source_root = os.path.dirname(__file__)
execute_external = platform.system() != 'Windows'
config.test_format = lit.formats.ShTest(execute_external)
config.substitutions.append(('%shlibext', '.dylib' if platform.system() == 'Darwin' else '.dll' if
    platform.system() == 'Windows' else '.so'))

# Lit uses an empty environment so we copy over the settings from
# Pkg.test(), most importantly that's the `JULIA_LOAD_PATH`.
config.environment['JULIA_PROJECT'] = os.environ.get('JULIA_PROJECT')

LOAD_PATH  = os.environ.get('JULIA_LOAD_PATH')
DEPOT_PATH = os.environ.get('JULIA_DEPOT_PATH')

if LOAD_PATH:
    config.environment['JULIA_LOAD_PATH'] = LOAD_PATH

if DEPOT_PATH:
    config.environment['JULIA_DEPOT_PATH'] = DEPOT_PATH

llvm_config.use_default_substitutions()

