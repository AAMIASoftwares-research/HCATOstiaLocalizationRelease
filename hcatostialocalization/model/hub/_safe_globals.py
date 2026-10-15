import torch
import numpy
import collections
import builtins

# Dynamically extract the exact dtype class objects (e.g., numpy.dtypes.Float64DType)
# This works universally across NumPy 1.x and 2.x without hardcoding namespaces.
_numpy_dtype_classes = [
    type(numpy.dtype(numpy.float16)),
    type(numpy.dtype(numpy.float32)),
    type(numpy.dtype(numpy.float64)),
    type(numpy.dtype(numpy.int8)),
    type(numpy.dtype(numpy.int16)),
    type(numpy.dtype(numpy.int32)),
    type(numpy.dtype(numpy.int64)),
    type(numpy.dtype(numpy.uint8)),
    type(numpy.dtype(numpy.uint16)),
    type(numpy.dtype(numpy.uint32)),
    type(numpy.dtype(numpy.uint64)),
    type(numpy.dtype(numpy.bool_)),
]

safe_checkpoint_globals = [
    # Numpy internal structures
    getattr(numpy, "_core", getattr(numpy, "core", None)).multiarray.scalar, 
    numpy.ndarray,
    numpy.dtype,
    
    # Python advanced collections
    collections.OrderedDict,
    collections.defaultdict,
    set,
    
    # Standard Python built-ins
    builtins.float,
    builtins.int,
    builtins.str,
    builtins.bool,
    builtins.dict,
    builtins.list,
    builtins.tuple,
    builtins.slice,
]

# Merge the dynamically generated dtype classes into the safe list
safe_checkpoint_globals.extend(_numpy_dtype_classes)

# Clean out any potential None values and remove duplicates
safe_checkpoint_globals = list(set([g for g in safe_checkpoint_globals if g is not None]))

