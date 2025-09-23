import logging
from collections.abc import Callable
from typing import Any

import numpy as np

from ..algebra import register_property
from ..codegen import (
    CContext,
    CKernel,
    CModule,
    NumpyBufferFType,
    c_literal,
    c_type,
    init_shared_lib,
)
from ..util import config
from . import nodes as apl

logger = logging.getLogger(__name__)


class APL2CGenerator:
    def __call__(self, prgm: apl.APLNode):
        ctx = APL2CContext()
        ctx(prgm)
        return ctx.emit_file()


class APL2CCompiler:
    """
    A class to compile and run APL.
    """

    def __init__(self, cc=None, cflags=None, shared_cflags=None):
        if cc is None:
            cc = config.get("cc")
        if cflags is None:
            cflags = config.get("cflags").split()
        if shared_cflags is None:
            shared_cflags = config.get("shared_cflags").split()
        self.cc = cc
        self.cflags = cflags
        self.shared_cflags = shared_cflags

    def __call__(self, prgm):
        ctx = APL2CContext()
        ctx(prgm)
        c_code = ctx.emit_file()
        c_globals = ctx.emit_globals()
        logger.info(f"Compiling C code:\n{c_code}")
        lib = init_shared_lib(
            c_code=c_code,
            globals=c_globals,
            cc=self.cc,
            cflags=(*self.cflags, *self.shared_cflags),
        )
        kernels = {}
        if prgm.head() != apl.Module:
            raise ValueError(
                "APL2CCompiler expects a Module as the head of the program, "
                f"got {type(prgm.head())}"
            )
        for func in prgm.funcs:
            match func:
                case apl.Function(apl.Variable(func_name, return_t), args, _):
                    arg_ts = [arg.result_ftype for arg in args]
                    kern = CKernel(getattr(lib, func_name), return_t, arg_ts)
                    kernels[func_name] = kern
                case _:
                    raise NotImplementedError(
                        f"Unrecognized function type: {type(func)}"
                    )
        return CModule(lib, kernels)


def _c_mkArray(ctx, *args: Any):
    # make a fresh variable name for the array
    arr_n = ctx.freshen("arr")
    dtype = args[0].result_ftype
    arr_ftype = NumpyBufferFType(dtype, 1)
    arr_ctr = arr_ftype.c_alloc(ctx, [apl.Literal(len(args))])
    arr_var = apl.Variable(arr_n, arr_ftype)
    ctx.exec(f"{ctx.feed}{ctx.ctype_name(c_type(arr_ftype))} {arr_n} = {arr_ctr};\n")
    for i, val in enumerate(args):
        arr_var = apl.Variable(arr_n, arr_ftype)
        idx = apl.Literal(i)
        arr_ftype.c_store(ctx, arr_var, (idx,), val)
    return arr_n


# Register the return type for mkArray function
# The return type should be NumpyBufferFType with proper dtype and ndim
register_property(
    "mkArray",
    "__call__",
    "return_type",
    lambda op, *arg_types: NumpyBufferFType(arg_types[0] if arg_types else np.int64, 1),
)


def _c_add(ctx, arr1: apl.Variable, arr2: apl.Variable):
    """
    Assignment: Implement the _c_add function to generate C code for element-wise 
                addition of two arrays.

    Parameters
    ----------
    ctx : APL2CContext
        The code generation context, used to emit C code statements and manage variable names.
    arr1 : apl.Variable
        The first input array, represented as an apl.Variable with type NumpyBufferFType(np.int64, ndim),
        where ndim is the number of dimensions (e.g., 1 for 1D arrays, 2 for 2D arrays).
    arr2 : apl.Variable
        The second input array, with the same shape and dtype as arr1.

    Returns
    -------
    str
        The name of the result variable (a string) that represents the output 
        NumpyBuffer in the generated C code.
    """
    
    #Start your implementation here. 
    pass

register_property(
    "add",
    "__call__",
    "return_type",
    lambda op, *arg_types: arg_types[0]
    if arg_types and isinstance(arg_types[0], NumpyBufferFType)
    else NumpyBufferFType(np.int64, 1),
)


def _c_sub(ctx, arr1: apl.Variable, arr2: apl.Variable):
    """
    Assignment: Implement the _c_sub function to generate C code for element-wise subtraction of two arrays.

    Parameters
    ----------
    ctx : APL2CContext
        The code generation context, used to emit C code statements and manage variable names.
    arr1 : apl.Variable
        The first input array, represented as an apl.Variable with type NumpyBufferFType(np.int64, ndim),
        where ndim is the number of dimensions (e.g., 1 for 1D arrays, 2 for 2D arrays).
    arr2 : apl.Variable
        The second input array, with the same shape and dtype as arr1.

    Returns
    -------
    str
        The name of the result variable (a string) that represents the output NumpyBuffer in the generated C code.
    """
    
    #Start your implementation here. 
    pass


register_property(
    "sub",
    "__call__",
    "return_type",
    lambda op, *arg_types: arg_types[0]
    if arg_types and isinstance(arg_types[0], NumpyBufferFType)
    else NumpyBufferFType(np.int64, 1),
)


def _c_neg(ctx, arr: apl.Variable):
    """
    Assignment: Implement the _c_neg function to generate C code for element-wise negation of an array.

    Parameters
    ----------
    ctx : APL2CContext
        The code generation context, used to emit C code statements and manage variable names.
    arr : apl.Variable
        The input array, represented as an apl.Variable with type NumpyBufferFType(np.int64, ndim),
        where ndim is the number of dimensions (e.g., 1 for 1D arrays, 2 for 2D arrays).

    Returns
    -------
    str
        The name of the result variable (a string) that represents the output NumpyBuffer in the generated C code.
    """
    
    #Start your implementation here.
    pass


register_property(
    "neg",
    "__call__",
    "return_type",
    lambda op, *arg_types: arg_types[0]
    if arg_types and isinstance(arg_types[0], NumpyBufferFType)
    else NumpyBufferFType(np.int64, 1),
)


def _c_exp(ctx, arr: apl.Variable, power: apl.Literal):
    """
    Assignment: Implement the _c_exp function to generate C code for element-wise exponentiation of an array.

    Parameters
    ----------
    ctx : APL2CContext
        The code generation context, used to emit C code statements and manage variable names.
    arr : apl.Variable
        The input array, represented as an apl.Variable with type NumpyBufferFType(np.int64, ndim),
        where ndim is the number of dimensions (e.g., 1 for 1D arrays, 2 for 2D arrays).
    power : apl.Literal
        A literal value (e.g., Literal(2)) representing the exponent to which each element is raised.

    Returns
    -------
    str
        The name of the result variable (a string) that represents the output NumpyBuffer in the generated C code.
    """
    
    #Start your implementation here. 
    pass


register_property(
    "exp",
    "__call__",
    "return_type",
    lambda op, *arg_types: arg_types[0]
    if arg_types and isinstance(arg_types[0], NumpyBufferFType)
    else NumpyBufferFType(np.int64, 1),
)


def _c_transpose(ctx, arr: apl.Variable):
    """
    Assignment: Implement the _c_transpose function to generate C code for transposing an array.

    Parameters
    ----------
    ctx : APL2CContext
        The code generation context, used to emit C code statements and manage variable names.
    arr : apl.Variable
        The input array, represented as an apl.Variable with type NumpyBufferFType(np.int64, ndim),
        where ndim is the number of dimensions (e.g., 1 for 1D arrays, 2 for 2D arrays).

    Returns
    -------
    str
        The name of the result variable (a string) that represents the output NumpyBuffer in the generated C code.
    """
    
    #Start your implementation here. 
    pass 
    
register_property(
    "transpose",
    "__call__",
    "return_type",
    lambda op, *arg_types: arg_types[0]
    if arg_types and isinstance(arg_types[0], NumpyBufferFType)
    else NumpyBufferFType(np.int64, 1),
)


def _c_iota(ctx, range: apl.Literal):
    """
    Assignment: Implement the _c_iota function to generate C code for creating an array of consecutive integers.

    Parameters
    ----------
    ctx : APL2CContext
        The code generation context, used to emit C code statements and manage variable names.
    range : apl.Literal
        A literal value (e.g., Literal(5)) representing the length of the output 1D array.

    Returns
    -------
    str
        The name of the result variable (a string) that represents the output NumpyBuffer in the generated C code.
    """
    
    #Start your implementation here. 
    pass


register_property(
    "iota",
    "__call__",
    "return_type",
    lambda op, *arg_types: arg_types[0]
    if arg_types and isinstance(arg_types[0], NumpyBufferFType)
    else NumpyBufferFType(np.int64, 1),
)


def _c_reduce(ctx, buf: apl.Variable):
    
    """
    Assignment: Implement the _c_reduce function to generate C code for summing elements along the last dimension.

    Parameters
    ----------
    ctx : APL2CContext
        The code generation context, used to emit C code statements and manage variable names.
    buf : apl.Variable
        The input array, represented as an apl.Variable with type NumpyBufferFType(np.int64, ndim),
        where ndim is the number of dimensions (e.g., 1 for 1D arrays, 2 for 2D arrays).

    Returns
    -------
    str
        The name of the result variable (a string) that represents the output NumpyBuffer in the generated C code.
    """
    
    #Start your implementation here.
    pass


register_property(
    "reduce",
    "__call__",
    "return_type",
    lambda op, *arg_types: arg_types[0]
    if arg_types and isinstance(arg_types[0], NumpyBufferFType)
    else NumpyBufferFType(np.int64, 1),
)


def _c_reshape(ctx, buf, shape):
    """
    Assignment: Implement the _c_reshape function to generate C code for reshaping an array to a new shape.

    Parameters
    ----------
    ctx : APL2CContext
        The code generation context, used to emit C code statements and manage variable names.
    buf : apl.Variable
        The input array, represented as an apl.Variable with type NumpyBufferFType(np.int64, ndim),
        where ndim is the number of dimensions (e.g., 1 for 1D arrays, 2 for 2D arrays).
    shape : apl.Literal
        A literal value (e.g., Literal((2, 3))) representing the new shape as a tuple or list of integers.

    Returns
    -------
    str
        The name of the result variable (a string) that represents the output NumpyBuffer in the generated C code.

    """
    
    #Start your implementation here. 
    pass

register_property(
    "reshape",
    "__call__",
    "return_type",
    lambda op, *arg_types: arg_types[0]
    if arg_types and isinstance(arg_types[0], NumpyBufferFType)
    else NumpyBufferFType(np.int64, 1),
)

dispatch: dict[str, Callable[..., Any]] = {
    "mkArray": _c_mkArray,
    "add": _c_add,
    "sub": _c_sub,
    "exp": _c_exp,
    "neg": _c_neg,
    "transpose": _c_transpose,
    "reshape": _c_reshape,
    "iota": _c_iota,
    "reduce": _c_reduce,
}


class APL2CContext(CContext):
    """
    A class to represent a C environment.
    """

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.preamble.append("#include <math.h>\n")  # For pow in _c_exp

    def cache(self, name, val):
        if isinstance(val, apl.Literal | apl.Variable):
            return val
        var_n = self.freshen(name)
        var_t = val.result_ftype
        var_t_code = self.ctype_name(c_type(var_t))
        self.exec(f"{self.feed}{var_t_code} {var_n} = {self(val)};")
        return apl.Variable(var_n, var_t)

    def __call__(self, prgm: apl.APLNode):
        feed = self.feed
        """
        lower the program to C code.
        """
        match prgm:
            case apl.Literal(value):
                # in the future, would be nice to be able to pass in constants that
                # are more complex than C literals, maybe as globals.
                return c_literal(self, value)
            case apl.Variable(name, t):
                return name
            case apl.Assign(apl.Variable(var_n, var_t), val):
                val_code = self(val)
                if val.result_ftype != var_t:
                    raise TypeError(f"Type mismatch: {val.result_ftype} != {var_t}")
                if var_n in self.types:
                    assert var_t == self.types[var_n]
                    self.exec(f"{feed}{var_n} = {val_code};")
                else:
                    self.types[var_n] = var_t
                    var_t_code = self.ctype_name(c_type(var_t))
                    self.exec(f"{feed}{var_t_code} {var_n} = {val_code};")
                return None
            case apl.Call(op, args):
                assert isinstance(op, apl.Literal)
                assert isinstance(op.val, str)
                return dispatch[op.val](self, *args)
            case apl.Block(bodies):
                ctx_2 = self.block()
                for body in bodies:
                    ctx_2(body)
                self.exec(ctx_2.emit())
                return None
            case apl.Function(apl.Variable(func_name, return_t), args, body):
                ctx_2 = self.subblock()
                arg_decls = []
                for arg in args:
                    match arg:
                        case apl.Variable(name, t):
                            t_name = self.ctype_name(c_type(t))
                            arg_decls.append(f"{t_name} {name}")
                            ctx_2.types[name] = t
                        case apl.Literal(value):
                            arg_name = ctx_2.freshen("literal")
                            ctx_2.types[arg_name] = np.int64
                            arg_decls.append(f"int64_t {arg_name}")
                        case _:
                            raise NotImplementedError(
                                f"Unrecognized argument type: {arg}"
                            )
                ctx_2(body)
                body_code = ctx_2.emit()
                return_t_name = self.ctype_name(c_type(return_t))
                feed = self.feed
                self.exec(
                    f"{feed}{return_t_name} {func_name}({', '.join(arg_decls)}) {{\n"
                    f"{body_code}\n"
                    f"{feed}}}"
                )
                return None
            case apl.Return(value):
                value = self(value)
                self.exec(f"{feed}return {value};")
                return None
            case apl.Module(funcs):
                for func in funcs:
                    if not isinstance(func, apl.Function):
                        raise NotImplementedError(
                            f"Unrecognized function type: {type(func)}"
                        )
                    self(func)
                return None
            case _:
                raise NotImplementedError(
                    f"Unrecognized assembly node type: {type(prgm)}"
                )
