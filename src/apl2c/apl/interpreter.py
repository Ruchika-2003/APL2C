from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from apl2c.codegen import NumpyBuffer, NumpyBufferFType
from apl2c.symbolic import ftype

from ..symbolic import ScopedDict, fisinstance
from . import nodes as apl


class APLInterpreterKernel:
    """
    A kernel for interpreting APL code.
    This is a simple interpreter that executes the assembly code.
    """

    def __init__(self, ctx, func_n, ret_t):
        self.ctx = ctx
        self.func = ctx.bindings[func_n]

    def __call__(self, *args):
        args_i = (arg for arg in args)
        return self.func(*args_i)


class APLInterpreterModule:
    """
    A class to represent an interpreted module of APL.
    """

    def __init__(self, ctx, kernels):
        self.ctx = ctx
        self.kernels = kernels

    def __getattr__(self, name):
        # Allow attribute access to kernels by name
        if name in self.kernels:
            return self.kernels[name]
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}"
        )


@dataclass(eq=True)
class HaltState:
    """
    A class to represent the halt state of an assembly program.
    This is used to indicate whether we should break or return, and
    what the return value is if applicable.
    """

    should_halt: bool = False
    return_value: Any = None


def _mk_array(ctx, *args):
    """
    DO NOT CHANGE. REFERENCE IMPLEMENTATION.

    Create a 1-dimensional array from a variable number of values.

    Parameters
    ----------
    ctx : Any
        The interpreter context (not used in this function).
    *args : tuple
        Variable number of scalar values (integers, floats, or
        other numeric types) to form the array elements.

    Returns
    -------
    NumpyBuffer
        A 1-dimensional NumpyBuffer wrapping an np.ndarray
        with dtype np.int64 containing the input values. If no arguments are
        provided, returns an empty array.
    """
    values = args

    # Infer dtype from the first argument if possible, otherwise use default
    if len(args) > 0:
        try:
            # Try to get ftype from the first argument
            dtype = ftype(args[0])
        except (ValueError, TypeError, AttributeError):
            # Fall back to int64 as default
            dtype = np.int64
    else:
        dtype = np.int64

    arr = np.array(values, dtype=dtype)
    return NumpyBuffer(arr)


def _neg(ctx, buf):
    """
    DO NOT CHANGE. REFERENCE IMPLEMENTATION.

    Negate each element of the input array.

    Parameters
    ----------
    ctx : Any
        The interpreter context (not used in this function).
    buf : NumpyBuffer
        The input array to negate, wrapped in a NumpyBuffer.

    Returns
    -------
    NumpyBuffer
        A NumpyBuffer wrapping an np.ndarray with the same shape and dtype
        np.int64, containing the negated values of the input array.
    """
    return NumpyBuffer(-buf.arr)


def _exp(ctx, buf, power):
    """
    Raise each element of the input array to the given power.

    Parameters
    ----------
    ctx : Any
        The interpreter context (not used in this function).
    buf : NumpyBuffer
        The input array whose elements will be raised to the given power.
    power : int or float
        The exponent to which each element is raised.

    Returns
    -------
    NumpyBuffer
        A NumpyBuffer wrapping an np.ndarray with the same shape as the input
        and dtype np.int64, containing each element raised to the given power.
    """
    return NumpyBuffer(np.power(buf.arr, power, dtype=np.int64))


def _add(ctx, buf1, buf2):
    """
    Perform elementwise addition of two arrays.

    Parameters
    ----------
    ctx : Any
        The interpreter context (not used in this function).
    buf1 : NumpyBuffer
        The first input array.
    buf2 : NumpyBuffer
        The second input array, with compatible shape for broadcasting.

    Returns
    -------
    NumpyBuffer
        A NumpyBuffer wrapping an np.ndarray with the broadcasted shape and dtype
        np.int64, containing the elementwise sum of the input arrays.
    """
    return NumpyBuffer(np.add(buf1.arr, buf2.arr, dtype=np.int64))


def _sub(ctx, buf1, buf2):
    """
    Perform elementwise subtraction of two arrays.

    Parameters
    ----------
    ctx : Any
        The interpreter context (not used in this function).
    buf1 : NumpyBuffer
        The first input array (minuend).
    buf2 : NumpyBuffer
        The second input array (subtrahend), with compatible shape for broadcasting.

    Returns
    -------
    NumpyBuffer
        A NumpyBuffer wrapping an np.ndarray with the broadcasted shape and dtype
        np.int64, containing the elementwise difference of the input arrays.
    """
    # raise NotImplementedError #students to implement.
    return NumpyBuffer(np.subtract(buf1.arr, buf2.arr, dtype=np.int64))


def _reduce(ctx, buf):
    """
    Sum elements along the last dimension of the input array.

    Parameters
    ----------
    ctx : Any
        The interpreter context (not used in this function).
    buf : NumpyBuffer
        The input array to reduce.

    Returns
    -------
    NumpyBuffer
        A NumpyBuffer wrapping an np.ndarray with one fewer dimension than the
        input (or a scalar wrapped in a 0D array for 1D input) and dtype np.int64,
        containing the sum of elements along the last dimension.
    """
    arr = buf.arr if isinstance(buf, NumpyBuffer) else buf
    if arr.size == 0:
        raise ValueError("Cannot reduce an empty array")
    result = np.sum(arr, axis=-1, dtype=np.int64)
    # Ensure return value is an array
    if np.isscalar(result):
        result = np.array(result, dtype=np.int64)
    return NumpyBuffer(result)


def _iota(ctx, n):
    """
    Generate a 1-dimensional array of integers from 1 to N (inclusive).

    Parameters
    ----------
    ctx : Any
        The interpreter context (not used in this function).
    n : int
        A non-negative integer specifying the length of the output array.

    Returns
    -------
    NumpyBuffer
        A 1-dimensional NumpyBuffer wrapping an np.ndarray with dtype np.int64,
        containing integers [1, 2, ..., n].
    """
    # raise NotImplementedError #students to implement.
    return NumpyBuffer(np.arange(1, n + 1, dtype=np.int64))


def _reshape(ctx, buf, shape):
    """
    Reshape the input array to the given shape, cycling or truncating elements.

    Parameters
    ----------
    ctx : Any
        The interpreter context (not used in this function).
    buf : NumpyBuffer
        The input array to reshape.
    shape : tuple
        A tuple of positive integers specifying the target shape.

    Returns
    -------
    NumpyBuffer
        A NumpyBuffer wrapping an np.ndarray with the specified shape and dtype
        np.int64, containing the input array’s elements cycled or truncated to fit.
    """
    arr = buf.arr

    try:
        int_shape = tuple(int(s) for s in shape)
        if not all(s > 0 for s in int_shape):
            raise ValueError("Shape elements must be positive integers")
    except (ValueError, TypeError) as err:
        raise ValueError("Shape elements must be positive integers") from err

    total_elements = np.prod(int_shape) if int_shape else 1
    if total_elements == 0:
        return NumpyBuffer(np.zeros(int_shape, dtype=np.int64))

    flat_arr = arr.flatten()
    cycled = np.resize(flat_arr, total_elements)
    result = cycled.reshape(int_shape)
    return NumpyBuffer(result)


def _transpose(ctx, buf):
    """
    Transpose the input array by reversing its axis order.

    Parameters
    ----------
    ctx : Any
        The interpreter context (not used in this function).
    buf : NumpyBuffer
        The input array to transpose.

    Returns
    -------
    NumpyBuffer
        A NumpyBuffer wrapping an np.ndarray with the same dtype np.int64 and
        number of dimensions as the input, but with axes reversed (e.g., (1, 0)
        for 2D, (2, 1, 0) for 3D). The output array must be C-contiguous.

    """
    return NumpyBuffer(np.ascontiguousarray(buf.arr.T))


dispatch: dict[str, Callable[..., Any]] = {
    "mkArray": _mk_array,
    "neg": _neg,
    "exp": _exp,
    "add": _add,
    "sub": _sub,
    "reduce": _reduce,
    "iota": _iota,
    "reshape": _reshape,
    "transpose": _transpose,
}


class APLInterpreter:
    """
    An interpreter for APL.
    """

    def __init__(
        self,
        bindings=None,
        types=None,
        function_state=None,
    ):
        if bindings is None:
            bindings = ScopedDict()
        if types is None:
            types = ScopedDict()
        self.bindings = bindings
        self.types = types
        self.function_state = function_state

    def scope(
        self,
        bindings=None,
        types=None,
        function_state=None,
    ):
        """
        Create a new scope for the interpreter.
        This allows for nested scopes and variable shadowing.
        """
        if bindings is None:
            bindings = self.bindings.scope()
        if types is None:
            types = self.types.scope()
        if function_state is None:
            function_state = self.function_state
        return APLInterpreter(
            bindings=bindings,
            types=types,
            function_state=function_state,
        )

    def should_halt(self):
        """
        Check if the interpreter should halt execution.
        This is used to stop execution in loops or when a return
        statement is encountered.
        """
        return self.function_state and self.function_state.should_halt

    def __call__(self, prgm: apl.APLNode):
        """
        Run the program.
        """
        match prgm:
            case apl.Literal(value):
                return value
            case apl.Variable(var_n, var_t):
                if var_n in self.types:
                    def_t = self.types[var_n]
                    if def_t != var_t:
                        raise TypeError(
                            f"Variable '{var_n}' is declared as type {def_t}, "
                            f"but used as type {var_t}."
                        )
                if var_n in self.bindings:
                    return self.bindings[var_n]
                raise KeyError(
                    f"Variable '{var_n}' is not defined in the current context."
                )
            case apl.Assign(apl.Variable(var_n, var_t), val):
                val_e = self(val)
                if isinstance(var_t, NumpyBufferFType):
                    if not isinstance(val_e, NumpyBuffer) or val_e.ftype != var_t:
                        raise TypeError(
                            f"Assigned value {val_e} is not of type {var_t} for "
                            f"variable '{var_n}'."
                        )
                else:
                    if not fisinstance(val_e, var_t):
                        raise TypeError(
                            f"Assigned value {val_e} is not of type {var_t} for "
                            f"variable '{var_n}'."
                        )
                self.bindings[var_n] = val_e
                self.types[var_n] = var_t
                return None
            case apl.Call(op, args):
                if not isinstance(op, apl.Literal):
                    raise TypeError("Function calls must be to literal function names.")
                op_name = op.val
                if not isinstance(op_name, str):
                    raise TypeError(
                        f"Function name must be a string, got {type(op_name).__name__}"
                    )
                if op_name not in dispatch:
                    raise ValueError(f"Unknown function '{op_name}'")
                args_e = [self(arg) for arg in args]
                f_e = dispatch[op_name]
                return f_e(self, *args_e)
            case apl.Block(bodies):
                for body in bodies:
                    if self.should_halt():
                        break
                    self(body)
                return None
            case apl.Function(apl.Variable(func_n, ret_t), args, body):

                def my_func(*args_e):
                    ctx_2 = self.scope(function_state=HaltState())
                    if len(args_e) != len(args):
                        raise ValueError(
                            f"Function '{func_n}' expects {len(args)} arguments, "
                            f"but got {len(args_e)}."
                        )
                    for arg, arg_e in zip(args, args_e, strict=False):
                        match arg:
                            case apl.Variable(arg_n, arg_t):
                                if isinstance(arg_t, NumpyBufferFType):
                                    if (
                                        not isinstance(arg_e, NumpyBuffer)
                                        or arg_e.ftype != arg_t
                                    ):
                                        raise TypeError(
                                            f"Argument '{arg_n}' is expected"
                                            "to be of type "
                                            f"{arg_t}, but got {type(arg_e)}."
                                        )
                                else:
                                    if not fisinstance(arg_e, arg_t):
                                        raise TypeError(
                                            f"Argument '{arg_n}'"
                                            "is expected to be of type "
                                            f"{arg_t}, but got {type(arg_e)}."
                                        )
                                ctx_2.bindings[arg_n] = arg_e
                            case apl.Literal(value):
                                if not isinstance(arg_e, (int, float)):
                                    raise TypeError(
                                        f"Literal argument expected to be numeric, "
                                        f"but got {type(arg_e)}."
                                    )
                                ctx_2.bindings[value] = arg_e
                            case _:
                                raise NotImplementedError(
                                    f"Unrecognized argument type: {arg}"
                                )
                    ctx_2(body)
                    if ctx_2.function_state.should_halt:
                        ret_e = ctx_2.function_state.return_value
                        if isinstance(ret_t, NumpyBufferFType):
                            if (
                                not isinstance(ret_e, NumpyBuffer)
                                or ret_e.ftype != ret_t
                            ):
                                raise TypeError(
                                    f"Return value {ret_e} is not of type {ret_t} "
                                    f"for function '{func_n}'."
                                )
                        else:
                            if not isinstance(ret_e, NumpyBuffer | type(ret_t)):
                                raise TypeError(
                                    f"Return value {ret_e} is not of type {ret_t} "
                                    f"for function '{func_n}'."
                                )
                        return ret_e
                    raise ValueError(
                        f"Function '{func_n}' did not return a value, "
                        f"but expected type {ret_t}."
                    )

                self.bindings[func_n] = my_func
                return None
            case apl.Return(value):
                self.function_state.return_value = self(value)
                self.function_state.should_halt = True
                return None
            case apl.Module(funcs):
                for func in funcs:
                    self(func)
                kernels = {}
                for func in funcs:
                    match func:
                        case apl.Function(apl.Variable(func_n, ret_t), args, _):
                            kernel = APLInterpreterKernel(self, func_n, ret_t)
                            kernels[func_n] = kernel
                        case _:
                            raise NotImplementedError(
                                f"Unrecognized function definition: {func}"
                            )
                return APLInterpreterModule(self, kernels)
            case _:
                raise NotImplementedError(
                    f"Unrecognized assembly node type: {type(prgm)}"
                )
