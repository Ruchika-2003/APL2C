from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..symbolic import ScopedDict, fisinstance
from . import nodes as exmpl


class ExampleLangInterpreterKernel:
    """
    A kernel for interpreting APL2CExampleLang code.
    This is a simple interpreter that executes the assembly code.
    """

    def __init__(self, ctx, func_n, ret_t):
        self.ctx = ctx
        self.func = exmpl.Variable(func_n, ret_t)

    def __call__(self, *args):
        args_i = (exmpl.Literal(arg) for arg in args)
        return self.ctx(exmpl.Call(self.func, args_i))


class ExampleLangInterpreterModule:
    """
    A class to represent an interpreted module of APL2CExampleLang.
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


class ExampleLangInterpreter:
    """
    An interpreter for APL2CExampleLang.
    """

    def __init__(
        self,
        bindings=None,
        types=None,
        loop_state=None,
        function_state=None,
    ):
        if bindings is None:
            bindings = ScopedDict()
        if types is None:
            types = ScopedDict()
        self.bindings = bindings
        self.types = types
        self.loop_state = loop_state
        self.function_state = function_state

    def scope(
        self,
        bindings=None,
        types=None,
        loop_state=None,
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
        if loop_state is None:
            loop_state = self.loop_state
        if function_state is None:
            function_state = self.function_state
        return ExampleLangInterpreter(
            bindings=bindings,
            types=types,
            loop_state=loop_state,
            function_state=function_state,
        )

    def should_halt(self):
        """
        Check if the interpreter should halt execution.
        This is used to stop execution in loops or when a return
        statement is encountered.
        """
        return (
            self.loop_state
            and self.loop_state.should_halt
            or self.function_state
            and self.function_state.should_halt
        )

    def __call__(self, prgm: exmpl.ExampleLangNode):
        """
        Run the program.
        """
        match prgm:
            case exmpl.Literal(value):
                return value
            case exmpl.Variable(var_n, var_t):
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
            case exmpl.Assign(exmpl.Variable(var_n, var_t), val):
                val_e = self(val)
                if not isinstance(val_e, var_t):
                    raise TypeError(
                        f"Assigned value {val_e} is not of type {var_t} for "
                        f"variable '{var_n}'."
                    )
                self.bindings[var_n] = val_e
                self.types[var_n] = var_t
                return None
            case exmpl.GetAttr(obj, attr):
                obj_e = self(obj)
                attr = attr.val
                return obj.result_ftype.struct_getattr(obj_e, attr)
            case exmpl.SetAttr(obj, attr, val):
                obj_e = self(obj)
                attr = attr.val
                val_e = self(val)
                obj.result_ftype.struct_setattr(obj_e, attr, val_e)
                return None
            case exmpl.Call(f, args):
                f_e = self(f)
                args_e = [self(arg) for arg in args]
                return f_e(*args_e)
            case exmpl.Load(buf, indices):
                buf_e = self(buf)
                indices_e = [self(idx) for idx in indices]
                return buf_e.load(indices_e)
            case exmpl.Store(buf, indices, val):
                buf_e = self(buf)
                indices_e = [self(idx) for idx in indices]
                val_e = self(val)
                buf_e.store(indices_e, val_e)
                return None
            case exmpl.Length(buf):
                buf_e = self(buf)
                return buf_e.length()
            case exmpl.Shape(buf):
                buf_e = self(buf)
                return buf_e.shape
            case exmpl.Block(bodies):
                for body in bodies:
                    if self.should_halt():
                        break
                    self(body)
                return None
            case exmpl.ForLoop(exmpl.Variable(var_n, var_t) as var, start, end, body):
                start_e = self(start)
                end_e = self(end)
                if not isinstance(start_e, var_t):
                    raise TypeError(
                        f"Start value {start_e} is not of type {var_t} for "
                        f"variable '{var_n}'."
                    )
                ctx_2 = self.scope(loop_state=HaltState())
                var_e = start_e
                while var_e < end_e:
                    if ctx_2.should_halt():
                        break
                    ctx_3 = self.scope()
                    ctx_3(exmpl.Block((exmpl.Assign(var, exmpl.Literal(var_e)), body)))
                    var_e = type(var_e)(var_e + 1)  # type: ignore[call-arg,operator]
                return None
            case exmpl.WhileLoop(cond, body):
                ctx_2 = self.scope(loop_state=HaltState())
                while self(cond):
                    ctx_3 = ctx_2.scope()
                    if ctx_3.should_halt():
                        break
                    ctx_3(body)
                return None
            case exmpl.If(cond, body):
                if self(cond):
                    ctx_2 = self.scope()
                    ctx_2(body)
                return None
            case exmpl.IfElse(cond, body, else_body):
                if not self(cond):
                    body = else_body
                ctx_2 = self.scope()
                ctx_2(body)
                return None
            case exmpl.Function(exmpl.Variable(func_n, ret_t), args, body):

                def my_func(*args_e):
                    ctx_2 = self.scope(function_state=HaltState())
                    if len(args_e) != len(args):
                        raise ValueError(
                            f"Function '{func_n}' expects {len(args)} arguments, "
                            f"but got {len(args_e)}."
                        )
                    for arg, arg_e in zip(args, args_e, strict=False):
                        match arg:
                            case exmpl.Variable(arg_n, arg_t):
                                if not fisinstance(arg_e, arg_t):
                                    raise TypeError(
                                        f"Argument '{arg_n}' is expected to be of type "
                                        f"{arg_t}, but got {type(arg_e)}."
                                    )
                                ctx_2.bindings[arg_n] = arg_e
                            case _:
                                raise NotImplementedError(
                                    f"Unrecognized argument type: {arg}"
                                )
                    ctx_2(body)
                    if ctx_2.function_state.should_halt:
                        ret_e = ctx_2.function_state.return_value
                        if not isinstance(ret_e, ret_t):
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
            case exmpl.Return(value):
                self.function_state.return_value = self(value)
                self.function_state.should_halt = True
                return None
            case exmpl.Break():
                self.loop_state.should_halt = True
                return None
            case exmpl.Module(funcs):
                for func in funcs:
                    self(func)
                kernels = {}
                for func in funcs:
                    match func:
                        case exmpl.Function(exmpl.Variable(func_n, ret_t), args, _):
                            kernel = ExampleLangInterpreterKernel(self, func_n, ret_t)
                            kernels[func_n] = kernel
                        case _:
                            raise NotImplementedError(
                                f"Unrecognized function definition: {func}"
                            )
                return ExampleLangInterpreterModule(self, kernels)
            case exmpl.Stack(val):
                raise NotImplementedError(
                    "ExampleLangInterpreter does not support symbolic, no target lang"
                )
            case _:
                raise NotImplementedError(
                    f"Unrecognized assembly node type: {type(prgm)}"
                )
