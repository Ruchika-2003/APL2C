import logging

from ..codegen import (
    CContext,
    CKernel,
    CModule,
    c_function_call,
    c_getattr,
    c_literal,
    c_setattr,
    c_type,
    init_shared_lib,
)
from ..symbolic import fisinstance
from ..util import config
from . import nodes as exmpl

logger = logging.getLogger(__name__)


class Example2CGenerator:
    def __call__(self, prgm: exmpl.ExampleLangNode):
        ctx = Example2CContext()
        ctx(prgm)
        return ctx.emit_file()


class Example2CCompiler:
    """
    A class to compile and run APL2CExampleLang.
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
        ctx = Example2CContext()
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
        if prgm.head() != exmpl.Module:
            raise ValueError(
                "Example2CCompiler expects a Module as the head of the program, "
                f"got {type(prgm.head())}"
            )
        for func in prgm.funcs:
            match func:
                case exmpl.Function(exmpl.Variable(func_name, return_t), args, _):
                    arg_ts = [arg.result_ftype for arg in args]
                    kern = CKernel(getattr(lib, func_name), return_t, arg_ts)
                    kernels[func_name] = kern
                case _:
                    raise NotImplementedError(
                        f"Unrecognized function type: {type(func)}"
                    )
        return CModule(lib, kernels)


class Example2CContext(CContext):
    """
    A class to represent a C environment.
    """

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

    def cache(self, name, val):
        if isinstance(val, exmpl.Literal | exmpl.Variable):
            return val
        var_n = self.freshen(name)
        var_t = val.result_ftype
        var_t_code = self.ctype_name(c_type(var_t))
        self.exec(f"{self.feed}{var_t_code} {var_n} = {self(val)};")
        return exmpl.Variable(var_n, var_t)

    def __call__(self, prgm: exmpl.ExampleLangNode):
        feed = self.feed
        """
        lower the program to C code.
        """
        match prgm:
            case exmpl.Literal(value):
                # in the future, would be nice to be able to pass in constants that
                # are more complex than C literals, maybe as globals.
                return c_literal(self, value)
            case exmpl.Variable(name, t):
                return name
            case exmpl.Assign(exmpl.Variable(var_n, var_t), val):
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
            case exmpl.GetAttr(obj, attr):
                if not obj.result_ftype.struct_hasattr(attr.val):
                    raise ValueError("trying to get missing attr")
                return c_getattr(obj.result_ftype, self, self(obj), attr.val)
            case exmpl.SetAttr(obj, attr, val):
                obj = self.cache("obj", obj)
                if not fisinstance(val, obj.result_ftype.struct_attrtype(attr.val)):
                    raise TypeError(
                        f"Type mismatch: {val.result_ftype} != "
                        f"{obj.result_ftype.struct_attrtype(attr.val)}"
                    )
                val_code = self(val)
                c_setattr(obj.result_ftype, self, self(obj), attr.val, val_code)
                return None
            case exmpl.Call(f, args):
                assert isinstance(f, exmpl.Literal)
                return c_function_call(f.val, self, *args)
            case exmpl.Load(buf, indices):
                buf = self.cache("buf", buf)
                return buf.result_ftype.c_load(self, buf, indices)
            case exmpl.Store(buf, indices, val):
                buf = self.cache("buf", buf)
                return buf.result_ftype.c_store(self, buf, indices, val)
            case exmpl.Length(buf):
                buf = self.cache("buf", buf)
                return buf.result_ftype.c_length(self, buf)
            case exmpl.Shape(buf):
                buf = self.cache("buf", buf)
                return buf.result_ftype.c_shape(self, buf)
            case exmpl.Block(bodies):
                ctx_2 = self.block()
                for body in bodies:
                    ctx_2(body)
                self.exec(ctx_2.emit())
                return None
            case exmpl.ForLoop(var, start, end, body):
                var_t = self.ctype_name(c_type(var.result_ftype))
                var_2 = self(var)
                start = self(start)
                end = self(end)
                ctx_2 = self.subblock()
                ctx_2(body)
                ctx_2.types[var.name] = var.result_ftype
                body_code = ctx_2.emit()
                self.exec(
                    f"{feed}for ({var_t} {var_2} = {start}; "
                    f"{var_2} < {end}; {var_2}++) {{\n"
                    f"{body_code}"
                    f"\n{feed}}}"
                )
                return None
            case exmpl.WhileLoop(cond, body):
                if not isinstance(cond, exmpl.Literal | exmpl.Variable):
                    cond_var = exmpl.Variable(self.freshen("cond"), cond.result_ftype)
                    new_prgm = exmpl.Block(
                        (
                            exmpl.Assign(cond_var, cond),
                            exmpl.WhileLoop(
                                cond_var,
                                exmpl.Block(
                                    (
                                        body,
                                        exmpl.Assign(cond_var, cond),
                                    )
                                ),
                            ),
                        )
                    )
                    return self(new_prgm)
                cond_code = self(cond)
                ctx_2 = self.subblock()
                ctx_2(body)
                body_code = ctx_2.emit()
                self.exec(f"{feed}while ({cond_code}) {{\n{body_code}\n{feed}}}")
                return None
            case exmpl.If(cond, body):
                cond_code = self(cond)
                ctx_2 = self.subblock()
                ctx_2(body)
                body_code = ctx_2.emit()
                self.exec(f"{feed}if ({cond_code}) {{\n{body_code}\n{feed}}}")
                return None
            case exmpl.IfElse(cond, body, else_body):
                cond_code = self(cond)
                ctx_2 = self.subblock()
                ctx_2(body)
                body_code = ctx_2.emit()
                ctx_3 = self.subblock()
                ctx_3(else_body)
                else_body_code = ctx_3.emit()
                self.exec(
                    f"{feed}if ({cond_code}) {{\n{body_code}\n{feed}}} "
                    f"else {{\n{else_body_code}\n{feed}}}"
                )
                return None
            case exmpl.Function(exmpl.Variable(func_name, return_t), args, body):
                ctx_2 = self.subblock()
                arg_decls = []
                for arg in args:
                    match arg:
                        case exmpl.Variable(name, t):
                            t_name = self.ctype_name(c_type(t))
                            arg_decls.append(f"{t_name} {name}")
                            ctx_2.types[name] = t
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
            case exmpl.Return(value):
                value = self(value)
                self.exec(f"{feed}return {value};")
                return None
            case exmpl.Break():
                self.exec(f"{feed}break;")
                return None
            case exmpl.Module(funcs):
                for func in funcs:
                    if not isinstance(func, exmpl.Function):
                        raise NotImplementedError(
                            f"Unrecognized function type: {type(func)}"
                        )
                    self(func)
                return None
            case _:
                raise NotImplementedError(
                    f"Unrecognized assembly node type: {type(prgm)}"
                )
