from abc import abstractmethod
from dataclasses import asdict, dataclass
from typing import Any

from apl2c.symbolic import ftype

from ..algebra import return_type
from ..symbolic import Term, TermTree, literal_repr
from ..util import qual_str


class APLNode(Term):
    """
    APLNode

    Represents a APL2CAPL IR node. APL2CAPL is the final intermediate
    representation before code generation (translation to the output language).
    It is a low-level imperative description of the program, with control flow,
    linear memory regions called "buffers", and explicit memory management.
    """

    @classmethod
    def head(cls):
        """Returns the head of the node."""
        return cls

    @classmethod
    def make_term(cls, head, *children):
        """Creates a term with the given head and arguments."""
        return head.from_children(*children)

    @classmethod
    def from_children(cls, *children):
        """
        Creates a term from the given children. This is used to create terms
        from the children of a node.
        """
        return cls(*children)

    def __str__(self):
        """Returns a string representation of the node."""
        ctx = APL2CPrinterContext()
        ctx(self)
        return ctx.emit()


class APLTree(APLNode, TermTree):
    @property
    def children(self):
        """Returns the children of the node."""
        raise Exception(f"`children isn't supported for {self.__class__}")


class APLExpression(APLNode):
    """Base class for expression nodes, which produce values (arrays or scalars)."""

    @property
    @abstractmethod
    def result_ftype(self) -> Any:
        """Returns the type of the expression's result."""
        ...


@dataclass(eq=True, frozen=True)
class Literal(APLExpression):
    """Represents a literal value (e.g., 5 in iota(5))."""

    val: Any

    @property
    def children(self) -> list[APLNode]:
        return []

    @property
    def result_ftype(self) -> Any:
        """Returns the type of the expression."""
        return ftype(self.val)

    def __repr__(self) -> str:
        return literal_repr(type(self).__name__, asdict(self))


@dataclass(eq=True, frozen=True)
class Variable(APLExpression):
    """
    Represents a logical AST expression for a variable named `name`, which
    will hold a value of type `type`.

    Attributes:
        name: The name of the variable.
        type: The type of the variable.
    """

    name: str
    type: Any

    @property
    def result_ftype(self):
        """Returns the type of the expression."""
        return self.type

    def __repr__(self) -> str:
        return literal_repr(type(self).__name__, asdict(self))


@dataclass(eq=True, frozen=True)
class Assign(APLTree):
    """
    Represents a logical AST statement that evaluates `rhs`, binding the result
    to `lhs`.

    Attributes:
        lhs: The left-hand side of the binding.
        rhs: The right-hand side to evaluate.
    """

    lhs: Variable
    rhs: APLExpression

    @property
    def children(self):
        """Returns the children of the node."""
        return [self.lhs, self.rhs]


@dataclass(eq=True, frozen=True)
class Call(APLExpression, APLTree):
    """
    Represents an expression for calling the function `op` on `args...`.

    Attributes:
        op: The function to call.
        args: The arguments to call on the function.
    """

    op: Literal
    args: tuple[APLNode, ...]

    @property
    def children(self):
        """Returns the children of the node."""
        return [self.op, *self.args]

    @classmethod
    def from_children(cls, op, *args):
        return cls(op, args)

    @property
    def result_ftype(self):
        """Returns the type of the expression."""
        arg_types = [arg.result_ftype for arg in self.args]
        # Placeholder: replace with actual return_type implementation
        return return_type(
            self.op.val, *arg_types
        )  # FIXME: add operator def in register_property.


@dataclass(eq=True, frozen=True)
class Function(APLTree):
    """
    Represents a logical AST statement that defines a function `fun` on the
    arguments `args...`.

    Attributes:
        name: The name of the function to define as a variable typed with the
            return type of this function.
        args: The arguments to the function.
        body: The body of the function. If it does not contain a return statement,
            the function returns the value of `body`.
    """

    name: Variable
    args: tuple[Variable, ...]
    body: APLNode

    @property
    def children(self):
        """Returns the children of the node."""
        return [self.name, *self.args, self.body]

    @classmethod
    def from_children(cls, name, *args, body):
        """Creates a term with the given head and arguments."""
        return cls(name, args, body)


@dataclass(eq=True, frozen=True)
class Return(APLTree):
    """
    Represents a return statement that returns `arg` from the current function.
    Halts execution of the function body.

    Attributes:
        arg: The argument to return.
    """

    arg: APLExpression

    @property
    def children(self):
        """Returns the children of the node."""
        return [self.arg]


@dataclass(eq=True, frozen=True)
class Block(APLTree):
    """
    Represents a statement that executes a sequence of statements `bodies...`.

    Attributes:
        bodies: The sequence of statements to execute.
    """

    bodies: tuple[APLNode, ...] = ()

    @property
    def children(self):
        """Returns the children of the node."""
        return [*self.bodies]

    @classmethod
    def from_children(cls, *bodies):
        return cls(bodies)


@dataclass(eq=True, frozen=True)
class Module(APLTree):
    """
    Represents a group of functions. This is the toplevel translation unit for
    APL2CAPL.

    Attributes:
        funcs: The functions defined in the module.
    """

    funcs: tuple[APLNode, ...]

    @property
    def children(self):
        """Returns the children of the node."""
        return [*self.funcs]

    @classmethod
    def from_children(cls, *funcs):
        return cls(funcs)


class APL2CPrinterContext:
    def __init__(self, tab="    ", indent=0):
        self.preamble = []
        self.epilogue = []
        self.tab = tab
        self.indent = indent

    @property
    def feed(self) -> str:
        return self.tab * self.indent

    def exec(self, line: str):
        self.preamble.append(line)

    def emit(self):
        return "\n".join([*self.preamble, *self.epilogue])

    def block(self) -> "APL2CPrinterContext":
        blk = APL2CPrinterContext()
        blk.indent = self.indent
        blk.tab = self.tab
        return blk

    def subblock(self):
        blk = self.block()
        blk.indent = self.indent + 1
        return blk

    def __call__(self, prgm: APLNode):
        feed = self.feed
        match prgm:
            case Literal(val):
                return str(val)
            case Variable(name, _):
                return str(name)
            case Assign(Variable(var_n, var_t), rhs):
                self.exec(f"{feed}{var_n}: {qual_str(var_t)} = {self(rhs)};")
                return None
            case Call(op, args):
                args_str = ", ".join(self(arg) for arg in args)
                return f"{self(op)}({args_str})"
            case Block(bodies):
                ctx_2 = self.block()
                for body in bodies:
                    ctx_2(body)
                self.exec(ctx_2.emit())
                return None
            case Function(Variable(func_name, return_t), args, body):
                ctx_2 = self.subblock()
                arg_decls = []
                for arg in args:
                    match arg:
                        case Variable(name, t):
                            arg_decls.append(f"{name}: {qual_str(t)}")
                        case _:
                            raise NotImplementedError(
                                f"Unrecognized argument type: {arg}"
                            )
                ctx_2(body)
                body_code = ctx_2.emit()
                feed = self.feed
                self.exec(
                    f"{feed}def {func_name}({', '.join(arg_decls)}) -> "
                    f"{qual_str(return_t)}:\n"
                    f"{body_code}\n"
                )
                return None
            case Return(value):
                self.exec(f"{feed}return {self(value)}")
                return None
            case Module(funcs):
                for func in funcs:
                    if not isinstance(func, Function):
                        raise NotImplementedError(
                            f"Unrecognized function type: {type(func)}"
                        )
                    self(func)
                return None
            case _:
                raise NotImplementedError(f"Unrecognized node type: {type(prgm)}")
