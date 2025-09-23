import _operator  # noqa: F401
import operator
from collections import namedtuple

import pytest

import numpy  # noqa: F401, ICN001
import numpy as np

from apl2c import example_lang as exmpl
from apl2c.codegen import NumpyBuffer
from apl2c.example_lang import (  # noqa: F401
    Assign,
    Block,
    Call,
    ExampleLangInterpreter,
    Function,
    If,
    IfElse,
    Literal,
    Module,
    Return,
    Variable,
)
from apl2c.symbolic import ftype


@pytest.mark.parametrize(
    "a, b",
    [
        (np.array([1, 2, 3], dtype=np.float64), np.array([4, 5, 6], dtype=np.float64)),
        (np.array([0], dtype=np.float64), np.array([7], dtype=np.float64)),
        (
            np.array([1.5, 2.5], dtype=np.float64),
            np.array([3.5, 4.5], dtype=np.float64),
        ),
    ],
)
def test_dot_product(a, b):
    # Simple dot product using numpy for expected result
    c = exmpl.Variable("c", np.float64)
    i = exmpl.Variable("i", np.int64)
    ab = NumpyBuffer(a)
    bb = NumpyBuffer(b)
    ab_v = exmpl.Variable("a", ab.ftype)
    bb_v = exmpl.Variable("b", bb.ftype)

    mod = ExampleLangInterpreter()(
        exmpl.Module(
            (
                exmpl.Function(
                    exmpl.Variable("dot_product", np.float64),
                    (
                        ab_v,
                        bb_v,
                    ),
                    exmpl.Block(
                        (
                            exmpl.Assign(c, exmpl.Literal(np.float64(0.0))),
                            exmpl.ForLoop(
                                i,
                                exmpl.Literal(np.int64(0)),
                                exmpl.Length(ab_v),
                                exmpl.Block(
                                    (
                                        exmpl.Assign(
                                            c,
                                            exmpl.Call(
                                                exmpl.Literal(operator.add),
                                                (
                                                    c,
                                                    exmpl.Call(
                                                        exmpl.Literal(operator.mul),
                                                        (
                                                            exmpl.Load(ab_v, (i,)),
                                                            exmpl.Load(bb_v, (i,)),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    )
                                ),
                            ),
                            exmpl.Return(c),
                        )
                    ),
                ),
            )
        )
    )

    result = mod.dot_product(ab, bb)
    expected = np.dot(a, b)
    assert np.allclose(result, expected)


def test_if_statement():
    var = exmpl.Variable("a", np.int64)
    root = exmpl.Module(
        (
            exmpl.Function(
                exmpl.Variable("if_else", np.int64),
                (),
                exmpl.Block(
                    (
                        exmpl.Assign(var, exmpl.Literal(np.int64(5))),
                        exmpl.If(
                            exmpl.Call(
                                exmpl.Literal(operator.eq),
                                (var, exmpl.Literal(np.int64(5))),
                            ),
                            exmpl.Block(
                                (
                                    exmpl.Assign(
                                        var,
                                        exmpl.Call(
                                            exmpl.Literal(operator.add),
                                            (var, exmpl.Literal(np.int64(10))),
                                        ),
                                    ),
                                )
                            ),
                        ),
                        exmpl.IfElse(
                            exmpl.Call(
                                exmpl.Literal(operator.lt),
                                (var, exmpl.Literal(np.int64(15))),
                            ),
                            exmpl.Block(
                                (
                                    exmpl.Assign(
                                        var,
                                        exmpl.Call(
                                            exmpl.Literal(operator.sub),
                                            (var, exmpl.Literal(np.int64(3))),
                                        ),
                                    ),
                                )
                            ),
                            exmpl.Block(
                                (
                                    exmpl.Assign(
                                        var,
                                        exmpl.Call(
                                            exmpl.Literal(operator.mul),
                                            (var, exmpl.Literal(np.int64(2))),
                                        ),
                                    ),
                                )
                            ),
                        ),
                        exmpl.Return(var),
                    )
                ),
            ),
        )
    )

    mod = ExampleLangInterpreter()(root)

    result = mod.if_else()
    assert result == 30

    assert root == eval(repr(root))


def test_simple_struct():
    Point = namedtuple("Point", ["x", "y"])
    p = Point(np.float64(1.0), np.float64(2.0))
    x = (1, 4)

    p_var = exmpl.Variable("p", ftype(p))
    x_var = exmpl.Variable("x", ftype(x))
    res_var = exmpl.Variable("res", np.float64)
    mod = ExampleLangInterpreter()(
        exmpl.Module(
            (
                exmpl.Function(
                    exmpl.Variable("simple_struct", np.float64),
                    (p_var, x_var),
                    exmpl.Block(
                        (
                            exmpl.Assign(
                                res_var,
                                exmpl.Call(
                                    exmpl.Literal(operator.mul),
                                    (
                                        exmpl.GetAttr(p_var, exmpl.Literal("x")),
                                        exmpl.GetAttr(
                                            x_var, exmpl.Literal("element_0")
                                        ),
                                    ),
                                ),
                            ),
                            exmpl.Assign(
                                res_var,
                                exmpl.Call(
                                    exmpl.Literal(operator.add),
                                    (
                                        res_var,
                                        exmpl.Call(
                                            exmpl.Literal(operator.mul),
                                            (
                                                exmpl.GetAttr(
                                                    p_var, exmpl.Literal("y")
                                                ),
                                                exmpl.GetAttr(
                                                    x_var, exmpl.Literal("element_1")
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                            exmpl.Return(res_var),
                        )
                    ),
                ),
            ),
        )
    )

    result = mod.simple_struct(p, x)
    assert result == 9.0


@pytest.mark.parametrize(
    "a",
    [
        np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64),
        np.array([[0]], dtype=np.float64),
        np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float64),
    ],
)
def test_sum(a):
    # Simple sum using numpy for expected result
    s = exmpl.Variable("s", np.float64)
    i = exmpl.Variable("i", np.int64)
    j = exmpl.Variable("j", np.int64)
    ab = NumpyBuffer(a)
    ab_v = exmpl.Variable("a", ab.ftype)

    mod = ExampleLangInterpreter()(
        exmpl.Module(
            (
                exmpl.Function(
                    exmpl.Variable("sum", np.float64),
                    (ab_v,),
                    exmpl.Block(
                        (
                            exmpl.Assign(s, exmpl.Literal(np.float64(0.0))),
                            exmpl.ForLoop(
                                i,
                                exmpl.Literal(np.int64(0)),
                                exmpl.GetAttr(
                                    exmpl.Shape(ab_v), exmpl.Literal("element_0")
                                ),
                                exmpl.ForLoop(
                                    j,
                                    exmpl.Literal(np.int64(0)),
                                    exmpl.GetAttr(
                                        exmpl.Shape(ab_v), exmpl.Literal("element_1")
                                    ),
                                    exmpl.Assign(
                                        s,
                                        exmpl.Call(
                                            exmpl.Literal(operator.add),
                                            (
                                                s,
                                                exmpl.Load(ab_v, (i, j)),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                            exmpl.Return(s),
                        )
                    ),
                ),
            )
        )
    )

    result = mod.sum(ab)
    expected = np.sum(a)
    assert np.allclose(result, expected)
