import ctypes
import operator
from collections import namedtuple

import pytest

import numpy as np
from numpy.testing import assert_equal

import apl2c
import apl2c.example_lang as exmpl
from apl2c import ftype
from apl2c.codegen import (
    NumpyBuffer,
)
from apl2c.example_lang import (
    Example2CCompiler,
    Example2CGenerator,
)
from apl2c.example_lang import (
    Example2CCompiler,
    Example2CContext,
    Example2CGenerator,
)


def test_add_function():
    c_code = """
    #include <stdio.h>

    int add(int a, int b) {
        return a + b;
    }
    """
    f = apl2c.codegen.c.load_shared_lib(c_code).add
    result = f(3, 4)
    assert result == 7, f"Expected 7, got {result}"


@pytest.mark.parametrize(
    ["compiler", "buffer"],
    [
        (Example2CCompiler(), NumpyBuffer),
    ],
)
def test_codegen(compiler, buffer):
    a = np.array([1, 2, 3], dtype=np.float64)
    buf = buffer(a)

    a_var = exmpl.Variable("a", buf.ftype)
    i_var = exmpl.Variable("i", np.intp)
    length_var = exmpl.Variable("l", np.intp)
    prgm = exmpl.Module(
        (
            exmpl.Function(
                exmpl.Variable("test_function", np.intp),
                (a_var,),
                exmpl.Block(
                    (
                        exmpl.Assign(length_var, exmpl.Length(a_var)),
                        exmpl.ForLoop(
                            i_var,
                            exmpl.Literal(0),
                            length_var,
                            exmpl.Store(
                                a_var,
                                (i_var,),
                                exmpl.Call(
                                    exmpl.Literal(operator.add),
                                    (exmpl.Load(a_var, (i_var,)), exmpl.Literal(1)),
                                ),
                            ),
                        ),
                        exmpl.Return(exmpl.Literal(0)),
                    )
                ),
            ),
        )
    )
    mod = compiler(prgm)
    f = mod.test_function
    f(buf)
    result = buf.arr
    expected = np.array([2, 3, 4], dtype=np.float64)
    assert_equal(result, expected)


@pytest.mark.parametrize(
    ["compiler", "buffer"],
    [
        (Example2CCompiler(), NumpyBuffer),
        (exmpl.ExampleLangInterpreter(), NumpyBuffer),
    ],
)
def test_dot_product(compiler, buffer):
    a = np.array([1, 2, 3], dtype=np.float64)
    b = np.array([4, 5, 6], dtype=np.float64)

    a_buf = buffer(a)
    b_buf = buffer(b)

    c = exmpl.Variable("c", np.float64)
    i = exmpl.Variable("i", np.int64)
    ab = buffer(a)
    bb = buffer(b)
    ab_v = exmpl.Variable("a", ab.ftype)
    bb_v = exmpl.Variable("b", bb.ftype)
    prgm = exmpl.Module(
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

    mod = compiler(prgm)

    result = mod.dot_product(a_buf, b_buf)

    interp = exmpl.ExampleLangInterpreter()(prgm)

    expected = interp.dot_product(a_buf, b_buf)

    assert np.isclose(result, expected), f"Expected {expected}, got {result}"


@pytest.mark.parametrize(
    ["compiler", "extension", "buffer"],
    [
        (Example2CGenerator(), ".c", NumpyBuffer),
    ],
)
def test_dot_product_regression(compiler, extension, buffer, file_regression):
    a = np.array([1, 2, 3], dtype=np.float64)
    b = np.array([4, 5, 6], dtype=np.float64)

    c = exmpl.Variable("c", np.float64)
    i = exmpl.Variable("i", np.int64)
    ab = buffer(a)
    bb = buffer(b)
    ab_v = exmpl.Variable("a", ab.ftype)
    bb_v = exmpl.Variable("b", bb.ftype)
    prgm = exmpl.Module(
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

    file_regression.check(compiler(prgm), extension=extension)


@pytest.mark.parametrize(
    ["compiler", "buffer"],
    [
        (Example2CCompiler(), NumpyBuffer),
        (exmpl.ExampleLangInterpreter(), NumpyBuffer),
    ],
)
def test_if_statement(compiler, buffer):
    var = exmpl.Variable("a", np.int64)
    prgm = exmpl.Module(
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

    mod = compiler(prgm)

    result = mod.if_else()

    interp = exmpl.ExampleLangInterpreter()(prgm)

    expected = interp.if_else()

    assert np.isclose(result, expected), f"Expected {expected}, got {result}"


@pytest.mark.parametrize(
    "compiler",
    [
        Example2CCompiler(),
    ],
)
def test_simple_struct(compiler):
    Point = namedtuple("Point", ["x", "y"])
    p = Point(np.float64(1.0), np.float64(2.0))
    x = (1, 4)

    p_var = exmpl.Variable("p", ftype(p))
    x_var = exmpl.Variable("x", ftype(x))
    res_var = exmpl.Variable("res", np.float64)
    mod = compiler(
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
    assert result == np.float64(9.0)


@pytest.mark.parametrize(
    "compiler",
    [
        Example2CCompiler(),
    ],
)
def test_basic_tuple_pass_through(compiler):
    """Test that a tuple can be passed to a function and returned unchanged."""
    x = (1, 2)

    x_var = exmpl.Variable("x", ftype(x))
    mod = compiler(
        exmpl.Module(
            (
                exmpl.Function(
                    exmpl.Variable("identity_tuple", ftype(x)),
                    (x_var,),
                    exmpl.Block((exmpl.Return(x_var),)),
                ),
            ),
        )
    )

    result = mod.identity_tuple(x)
    assert result == x


@pytest.mark.parametrize(
    "compiler",
    [
        Example2CCompiler(),
    ],
)
def test_tuple_element_access(compiler):
    """Test accessing individual elements of a tuple."""
    x = (10, 20, 30)

    x_var = exmpl.Variable("x", ftype(x))
    mod = compiler(
        exmpl.Module(
            (
                exmpl.Function(
                    exmpl.Variable("get_first_element", int),
                    (x_var,),
                    exmpl.Block(
                        (
                            exmpl.Return(
                                exmpl.GetAttr(x_var, exmpl.Literal("element_0"))
                            ),
                        )
                    ),
                ),
            ),
        )
    )

    result = mod.get_first_element(x)
    assert result == x[0]


@pytest.mark.parametrize(
    "compiler",
    [
        Example2CCompiler(),
    ],
)
def test_tuple_mixed_types(compiler):
    """Test tuples containing different data types."""
    x = (1, 2.5)  # int and float

    x_var = exmpl.Variable("x", ftype(x))
    result_var = exmpl.Variable("result", float)

    mod = compiler(
        exmpl.Module(
            (
                exmpl.Function(
                    exmpl.Variable("sum_mixed_tuple", float),
                    (x_var,),
                    exmpl.Block(
                        (
                            exmpl.Assign(
                                result_var,
                                exmpl.Call(
                                    exmpl.Literal(operator.add),
                                    (
                                        exmpl.GetAttr(
                                            x_var, exmpl.Literal("element_0")
                                        ),
                                        exmpl.GetAttr(
                                            x_var, exmpl.Literal("element_1")
                                        ),
                                    ),
                                ),
                            ),
                            exmpl.Return(result_var),
                        )
                    ),
                ),
            ),
        )
    )

    result = mod.sum_mixed_tuple(x)
    expected = float(x[0] + x[1])
    assert abs(result - expected) < 1e-10


@pytest.mark.parametrize(
    "compiler",
    [
        Example2CCompiler(),
    ],
)
def test_tuple_arithmetic(compiler):
    """Test performing arithmetic operations with tuple elements."""
    x = (5, 3)
    y = (2, 4)

    x_var = exmpl.Variable("x", ftype(x))
    y_var = exmpl.Variable("y", ftype(y))
    result_var = exmpl.Variable("result", int)

    mod = compiler(
        exmpl.Module(
            (
                exmpl.Function(
                    exmpl.Variable("tuple_dot_product", int),
                    (x_var, y_var),
                    exmpl.Block(
                        (
                            exmpl.Assign(
                                result_var,
                                exmpl.Call(
                                    exmpl.Literal(operator.add),
                                    (
                                        exmpl.Call(
                                            exmpl.Literal(operator.mul),
                                            (
                                                exmpl.GetAttr(
                                                    x_var, exmpl.Literal("element_0")
                                                ),
                                                exmpl.GetAttr(
                                                    y_var, exmpl.Literal("element_0")
                                                ),
                                            ),
                                        ),
                                        exmpl.Call(
                                            exmpl.Literal(operator.mul),
                                            (
                                                exmpl.GetAttr(
                                                    x_var, exmpl.Literal("element_1")
                                                ),
                                                exmpl.GetAttr(
                                                    y_var, exmpl.Literal("element_1")
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                            exmpl.Return(result_var),
                        )
                    ),
                ),
            ),
        )
    )

    result = mod.tuple_dot_product(x, y)
    expected = x[0] * y[0] + x[1] * y[1]
    assert result == expected


@pytest.mark.parametrize(
    "compiler",
    [
        Example2CCompiler(),
    ],
)
def test_nested_tuples(compiler):
    """Test tuples containing other tuples."""
    inner = (1, 2)
    x = (inner, 3)

    x_var = exmpl.Variable("x", ftype(x))
    result_var = exmpl.Variable("result", int)

    mod = compiler(
        exmpl.Module(
            (
                exmpl.Function(
                    exmpl.Variable("nested_tuple_access", int),
                    (x_var,),
                    exmpl.Block(
                        (
                            exmpl.Assign(
                                result_var,
                                exmpl.Call(
                                    exmpl.Literal(operator.add),
                                    (
                                        exmpl.GetAttr(
                                            exmpl.GetAttr(
                                                x_var, exmpl.Literal("element_0")
                                            ),
                                            exmpl.Literal("element_0"),
                                        ),
                                        exmpl.GetAttr(
                                            x_var, exmpl.Literal("element_1")
                                        ),
                                    ),
                                ),
                            ),
                            exmpl.Return(result_var),
                        )
                    ),
                ),
            ),
        )
    )

    result = mod.nested_tuple_access(x)
    expected = x[0][0] + x[1]
    assert result == expected
