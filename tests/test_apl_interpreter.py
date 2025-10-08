import pytest

import numpy as np

from apl2c.apl import APLInterpreter
from apl2c.apl import nodes as apl
from apl2c.codegen import NumpyBuffer, NumpyBufferFType


def create_array_literal(input_array):
    shape = map(apl.Literal, input_array.shape)
    flat_array = input_array.flatten()

    return apl.Call(
        op=apl.Literal("reshape"),
        args=(
            apl.Call(
                op=apl.Literal("mkArray"),
                args=tuple(apl.Literal(val) for val in flat_array),
            ),
            *shape,
        ),
    )


@pytest.mark.parametrize(
    "values, expected",
    [
        ([1, 2, 3], np.array([1, 2, 3], dtype=np.int64)),
        ([1, 2], np.array([1, 2], dtype=np.int64)),
    ],
)
def test_mk_array(values, expected):
    # Create variables
    A = apl.Variable("A", NumpyBufferFType(np.int64, 1))

    # Define test function
    test_function = apl.Function(
        apl.Variable("test_mk_array", NumpyBufferFType(np.int64, 1)),
        (),  # no parameters
        apl.Block(
            (
                apl.Assign(
                    lhs=A,
                    rhs=apl.Call(
                        op=apl.Literal("mkArray"),
                        args=tuple([apl.Literal(v) for v in values]),
                    ),
                ),
                apl.Return(A),
            )
        ),
    )

    # Create prgm with the function
    prgm = apl.Module((test_function,))

    mod = APLInterpreter()(prgm)
    result = mod.test_mk_array().arr
    print(result)
    assert np.array_equal(result, expected)
    assert result.dtype == np.int64


@pytest.mark.parametrize(
    "input_array, expected",
    [
        (np.array([1, -2, 3], dtype=np.int64), np.array([-1, 2, -3], dtype=np.int64)),
        (
            np.array([[1, 2], [3, 4]], dtype=np.int64),
            np.array([[-1, -2], [-3, -4]], dtype=np.int64),
        ),
    ],
)
def test_neg(input_array, expected):
    # Create variables
    a_var = apl.Variable("A", NumpyBufferFType(np.int64, input_array.ndim))
    b_var = apl.Variable("B", NumpyBufferFType(np.int64, input_array.ndim))

    # Define test function
    test_function = apl.Function(
        apl.Variable("test_neg", NumpyBufferFType(np.int64, input_array.ndim)),
        (a_var,),
        apl.Block(
            (
                apl.Assign(
                    lhs=b_var,
                    rhs=apl.Call(
                        op=apl.Literal("neg"),
                        args=(a_var,),
                    ),
                ),
                apl.Return(b_var),
            )
        ),
    )

    # Create prgm with the function
    prgm = apl.Module((test_function,))

    mod = APLInterpreter()(prgm)
    result = mod.test_neg(NumpyBuffer(input_array)).arr
    assert np.array_equal(result, expected)
    assert result.dtype == np.int64


@pytest.mark.parametrize(
    "input_array, power, expected",
    [
        (np.array([2, 3], dtype=np.int64), 2, np.array([4, 9], dtype=np.int64)),
        (
            np.array([[1, 2], [3, 4]], dtype=np.int64),
            3,
            np.array([[1, 8], [27, 64]], dtype=np.int64),
        ),
    ],
)
def test_exp(input_array, power, expected):
    # Create variables
    a_var = apl.Variable("A", NumpyBufferFType(np.int64, input_array.ndim))
    b_var = apl.Variable("B", NumpyBufferFType(np.int64, input_array.ndim))

    # Define test function
    test_function = apl.Function(
        apl.Variable("test_exp", NumpyBufferFType(np.int64, input_array.ndim)),
        (a_var,),
        apl.Block(
            (
                apl.Assign(
                    lhs=b_var,
                    rhs=apl.Call(
                        op=apl.Literal("exp"),
                        args=(
                            a_var,
                            apl.Literal(power),
                        ),
                    ),
                ),
                apl.Return(b_var),
            )
        ),
    )

    # Create prgm with the function
    prgm = apl.Module((test_function,))

    mod = APLInterpreter()(prgm)
    result = mod.test_exp(NumpyBuffer(input_array)).arr
    assert np.array_equal(result, expected)
    assert result.dtype == np.int64


@pytest.mark.parametrize(
    "arr1, arr2, expected",
    [
        (
            np.array([1, 2], dtype=np.int64),
            np.array([3, 4], dtype=np.int64),
            np.array([4, 6], dtype=np.int64),
        ),
        (
            np.array([[1, 2], [3, 4]], dtype=np.int64),
            np.array([[5, 6], [7, 8]], dtype=np.int64),
            np.array([[6, 8], [10, 12]], dtype=np.int64),
        ),
    ],
)
def test_add(arr1, arr2, expected):
    # Create variables
    a_var = apl.Variable("A", NumpyBufferFType(np.int64, arr1.ndim))
    b_var = apl.Variable("B", NumpyBufferFType(np.int64, arr1.ndim))
    c_var = apl.Variable("C", NumpyBufferFType(np.int64, arr1.ndim))

    # Define test function
    test_function = apl.Function(
        apl.Variable("test_add", NumpyBufferFType(np.int64, arr1.ndim)),
        (a_var, b_var),
        apl.Block(
            (
                apl.Assign(
                    lhs=c_var,
                    rhs=apl.Call(
                        op=apl.Literal("add"),
                        args=(
                            a_var,
                            b_var,
                        ),
                    ),
                ),
                apl.Return(c_var),
            )
        ),
    )

    # Create prgm with the function
    prgm = apl.Module((test_function,))

    mod = APLInterpreter()(prgm)
    result = mod.test_add(NumpyBuffer(arr1), NumpyBuffer(arr2)).arr
    assert np.array_equal(result, expected)
    assert result.dtype == np.int64


@pytest.mark.parametrize(
    "arr1, arr2, expected",
    [
        (
            np.array([1, 2], dtype=np.int64),
            np.array([3, 4], dtype=np.int64),
            np.array([-2, -2], dtype=np.int64),
        ),
        (
            np.array([[1, 2], [3, 4]], dtype=np.int64),
            np.array([[5, 6], [7, 8]], dtype=np.int64),
            np.array([[-4, -4], [-4, -4]], dtype=np.int64),
        ),
    ],
)
def test_sub(arr1, arr2, expected):
    # Create variables
    a_var = apl.Variable("A", NumpyBufferFType(np.int64, arr1.ndim))
    b_var = apl.Variable("B", NumpyBufferFType(np.int64, arr1.ndim))
    c_var = apl.Variable("C", NumpyBufferFType(np.int64, arr1.ndim))

    # Define test function
    test_function = apl.Function(
        apl.Variable("test_sub", NumpyBufferFType(np.int64, arr1.ndim)),
        (a_var, b_var),
        apl.Block(
            (
                apl.Assign(
                    lhs=c_var,
                    rhs=apl.Call(
                        op=apl.Literal("sub"),
                        args=(
                            a_var,
                            b_var,
                        ),
                    ),
                ),
                apl.Return(c_var),
            )
        ),
    )

    # Create prgm with the function
    prgm = apl.Module((test_function,))

    mod = APLInterpreter()(prgm)
    result = mod.test_sub(NumpyBuffer(arr1), NumpyBuffer(arr2)).arr
    assert np.array_equal(result, expected)
    assert result.dtype == np.int64


@pytest.mark.parametrize(
    "input_array, expected",
    [
        (np.array([1, 2, 3, 4], dtype=np.int64), np.int64(10)),
        (
            np.array([[1, 2], [3, 4]], dtype=np.int64),
            np.array([3, 7], dtype=np.int64),
        ),
    ],
)
def test_reduce(input_array, expected):
    # Create variables
    a_var = apl.Variable("A", NumpyBufferFType(np.int64, input_array.ndim))
    b_var = apl.Variable("B", NumpyBufferFType(np.int64, max(0, input_array.ndim - 1)))

    # Define test function
    test_function = apl.Function(
        apl.Variable(
            "test_reduce", NumpyBufferFType(np.int64, max(0, input_array.ndim - 1))
        ),
        (a_var,),
        apl.Block(
            (
                apl.Assign(
                    lhs=b_var,
                    rhs=apl.Call(
                        op=apl.Literal("reduce"),
                        args=(a_var,),
                    ),
                ),
                apl.Return(b_var),
            )
        ),
    )

    # Create prgm with the function
    prgm = apl.Module((test_function,))

    mod = APLInterpreter()(prgm)
    result = mod.test_reduce(NumpyBuffer(input_array)).arr
    assert np.array_equal(result, expected)
    assert result.dtype == np.int64


@pytest.mark.parametrize(
    "n, expected",
    [
        (5, np.array([1, 2, 3, 4, 5], dtype=np.int64)),
        (0, np.array([], dtype=np.int64)),
    ],
)
def test_iota(n, expected):
    # Create variables
    n_var = apl.Variable("N", int)
    a_var = apl.Variable("A", NumpyBufferFType(np.int64, 1))

    # Define test function
    test_function = apl.Function(
        apl.Variable("test_iota", NumpyBufferFType(np.int64, 1)),
        (n_var,),
        apl.Block(
            (
                apl.Assign(
                    lhs=a_var,
                    rhs=apl.Call(
                        op=apl.Literal("iota"),
                        args=(n_var,),
                    ),
                ),
                apl.Return(a_var),
            )
        ),
    )

    # Create prgm with the function
    prgm = apl.Module((test_function,))

    mod = APLInterpreter()(prgm)
    result = mod.test_iota(n).arr
    assert np.array_equal(result, expected)
    assert result.dtype == np.int64


@pytest.mark.parametrize(
    "shape, input_array, expected",
    [
        (
            np.array([2, 3], dtype=np.int64),
            np.array([1, 2, 3, 4], dtype=np.int64),
            np.array([[1, 2, 3], [4, 1, 2]], dtype=np.int64),
        ),
        (
            np.array([3], dtype=np.int64),
            np.array([1], dtype=np.int64),
            np.array([1, 1, 1], dtype=np.int64),
        ),
    ],
)
def test_reshape(shape, input_array, expected):
    # Create variables
    a_var = apl.Variable("A", NumpyBufferFType(np.int64, input_array.ndim))
    b_var = apl.Variable("B", NumpyBufferFType(np.int64, len(shape)))

    # Define test function
    test_function = apl.Function(
        apl.Variable("test_reshape", NumpyBufferFType(np.int64, len(shape))),
        (a_var,),
        apl.Block(
            (
                apl.Assign(
                    lhs=b_var,
                    rhs=apl.Call(
                        op=apl.Literal("reshape"),
                        args=(
                            a_var,
                            *[apl.Literal(s) for s in shape],
                        ),
                    ),
                ),
                apl.Return(b_var),
            )
        ),
    )

    # Create prgm with the function
    prgm = apl.Module((test_function,))

    mod = APLInterpreter()(prgm)
    result = mod.test_reshape(NumpyBuffer(input_array)).arr
    assert np.array_equal(result, expected)
    assert result.dtype == np.int64


@pytest.mark.parametrize(
    "input_array, expected",
    [
        (
            np.array([[1, 2], [3, 4]], dtype=np.int64),
            np.array([[1, 3], [2, 4]], dtype=np.int64),
        ),
        (
            np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int64),
            np.array([[[1, 5], [3, 7]], [[2, 6], [4, 8]]], dtype=np.int64),
        ),
    ],
)
def test_transpose(input_array, expected):
    # Create variables
    a_var = apl.Variable("A", NumpyBufferFType(np.int64, input_array.ndim))
    b_var = apl.Variable("B", NumpyBufferFType(np.int64, input_array.ndim))

    # Define test function
    test_function = apl.Function(
        apl.Variable("test_transpose", NumpyBufferFType(np.int64, input_array.ndim)),
        (a_var,),
        apl.Block(
            (
                apl.Assign(
                    lhs=b_var,
                    rhs=apl.Call(
                        op=apl.Literal("transpose"),
                        args=(a_var,),
                    ),
                ),
                apl.Return(b_var),
            )
        ),
    )

    # Create prgm with the function
    prgm = apl.Module((test_function,))

    mod = APLInterpreter()(prgm)
    result = mod.test_transpose(NumpyBuffer(input_array)).arr
    assert np.array_equal(result, expected)
    assert result.dtype == np.int64
