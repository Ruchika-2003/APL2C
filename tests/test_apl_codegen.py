import pytest

import numpy as np

import apl2c.apl as apl
from apl2c.apl import (
    APL2CCompiler,
    APLInterpreter,
)
from apl2c.codegen import NumpyBuffer, NumpyBufferFType


@pytest.mark.parametrize(
    "values, expected",
    [
        ([1, 2, 3], np.array([1, 2, 3], dtype=np.int64)),
        ([1, 2], np.array([1, 2], dtype=np.int64)),
    ],
)
def test_mk_array(values, expected):
    A = apl.Variable("A", NumpyBufferFType(np.int64, 1))

    params = tuple(apl.Variable(f"v{i}", int) for i in range(len(values)))

    test_function = apl.Function(
        apl.Variable("test_mk_array", NumpyBufferFType(np.int64, 1)),
        params,  # no parameters
        apl.Block(
            (
                apl.Assign(
                    lhs=A,
                    rhs=apl.Call(
                        op=apl.Literal("mkArray"),
                        args=params,
                    ),
                ),
                apl.Return(A),
            )
        ),
    )

    prgm = apl.Module((test_function,))

    mod_codegen = APL2CCompiler()(prgm)
    mod_interpreter = APLInterpreter()(prgm)

    expected = mod_interpreter.test_mk_array(*values).arr
    result = mod_codegen.test_mk_array(*values).arr
    assert np.array_equal(result, expected), (
        f"Codegen result {result} does not match interpreter result {expected}"
    )
    assert result.dtype == np.int64

    """
    Tests have been commented out for your convinience. 
    Uncomment tests as you implement the function.
    """

@pytest.mark.skip(reason="Delete this line once ops are implemented")
@pytest.mark.parametrize(
    "op, inputs, expected",
    [
        # add: 1D cases
        (
            "add",
            (np.array([1, 2, 3], dtype=np.int64), np.array([4, 5, 6], dtype=np.int64)),
            np.array([5, 7, 9], dtype=np.int64),
        ),
        (
            "add",
            (np.array([1, 2], dtype=np.int64), np.array([3, 4], dtype=np.int64)),
            np.array([4, 6], dtype=np.int64),
        ),
        # add: 2D cases
        (
            "add",
            (
                np.array([[1, 2], [3, 4]], dtype=np.int64),
                np.array([[5, 6], [7, 8]], dtype=np.int64),
            ),
            np.array([[6, 8], [10, 12]], dtype=np.int64),
        ),
        (
            "add",
            (np.array([[1]], dtype=np.int64), np.array([[9]], dtype=np.int64)),
            np.array([[10]], dtype=np.int64),
        ),
        # sub: 1D cases
        (
            "sub",
            (np.array([5, 7, 9], dtype=np.int64), np.array([1, 2, 3], dtype=np.int64)),
            np.array([4, 5, 6], dtype=np.int64),
        ),
        (
            "sub",
            (np.array([4, 6], dtype=np.int64), np.array([1, 2], dtype=np.int64)),
            np.array([3, 4], dtype=np.int64),
        ),
        # sub: 2D cases
        (
            "sub",
            (
                np.array([[6, 8], [10, 12]], dtype=np.int64),
                np.array([[1, 2], [3, 4]], dtype=np.int64),
            ),
            np.array([[5, 6], [7, 8]], dtype=np.int64),
        ),
        (
            "sub",
            (np.array([[10]], dtype=np.int64), np.array([[1]], dtype=np.int64)),
            np.array([[9]], dtype=np.int64),
        ),
        # neg: 1D cases
        (
            "neg",
            (np.array([1, -2, 3], dtype=np.int64),),
            np.array([-1, 2, -3], dtype=np.int64),
        ),
        (
            "neg",
            (np.array([-4, 5], dtype=np.int64),),
            np.array([4, -5], dtype=np.int64),
        ),
        # neg: 2D cases
        (
            "neg",
            (np.array([[1, -2], [3, -4]], dtype=np.int64),),
            np.array([[-1, 2], [-3, 4]], dtype=np.int64),
        ),
        (
            "neg",
            (np.array([[5]], dtype=np.int64),),
            np.array([[-5]], dtype=np.int64),
        ),
    ],
)
def test_operations(op, inputs, expected):
    ndim = inputs[0].ndim
    if op in ["add", "sub"]:
        a_var = apl.Variable("A", NumpyBufferFType(np.int64, ndim))
        b_var = apl.Variable("B", NumpyBufferFType(np.int64, ndim))
        args = (a_var, b_var)
    elif op == "neg":
        a_var = apl.Variable("A", NumpyBufferFType(np.int64, ndim))
        args = (a_var,)
    result_var = apl.Variable("result", NumpyBufferFType(np.int64, ndim))

    test_function = apl.Function(
        apl.Variable(f"test_{op}", NumpyBufferFType(np.int64, ndim)),
        args,
        apl.Block(
            (
                apl.Assign(
                    lhs=result_var,
                    rhs=apl.Call(
                        op=apl.Literal(op),
                        args=args,
                    ),
                ),
                apl.Return(result_var),
            )
        ),
    )

    prgm = apl.Module((test_function,))

    # Run with APL Interpreter
    mod_interpreter = APLInterpreter()(prgm)
    expected = getattr(mod_interpreter, f"test_{op}")(
        *[NumpyBuffer(x) if isinstance(x, np.ndarray) else x for x in inputs]
    ).arr

    # Run with APL2C Compiler
    mod_codegen = APL2CCompiler()(prgm)
    result = getattr(mod_codegen, f"test_{op}")(
        *[NumpyBuffer(x) if isinstance(x, np.ndarray) else x for x in inputs]
    ).arr

    assert np.array_equal(result, expected), (
        f"Codegen result {result} does not match interpreter result {expected}"
    )
    assert result.dtype == np.int64


@pytest.mark.skip(reason="Delete this line once ops are implemented")
@pytest.mark.parametrize(
    "input_array, power, expected",
    [
        # 1D cases
        (
            np.array([1, 2, 3], dtype=np.int64),
            apl.Literal(2),
            np.array([1, 4, 9], dtype=np.int64),
        ),
        (
            np.array([2, 3], dtype=np.int64),
            apl.Literal(2),
            np.array([4, 9], dtype=np.int64),
        ),
        # 2D cases
        (
            np.array([[1, 2], [3, 4]], dtype=np.int64),
            apl.Literal(2),
            np.array([[1, 4], [9, 16]], dtype=np.int64),
        ),
        (
            np.array([[2]], dtype=np.int64),
            apl.Literal(2),
            np.array([[4]], dtype=np.int64),
        ),
    ],
)
def test_exp(input_array, power, expected):
    # Create variables with dynamic ndim
    ndim = input_array.ndim
    a_var = apl.Variable("A", NumpyBufferFType(np.int64, ndim))
    result_var = apl.Variable("result", NumpyBufferFType(np.int64, ndim))

    # Define test function
    test_function = apl.Function(
        apl.Variable("test_exp", NumpyBufferFType(np.int64, ndim)),
        (a_var, power),
        apl.Block(
            (
                apl.Assign(
                    lhs=result_var,
                    rhs=apl.Call(
                        op=apl.Literal("exp"),
                        args=(a_var, power),
                    ),
                ),
                apl.Return(result_var),
            )
        ),
    )

    prgm = apl.Module((test_function,))

    # Run with APL2C Compiler
    mod_codegen = APL2CCompiler()(prgm)
    result = mod_codegen.test_exp(NumpyBuffer(input_array), power.val).arr

    # Run with APL Interpreter
    mod_interpreter = APLInterpreter()(prgm)
    expected = mod_interpreter.test_exp(NumpyBuffer(input_array), power.val).arr
    assert np.array_equal(result, expected), (
        f"Codegen result {result} does not match interpreter result {expected}"
    )
    assert result.dtype == np.int64

@pytest.mark.skip(reason="Delete this line once ops are implemented")
@pytest.mark.parametrize(
    "input_array, expected",
    [
        # 1D cases (transpose is no-op)
        (
            np.array([1, 2, 3], dtype=np.int64),
            np.array([1, 2, 3], dtype=np.int64),
        ),
        (
            np.array([4, 5], dtype=np.int64),
            np.array([4, 5], dtype=np.int64),
        ),
        # 2D cases
        (
            np.array([[1, 2], [3, 4]], dtype=np.int64),
            np.array([[1, 3], [2, 4]], dtype=np.int64),
        ),
        (
            np.array([[1]], dtype=np.int64),
            np.array([[1]], dtype=np.int64),
        ),
        # Additional 2D case with non-square shape
        (
            np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64),
            np.array([[1, 4], [2, 5], [3, 6]], dtype=np.int64),
        ),
    ],
)
def test_transpose(input_array, expected):
    ndim = input_array.ndim
    a_var = apl.Variable("A", NumpyBufferFType(np.int64, ndim))
    result_var = apl.Variable("result", NumpyBufferFType(np.int64, ndim))

    test_function = apl.Function(
        apl.Variable("test_transpose", NumpyBufferFType(np.int64, ndim)),
        (a_var,),
        apl.Block(
            (
                apl.Assign(
                    lhs=result_var,
                    rhs=apl.Call(
                        op=apl.Literal("transpose"),
                        args=(a_var,),
                    ),
                ),
                apl.Return(result_var),
            )
        ),
    )

    prgm = apl.Module((test_function,))

    # Run with APL2C Compiler
    mod_codegen = APL2CCompiler()(prgm)
    result = mod_codegen.test_transpose(NumpyBuffer(input_array)).arr

    # Run with APL2C Interpreter
    mod_interpreter = APLInterpreter()(prgm)
    expected = mod_interpreter.test_transpose(NumpyBuffer(input_array)).arr

    assert np.array_equal(result, expected), (
        f"Codegen result {result} does not match interpreter result {expected}"
    )
    assert result.dtype == np.int64

@pytest.mark.skip(reason="Delete this line once ops are implemented")
@pytest.mark.parametrize(
    "range_val, expected",
    [
        (
            apl.Literal(3),
            np.array([1, 2, 3], dtype=np.int64),
        ),
        (
            apl.Literal(5),
            np.array([1, 2, 3, 4, 5], dtype=np.int64),
        ),
        (
            apl.Literal(1),
            np.array([1], dtype=np.int64),
        ),
        (
            apl.Literal(0),
            np.array([], dtype=np.int64),
        ),
    ],
)
def test_iota(range_val, expected):
    # Create result variable (always 1D)
    result_var = apl.Variable("result", NumpyBufferFType(np.int64, 1))

    # Define test function
    test_function = apl.Function(
        apl.Variable("test_iota", NumpyBufferFType(np.int64, 1)),
        (range_val,),
        apl.Block(
            (
                apl.Assign(
                    lhs=result_var,
                    rhs=apl.Call(
                        op=apl.Literal("iota"),
                        args=(range_val,),
                    ),
                ),
                apl.Return(result_var),
            )
        ),
    )

    # Create program with the function
    prgm = apl.Module((test_function,))

    # Run with APL2CCompiler
    mod_codegen = APL2CCompiler()(prgm)
    result = mod_codegen.test_iota(range_val.val).arr

    # Run with APLInterpreter
    mod_interpreter = APLInterpreter()(prgm)
    expected = mod_interpreter.test_iota(range_val.val).arr

    # Compare results
    assert np.array_equal(result, expected), (
        f"Codegen result {result} does not match interpreter result {result}"
    )
    assert result.dtype == np.int64
    assert expected.dtype == np.int64

@pytest.mark.skip(reason="Delete this line once ops are implemented")
@pytest.mark.parametrize(
    "input_array, expected",
    [
        # 1D cases (reduce to 0D)
        (
            np.array([1, 2, 3], dtype=np.int64),
            np.array(6, dtype=np.int64),
        ),
        (
            np.array([4, 5], dtype=np.int64),
            np.array(9, dtype=np.int64),
        ),
        # 2D cases (reduce to 1D)
        (
            np.array([[1, 2], [3, 4]], dtype=np.int64),
            np.array([3, 7], dtype=np.int64),
        ),
        (
            np.array([[1]], dtype=np.int64),
            np.array([1], dtype=np.int64),
        ),
        # Additional 2D case with non-square shape
        (
            np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64),
            np.array([6, 15], dtype=np.int64),
        ),
    ],
)
def test_reduce(input_array, expected):
    ndim = input_array.ndim
    a_var = apl.Variable("A", NumpyBufferFType(np.int64, ndim))
    result_ndim = max(0, ndim - 1)
    result_var = apl.Variable("result", NumpyBufferFType(np.int64, result_ndim))

    test_function = apl.Function(
        apl.Variable("test_reduce", NumpyBufferFType(np.int64, result_ndim)),
        (a_var,),
        apl.Block(
            (
                apl.Assign(
                    lhs=result_var,
                    rhs=apl.Call(
                        op=apl.Literal("reduce"),
                        args=(a_var,),
                    ),
                ),
                apl.Return(result_var),
            )
        ),
    )

    prgm = apl.Module((test_function,))

    # Run with APL2CCompiler
    mod_codegen = APL2CCompiler()(prgm)
    result = mod_codegen.test_reduce(NumpyBuffer(input_array)).arr

    # Run with APLInterpreter
    mod_interpreter = APLInterpreter()(prgm)
    expected = mod_interpreter.test_reduce(NumpyBuffer(input_array)).arr

    # Compare results
    assert np.array_equal(result, expected), (
        f"Codegen result {result} does not match interpreter result {expected}"
    )
    assert result.dtype == np.int64
    assert expected.dtype == np.int64

@pytest.mark.skip(reason="Delete this line once ops are implemented")
@pytest.mark.parametrize(
    "shape, input_array, expected",
    [
        (
            [2, 3],
            np.array([1, 2, 3, 4], dtype=np.int64),
            np.array([[1, 2, 3], [4, 1, 2]], dtype=np.int64),
        ),
        (
            [3],
            np.array([1], dtype=np.int64),
            np.array([1, 1, 1], dtype=np.int64),
        ),
    ],
)
def test_reshape(shape, input_array, expected):
    a_var = apl.Variable("A", NumpyBufferFType(np.int64, input_array.ndim))
    b_var = apl.Variable("B", NumpyBufferFType(np.int64, len(shape)))

    params = tuple(apl.Variable(f"shape_{i}", int) for i in range(len(shape)))

    test_function = apl.Function(
        apl.Variable("test_reshape", NumpyBufferFType(np.int64, len(shape))),
        (
            a_var,
            *params,
        ),
        apl.Block(
            (
                apl.Assign(
                    lhs=b_var,
                    rhs=apl.Call(
                        op=apl.Literal("reshape"),
                        args=(
                            a_var,
                            *params,
                        ),
                    ),
                ),
                apl.Return(b_var),
            )
        ),
    )

    prgm = apl.Module((test_function,))

    mod_interpreter = APLInterpreter()(prgm)
    mod_codegen = APL2CCompiler()(prgm)

    expected = mod_interpreter.test_reshape(NumpyBuffer(input_array), *shape).arr
    result = mod_codegen.test_reshape(NumpyBuffer(input_array), *shape).arr
    assert np.array_equal(result, expected), (
        f"Codegen result {result} does not match interpreter result {expected}"
    )
    assert result.dtype == np.int64
