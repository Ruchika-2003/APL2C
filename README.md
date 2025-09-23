# Assignment: APL2C

A python compiler that translates APL code to C code.

## Installation

APL2C uses [poetry](https://python-poetry.org/) for packaging. To install for
development, clone the repository and run:
```bash
poetry install --extras test
```
to install the current project and dev dependencies.

## Contributing
See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines, development setup, and best practices.


### Overview

APL2C is a compiler that translates [APL](https://xpqz.github.io/learnapl/intro.html), an array-oriented programming language known for its concise syntax and powerful multi-dimensional array operations, into C code for efficient execution. The project comprises two main components:

1. **Interpreter** (`src/apl2c/apl/interpreter.py`):
   - Executes APL operations directly in Python using NumPy for efficient array manipulation.
   - Defines functions such as `_add`, `_sub`, `_neg`, `_exp`, `_transpose`, `_iota`, `_reshape`, and `_reduce`, which implement the semantics of APL operations.
   - Serves as the reference for the expected behavior of each operation which is the __ground truth for your implementations__.
   - You __should not__ modify these functions.

2. **Codegen** (`src/apl2c/apl/codegen.py`):
   - Generates C code for APL operations, producing `NumpyBuffer` structs that interface with NumPy arrays in C.
   - The generated C code is compiled and executed.

**Your task is to implement the codegen functions (`_c_add`, `_c_sub`, `_c_neg`, `_c_exp`, `_c_transpose`, `_c_iota`, `_c_reshape`, `_c_reduce`) in `src/apl2c/apl/codegen.py` to generate C code that matches the behavior of the corresponding interpreter functions.**

### Task Description

Your goal is to implement the following functions in `codegen.py` to generate C code for APL operations:

- `_c_add`: Performs element-wise addition of two arrays (e.g., `[1, 2] + [3, 4] → [4, 6]`).
- `_c_sub`: Performs element-wise subtraction of two arrays (e.g., `[5, 6] - [1, 2] → [4, 4]`).
- `_c_neg`: Performs element-wise negation of an array (e.g., `[1, -2] → [-1, 2]`).
- `_c_exp`: Performs element-wise exponentiation of an array by a scalar power (e.g., `[2, 3]^2 → [4, 9]`).
- `_c_transpose`: Transposes an array by reversing its dimensions (e.g., `[[1, 2], [3, 4]] → [[1, 3], [2, 4]]`).
- `_c_iota`: Creates a 1D array of consecutive integers from 1 to a specified length (e.g., `5 → [1, 2, 3, 4, 5]`).
- `_c_reshape`: Reshapes an array to a new shape while preserving its elements (e.g., `[1, 2, 3, 4] → [[1, 2], [3, 4]]`).
- `_c_reduce`: Sums elements along the last dimension of an array (e.g., `[[1, 2], [3, 4]] → [3, 7]`).

Each function must:
- Use the `APL2CContext` (`ctx`) to emit C code statements that manipulate `NumpyBuffer` structs.
- Allocate output arrays using `c_alloc`, access input elements with `c_load`, and store results with `c_store`.
- Produce results identical to the corresponding interpreter function in `interpreter.py`.

### Key Components

#### Interpreter Reference (`src/apl2c/apl/interpreter.py`)

The interpreter defines the expected behavior of APL operations using NumPy. Key functions include:

- `_add(ctx, arr1, arr2)`: Returns `NumpyBuffer(np.add(arr1.arr, arr2.arr, dtype=np.int64))`.
- `_reshape(ctx, buf, shape)`: Returns `NumpyBuffer(np.reshape(buf.arr, shape, order='C'))`.
- `_reduce(ctx, buf)`: Returns `NumpyBuffer(np.sum(buf.arr, axis=-1, dtype=np.int64))`, wrapping scalars in 1D arrays for 1D inputs.
- Study these functions to understand the input/output behavior your C code must replicate.

#### Codegen Functions (`src/apl2c/apl/codegen.py`)

- **Inputs**: Each function takes an `APL2CContext` (`ctx`) and arguments (`apl.Variable` for arrays, `apl.Literal` for scalars or shapes).
- **Output**: Returns a string (the result variable name) representing the output `NumpyBuffer` in the generated C code.
- **Tools**:
  - `ctx.exec(f"{ctx.feed}...")`: Emits C code statements.
  - `ctx.freshen(name)`: Generates unique variable names (e.g., `result_1`, `i_0`).

#### NumpyBuffer (`src/apl2c/codegen/numpy_buffer.py`)

- Defines the `NumpyBuffer` struct with fields:
  - `arr`: Pointer to the NumPy array.
  - `data`: Raw `int64_t*` array data.
  - `length`: Total number of elements.
  - `shape`: Dimension sizes (e.g., `element_0`, `element_1` for a 2D array).
- Provides:
  - `c_alloc(ctx, shape)`: Allocates a `NumpyBuffer` with the specified shape.
  - `c_load(ctx, arr, indices)`: Loads an element at the given indices.
  - `c_store(ctx, arr, indices, value)`: Stores a value at the given indices.

### Tests (`tests/test_codegen.py`)

We have provided a few test cases for the functions that we want to you to fill. Please remove the pytest skips, as you complete the implementations
of the specific functions. 

### Implementation Guidance

- **Read Function Docstrings**: Each codegen function in `codegen.py` has a detailed docstring specifying:
  - Parameters and return type.
  - Operation description and examples (e.g., `[1, 2, 3] + [4, 5, 6] → [5, 7, 9]` for `_c_add`).
  - Step-by-step requirements for array allocation, loop generation, and operation logic.
  - Hints for using `ctx`, `c_alloc`, `c_load`, `c_store`, and managing indentation.

- **Use Test Cases**: The test suite in `tests/test_apl_codegen.py` includes cases for each operation (e.g., `test_add`, `test_reshape`). Run:
  ```bash
  poetry run pytest tests/test_apl_codegen.py -v
  ```
  Tests compare codegen output to interpreter output, ensuring correctness.

- **Study the Interpreter**: Review `interpreter.py` to understand each operation’s semantics. For example:
  - `_iota(ctx, range)`: Uses `np.arange(1, range + 1, dtype=np.int64)`.
  - `_transpose(ctx, arr)`: Uses `np.transpose(arr.arr)`.

- **C Code Structure**: Your generated C code should follow a pattern like:
  ```c
  struct CNumpyBuffer result = *(struct CNumpyBuffer*)alloc_int64((size_t[ndim]){dim0, dim1, ...});
  for (size_t i_0 = 0; i_0 < dim0; i_0++) {
      // ... nested loops ...
      // Perform operation and store result
  }
  ```

### Bonus

We would like to present you with the following options to avail a bonus:
1. Implement additional operations such as dot products, mul-reduce, etc. Please add the specific interpreter functions in `interpreter.py` and the corresponding codegen functions in `codegen.py`. Add test cases for these new operations in `test_apl_codegen.py`. Include a brief description of the new operation as a docstring in the codegen function.
2. Implement a parser that can parse a subset of APL syntax and generate an Abstract Syntax Tree (AST). This parser should be able to handle basic APL expressions and convert them into a format that can be processed by the existing codegen functions. You can create a new file `parser.py` in the `src/apl2c/apl/` directory for this purpose. Add test cases for the parser in a new test file `test_parser.py`.

You have the discretion to choose either one of the above options for a bonus. Please ensure that your code is well-documented and includes test cases to validate the functionality of the new features you implement.

### Additional Information

- **Environment Setup**:
  - Install dependencies with `poetry install --extras test`.
  - Ensure Python 3.12+ and a C compiler (e.g., `gcc`) are installed, as detailed in `CONTRIBUTING.md`.

- **Tips**:
  - Start with simpler functions like `_c_iota` or `_c_neg`, which involve straightforward operations.
  - Progress to `_c_add`, `_c_sub`, and `_c_exp`, which require similar loop structures.
  - Tackle `_c_transpose`, `_c_reduce`, and `_c_reshape` last, as they involve more complex index manipulation or validation.
  - Test each function incrementally to catch errors early.
  - Use the interpreter’s implementation as a reference but focus on generating equivalent C code.

### Deliverables

- Implement all eight codegen functions (`_c_add`, `_c_sub`, `_c_neg`, `_c_exp`, `_c_transpose`, `_c_iota`, `_c_reshape`, `_c_reduce`) in `src/apl2c/apl/codegen.py`.
- Ensure all tests in `tests/test_apl_codegen.py` pass, verifying that codegen output matches interpreter output.
- Adhere to coding standards in `CONTRIBUTING.md` (e.g., consistent indentation, clear variable names).
- Do not modify `interpreter.py`, `numpy_buffer.py`, or other files unless explicitly instructed.
- For submission, please zip the root folder of the project, ensuring all your changes are included and submit it on Canvas.
