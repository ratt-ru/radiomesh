from typing import Callable
from numba.core import cgutils, types
from numba.extending import intrinsic

STOKES_CONVERSION = {
    "RR": {("I", "V"): lambda i, v: i + v + 0j},
    "RL": {("Q", "U"): lambda q, u: q + u * 1j},
    "LR": {("Q", "U"): lambda q, u: q - u * 1j},
    "LL": {("I", "V"): lambda i, v: i - v + 0j},
    "XX": {("I", "Q"): lambda i, q: i + q + 0j},
    "XY": {("U", "V"): lambda u, v: u + v * 1j},
    "YX": {("U", "V"): lambda u, v: u - v * 1j},
    "YY": {("I", "Q"): lambda i, q: i - q + 0j},
}

POL_CONVERSION = {
    "I": {
        ("XX", "YY"): lambda xx, yy: (xx.real + yy.real) / 2.0,
        ("RR", "LL"): lambda rr, ll: (rr.real + ll.real) / 2.0,
    },
    "Q": {
        ("XX", "YY"): lambda xx, yy: (xx.real - yy.real) / 2.0,
        ("RL", "LR"): lambda rl, lr: (rl.real + lr.real) / 2.0,
    },
    "U": {
        ("XY", "YX"): lambda xy, yx: (xy.real + yx.real) / 2.0,
        ("RL", "LR"): lambda rl, lr: (rl.imag + lr.imag) / 2.0,
    },
    "V": {
        ("XY", "YX"): lambda xy, yx: (xy.imag - yx.imag) / 2.0,
        ("RR", "LL"): lambda rr, ll: (rr.real - ll.real) / 2.0,
    },
}


def load_data_factory(ndata: int) -> Callable:
    """Generates an intrinsic that reads :code:`ndata`
    values from an array at a given index into a tuple"""
    @intrinsic
    def load_data(typingctx, array, index):
        if (not isinstance(array, types.Array) or array.ndim != len(index) + 1):
            raise TypeError(f"'array' ({array}) should be a {len(index) + 1}D array")

        if not isinstance(index, types.BaseTuple) or not all(isinstance(i, types.Integer) for i in index):
            raise TypeError(f"'index' {index} must be a tuple of integers")

        return_type = types.Tuple([array.dtype] * ndata)
        sig = return_type(array, index)

        def index_factory(pol):
            """ Index array with the first N-1 indices combined with pol"""
            return lambda array, index: array[index + (pol,)]

        def codegen(context, builder, signature, args):
            array_type, index_type = signature.args
            llvm_ret_type = context.get_value_type(signature.return_type)
            pol_tuple = cgutils.get_null_value(llvm_ret_type)

            for p in range(ndata):
                sig = array_type.dtype(array_type, index_type)
                value = context.compile_internal(builder, index_factory(p), sig, args)
                pol_tuple = builder.insert_value(pol_tuple, value, p)

            return pol_tuple

        return sig, codegen

    return load_data


def apply_weight_factory(ndata):
    """Returns an intrinsic that applies weight to a tuple of data"""
    @intrinsic
    def apply_visibility_weights(typingctx, data, weight):
        if not isinstance(data, types.UniTuple) or len(data) != ndata:
            raise TypeError(f"'data' ({data}) must be a tuple of length {ndata}")

        is_float_weight = isinstance(weight, types.Float)
        is_tuple_weight = isinstance(weight, types.UniTuple) and len(weight) == ndata

        if not is_float_weight and not is_tuple_weight:
            raise TypeError(
                f"'weight' ({weight}) must be a float or "
                f"a tuple of floats of length {ndata}"
            )

        unified_type = typingctx.unify_types(data.dtype, weight if is_float_weight else weight.dtype)
        return_type = types.Tuple([unified_type] * ndata)
        sig = return_type(data, weight)

        def apply_weight_float_factory(p):
            return lambda t, w: t[p] * w

        def apply_weight_tuple_factory(p):
            return lambda t, w: t[p] * w[p]

        def codegen(context, builder, signature, args):
            data_type, weight_type = signature.args
            llvm_ret_type = context.get_value_type(signature.return_type)
            return_tuple = cgutils.get_null_value(llvm_ret_type)

            for p in range(ndata):
                # Apply weights to data
                sig = unified_type(data_type, weight_type)
                factory = apply_weight_float_factory if is_float_weight else apply_weight_tuple_factory
                value = context.compile_internal(builder, factory(p), sig, args)
                return_tuple = builder.insert_value(return_tuple, value, p)

            return return_tuple

        return sig, codegen

    return apply_visibility_weights


def accumulate_data_factory(ndata, store_index=-1):
    """Returns an intrinsic that accumulates a :code:`len(data_schema)`
    tuple of values in an array at a given `store_index`"""
    @intrinsic
    def accumulate_data(typingctx, data_tuple, array, index):
        if (not isinstance(data_tuple, types.UniTuple) or len(data_tuple) != ndata):
            raise TypeError(f"'pol_tuple' ({data_tuple}) should be a {ndata} tuple")

        if (not isinstance(array, types.Array) or array.ndim != len(index) + 1):
            raise TypeError(f"'array' ({array}) should be a {len(index) + 1}D array")

        if not isinstance(index, types.BaseTuple) or not all(isinstance(i, types.Integer) for i in index):
            raise TypeError(f"'index' {index} must be a tuple of integers")

        sig = types.NoneType("none")(data_tuple, array, index)
        # -1 signifies the store_index should be at the end of the tuple
        ii = ndata if store_index < 0 else store_index

        def assign_factory(pol):
            """ Index array with the first N-1 indices combined with pol"""
            def assign(value, array, index):
                array[index[:ii] + (pol,) + index[ii:]] += value[pol]

            return assign

        def codegen(context, builder, signature, args):
            data_tuple_type, array_type, index_type = signature.args
            sig = types.NoneType("none")(data_tuple_type, array_type, index_type)

            for p in range(ndata):
                context.compile_internal(builder, assign_factory(p), sig, args)

            return None

        return sig, codegen

    return accumulate_data



def pol_to_stokes_factory(pol_schema, stokes_schema):
    @intrinsic
    def converter(typingctx, data):
        if not isinstance(data, types.UniTuple) or len(data) != len(pol_schema):
            raise TypeError(f"data ({data}) should be a length {len(pol_schema)} tuple")

        pol_schema_map = {c: i for i, c in enumerate(pol_schema)}
        conv_map = {}

        for stokes in stokes_schema:
            try:
                conv_schema = POL_CONVERSION[stokes]
            except KeyError:
                raise KeyError(
                    f"No method for producing stokes {stokes} is registered. "
                    f"The following targets are registered: {list(POL_CONVERSION.keys())}"
                )

            for (c1, c2), fn in conv_schema.items():
              try:
                  i1 = pol_schema_map[c1]
                  i2 = pol_schema_map[c2]
              except KeyError:
                  continue
              else:
                  conv_map[stokes] = (i1, i2, fn)

            if stokes not in conv_map:
                raise TypeError(
                    f"No conversion to stokes {stokes} was possible. "
                    f"The following correlations are available {pol_schema} "
                    f"but one of the following combinations "
                    f"{list(conv_schema.keys())} is required "
                    f"to produces {stokes}"
                )

        float_type = data.dtype.underlying_float
        ret_type = types.Tuple([float_type] * len(stokes_schema))
        sig = ret_type(data)

        def codegen(context, builder, signature, args):
            data, = args
            data_type, = signature.args
            llvm_type = context.get_value_type(signature.return_type)
            stokes_tuple = cgutils.get_null_value(llvm_type)

            for s, (i1, i2, conv_fn) in enumerate(conv_map.values()):
                # Extract polarisations from the data tuple
                p1 = builder.extract_value(data, i1)
                p2 = builder.extract_value(data, i2)

                # Compute stokes from polarisations and insert into result tuple
                sig = signature.return_type[s](data_type.dtype, data_type.dtype)
                stokes = context.compile_internal(builder, conv_fn, sig, [p1, p2])
                stokes_tuple = builder.insert_value(stokes_tuple, stokes, s)

            return stokes_tuple

        return sig, codegen

    return converter

