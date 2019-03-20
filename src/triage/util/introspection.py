import inspect


def classpath(klass):
    return f"{klass.__module__}.{klass.__name__}"


def bound_call_signature(klass, local_variables):
    call_signature = inspect.signature(klass).bind_partial(**local_variables).arguments
    if 'kwargs' in call_signature:
        passed_kwargs = call_signature['kwargs']
    else:
        passed_kwargs = call_signature
    return passed_kwargs
