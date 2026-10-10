"""One evaluation plan shared by scope analysis and source insertion."""
import ast


def call_arguments(call, stem):
    arguments = []
    for slot, values in (('positional', call.args), ('keyword', [k.value for k in call.keywords])):
        for index, value in enumerate(values):
            suffix = 'arg' if slot == 'positional' else 'kw'
            arguments.append(dict(slot=slot, index=index, binding=f'{stem}_{suffix}{index}', expression=ast.unparse(value)))
    return arguments
