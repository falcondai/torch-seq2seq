problem_registry = {}
model_registry = {}
optimizer_registry = {}

def add_to_registry(registry):
    def register(obj):
        assert obj.__name__ not in registry, '%s is already registered by %r' % (obj.__name__, registry[obj.__name__])

        registry[obj.__name__] = obj
        return obj
    return register

register_problem = add_to_registry(problem_registry)
register_model = add_to_registry(model_registry)
