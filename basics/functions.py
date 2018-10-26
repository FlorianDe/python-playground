def spacer():
    print("-+" * 20 + "-")


# args->variadic arguments, kwargs->variadic keyword arguments
def foo_dynamic(a, *args, **kwargs):
    print(a)
    spacer()
    for arg in args:
        print(arg)
    spacer()
    for kw in kwargs:
        print(kw, ":", kwargs[kw])


foo_dynamic("First argument",
            "Some arg A",
            "Some arg B",
            keyword_a="keyword_value",
            keyword_b="keyword_b_value")


# typed function params and return through function annotations
def ip_join(*args: str, split: str = ".") -> str:
    print("Annotations:", ip_join.__annotations__)
    return split.join(args)


print("Formatted IP: ", ip_join("127", "0", "0", "1"))
