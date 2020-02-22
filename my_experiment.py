from sacred import Experiment

ex = Experiment()

@ex.config
def my_config():
    foo = 42
    bar = 'baz'

@ex.capture
def some_function(a, foo, bar=10):
    print("a={}, foo={}, bar={}".format(a, foo, bar))

@ex.automain
def my_main():
    some_function(1, 2, 3)
    some_function(1)
    some_function(1, bar=12)
    #some_function()             # error
