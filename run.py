from model import ex
x_choices = [2, 3, 4]
for x in x_choices:
    ex.run(config_updates={'x': x})