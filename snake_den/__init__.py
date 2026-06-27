"""snake_den: an experiment cockpit that spawns ./snake to train models.

A separate, turned-in pygame-ce program -- "the hub" in the docs, launched as
``./hub`` / ``python -m snake_den``. A den is where many
snakes live, which is what this is: it manages many ./snake subprocesses to
train, evaluate and compare models in parallel.

It drives the shipped ./snake as a black box -- it imports only slither's two
pure file-format modules (config, model_io) and never the agent/runner/
environment/gui, so the -42 firewall is untouched (it is not on the agent's
decision path at all).
"""
