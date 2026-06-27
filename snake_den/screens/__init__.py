"""The hub's pygame tab screens (models, train, eval, settings) + the footer.

Each screen imports pygame (via widgets), so -- like the product's gui.py -- no
test imports this package; the headless smoke + manual checklist are its
correctness authority. The pure data the screens render lives in pygame-free
modules (charts, registry, viewdata, scheme, suites) that *are* tested.
"""
