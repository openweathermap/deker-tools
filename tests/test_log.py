import logging

from logging import getLogger

from deker_tools.log import LoggerNode, set_logger, tree


def test_tree():
    logger_name = "test"
    getLogger(logger_name)  # registers the logger in the tree
    logger_tree = tree()

    def check_node(node: LoggerNode):
        if node.name == logger_name:
            return True

        for child_node in node.children:
            if check_node(child_node):
                return True

    found = check_node(logger_tree)
    assert found


def test_formatting_is_applied(caplog):
    format_ = "%(levelname)s | %(message)s | foo"
    logger = getLogger(__name__)
    set_logger(format_)
    with caplog.at_level(logging.DEBUG):
        text = "text"
        logger.debug(text)
        assert text in caplog.text
        assert "foo" in caplog.text
        assert "|" in caplog.text


def test_formatting_is_scoped_to_the_named_logger_and_its_children():
    format_ = "%(message)s | scoped"
    parent = getLogger("deker_tools.tests.scoped")
    child = getLogger("deker_tools.tests.scoped.child")
    sibling = getLogger("deker_tools.tests.other")
    handlers = {logger: logging.StreamHandler() for logger in (parent, child, sibling)}
    untouched = logging.Formatter("%(message)s | untouched")
    for logger, handler in handlers.items():
        handler.setFormatter(untouched)
        logger.addHandler(handler)
    try:
        set_logger(format_, "deker_tools.tests.scoped")
        assert handlers[parent].formatter._fmt == format_
        assert handlers[child].formatter._fmt == format_
        assert handlers[sibling].formatter is untouched
    finally:
        for logger, handler in handlers.items():
            logger.removeHandler(handler)


def test_formatting_scoped_to_a_logger_does_not_configure_the_root_logger():
    root_handlers = list(logging.root.handlers)
    set_logger("%(message)s", "deker_tools.tests.scoped")
    assert logging.root.handlers == root_handlers
