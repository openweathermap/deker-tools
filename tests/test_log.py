import logging
from logging import getLogger

from deker_tools.log import tree, LoggerNode, set_logger


def test_tree():
    logger_name = "test"
    logger = getLogger(logger_name)
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
        assert 'foo' in caplog.text
        assert '|' in caplog.text


