import functools
import threading
import time

import pytest

from cantok import DefaultToken, SimpleToken, TimeoutToken

from deker_tools.cancel import (
    TokenCancelledError,
    TokenTimeoutError,
    WriteToken,
    cancellable,
    cancellation_scope,
    check,
    mark_started,
    read_token,
    resolve_timeout,
    write_token,
)
from deker_tools.typings import Sentinel

pytestmark = pytest.mark.timeout(10)


def wait_until_cancelled(token):
    while not token.cancelled:
        time.sleep(0.01)


class TestCancellationScope:
    def test_no_scope_gives_a_default_write_token(self):
        assert isinstance(write_token(), DefaultToken)

    def test_scope_is_seen_inside_and_gone_after(self):
        scope = SimpleToken()
        with cancellation_scope(scope):
            token = write_token()
            assert isinstance(token, WriteToken)
            scope.cancel()
            with pytest.raises(TokenCancelledError):
                check(token)
        assert isinstance(write_token(), DefaultToken)

    def test_nested_scopes_are_restored_in_order(self):
        outer, inner = SimpleToken(), SimpleToken()
        with cancellation_scope(outer):
            with cancellation_scope(inner):
                inner_token = write_token()
            outer_token = write_token()
            inner.cancel()
            with pytest.raises(TokenCancelledError):
                check(inner_token)
            check(outer_token)
            outer.cancel()
            with pytest.raises(TokenCancelledError):
                check(outer_token)

    def test_scope_is_restored_after_an_exception(self):
        with pytest.raises(RuntimeError), cancellation_scope(SimpleToken()):
            raise RuntimeError("inside the scope")
        assert isinstance(write_token(), DefaultToken)


class TestCheck:
    @pytest.mark.parametrize(
        "make_token",
        [
            pytest.param(SimpleToken, id="simple"),
            pytest.param(DefaultToken, id="default"),
            pytest.param(functools.partial(TimeoutToken, 60), id="timeout"),
        ],
    )
    def test_a_live_token_passes(self, make_token):
        check(make_token())

    def test_a_cancelled_token_raises_the_cancel_error(self):
        token = SimpleToken()
        token.cancel()
        with pytest.raises(TokenCancelledError) as error:
            check(token)
        assert not isinstance(error.value, TokenTimeoutError)
        assert error.value.token is token

    def test_an_expired_timeout_raises_the_timeout_error(self):
        token = TimeoutToken(0.05)
        wait_until_cancelled(token)
        with pytest.raises(TokenTimeoutError) as error:
            check(token)
        assert error.value.token is token

    def test_the_error_carries_the_text_and_no_chain(self):
        token = SimpleToken()
        token.cancel()
        with pytest.raises(TokenCancelledError) as error:
            check(token)
        assert str(error.value)
        assert error.value.__cause__ is None
        assert error.value.__suppress_context__ is True


class TestReadToken:
    def test_live_until_the_timeout(self):
        token = read_token(60)
        check(token)
        assert not token.cancelled

    def test_timeout_without_scope(self):
        token = read_token(1)
        wait_until_cancelled(token)
        with pytest.raises(TokenTimeoutError) as error:
            check(token)
        assert isinstance(error.value.token, TimeoutToken)

    def test_timeout_inside_a_scope(self):
        with cancellation_scope(SimpleToken()):
            token = read_token(1)
        wait_until_cancelled(token)
        with pytest.raises(TokenTimeoutError):
            check(token)

    def test_cancelled_scope_cancels_the_read(self):
        scope = SimpleToken()
        with cancellation_scope(scope):
            token = read_token(60)
        scope.cancel()
        with pytest.raises(TokenCancelledError) as error:
            check(token)
        assert not isinstance(error.value, TokenTimeoutError)
        assert error.value.token is scope


class TestWriteToken:
    def test_default_token_without_scope(self):
        token = write_token()
        mark_started(token)
        check(token)
        assert not token.cancelled

    def test_scope_cancelled_before_the_start_cancels(self):
        scope = SimpleToken()
        with cancellation_scope(scope):
            token = write_token()
        scope.cancel()
        with pytest.raises(TokenCancelledError) as error:
            check(token)
        assert error.value.token is token

    def test_start_is_refused_when_the_scope_is_cancelled(self):
        scope = SimpleToken()
        with cancellation_scope(scope):
            token = write_token()
        scope.cancel()
        with pytest.raises(TokenCancelledError):
            mark_started(token)
        with pytest.raises(TokenCancelledError):
            check(token)

    def test_scope_cancelled_after_the_start_does_not_cancel(self):
        scope = SimpleToken()
        with cancellation_scope(scope):
            token = write_token()
        mark_started(token)
        scope.cancel()
        check(token)
        assert not token.cancelled

    @staticmethod
    def _check_into(token, errors):
        try:
            check(token)
        except TokenCancelledError as e:
            errors.append(e)

    def test_start_is_shared_by_the_threads_of_the_operation(self):
        scope = SimpleToken()
        with cancellation_scope(scope):
            token = write_token()
        starter = threading.Thread(target=mark_started, args=(token,))
        starter.start()
        starter.join()
        scope.cancel()
        errors = []
        checker = threading.Thread(target=self._check_into, args=(token, errors))
        checker.start()
        checker.join()
        assert errors == []

    def test_the_lock_timeout_and_the_scope_are_told_apart(self):
        scope = SimpleToken()
        with cancellation_scope(scope):
            token = write_token()
        lock_timeout = TimeoutToken(0.05)
        wait_until_cancelled(lock_timeout)
        with pytest.raises(TokenTimeoutError) as error:
            check(lock_timeout + token)
        assert error.value.token is lock_timeout

        live_lock_timeout = TimeoutToken(60)
        scope.cancel()
        with pytest.raises(TokenCancelledError) as error:
            check(live_lock_timeout + token)
        assert not isinstance(error.value, TokenTimeoutError)
        assert error.value.token is token


@pytest.mark.parametrize(
    "make_token",
    [
        pytest.param(SimpleToken, id="simple"),
        pytest.param(DefaultToken, id="default"),
        pytest.param(functools.partial(TimeoutToken, 60), id="timeout"),
    ],
)
def test_mark_started_does_nothing_to_other_tokens(make_token):
    token = make_token()
    mark_started(token)
    check(token)
    assert not token.cancelled


class TestCancellable:
    def test_gives_every_item_of_a_live_token(self):
        assert list(cancellable([1, 2, 3], SimpleToken())) == [1, 2, 3]

    def test_gives_nothing_of_an_empty_list(self):
        assert list(cancellable([], SimpleToken())) == []

    def test_stops_before_the_next_item_once_cancelled(self):
        token = SimpleToken()
        taken = []
        with pytest.raises(TokenCancelledError):
            for item in cancellable([1, 2, 3], token):
                taken.append(item)
                if item == 2:
                    token.cancel()
        assert taken == [1, 2]

    def test_a_cancelled_token_gives_no_item(self):
        token = SimpleToken()
        token.cancel()
        taken = []
        with pytest.raises(TokenCancelledError):
            taken.extend(cancellable([1, 2, 3], token))
        assert taken == []


@pytest.mark.parametrize(
    ("timeout", "default", "expected"),
    [
        pytest.param(Sentinel, 60, 60, id="not-given"),
        pytest.param(5, 60, 5, id="shorter"),
        pytest.param(300, 60, 300, id="longer"),
    ],
)
def test_resolve_timeout(timeout, default, expected):
    assert resolve_timeout(timeout, default) == expected
