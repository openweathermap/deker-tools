# deker-tools - shared functions library for deker components
# Copyright (C) 2023-2026  Openweather Ltd.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Cancellation tokens of the DEKER operations.

An operation gets one token when it starts and checks it at the points where stopping breaks nothing. A read stops
on its timeout or when the cancellation scope is cancelled. A write stops only before any of its files is started:
the first started file closes the start flag, and from then on the operation goes to its end whatever happens to the
scope. The scope is the integration channel: a server opens it around a request and cancels its token when the
client goes away.
"""

import threading

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TypeVar

from cantok import AbstractToken, ConditionToken, DefaultToken, TimeoutToken
from cantok.errors import CancellationError

from deker_tools.typings import Sentinel, SentinelType, is_sentinel

__all__ = (
    "TokenCancelledError",
    "TokenTimeoutError",
    "WriteToken",
    "cancellable",
    "cancellation_scope",
    "check",
    "mark_started",
    "read_token",
    "resolve_timeout",
    "write_token",
)

T = TypeVar("T")

_scope: ContextVar[AbstractToken | SentinelType] = ContextVar("deker_cancellation_scope", default=Sentinel)


class TokenCancelledError(Exception):
    """The token of an operation is cancelled: the operation stops where it is.

    :param message: what happened
    :param token: the token that fired
    """

    def __init__(self, message: str, token: AbstractToken) -> None:
        super().__init__(message)
        self.token = token


class TokenTimeoutError(TokenCancelledError):
    """The time of a timeout token of the operation is over."""


class _WriteStart:
    """Start flag of a write operation, shared by every thread of the operation.

    :param scope: token of the cancellation scope the operation was started in
    """

    def __init__(self, scope: AbstractToken) -> None:
        self._scope = scope
        self._started = False
        self._lock = threading.Lock()

    def cancelled(self) -> bool:
        """Tell whether the operation is cancelled: the scope is cancelled and no file is started yet.

        :returns: ``True`` if the operation shall stop
        """
        with self._lock:
            return self._scope.cancelled and not self._started

    def start(self, token: AbstractToken) -> None:
        """Close the flag before the first change of a file, unless the operation is cancelled already.

        The check and the closing are one step, so no file is started after another thread has seen the
        operation cancelled.

        :param token: the write token the flag belongs to, for the error
        :raises TokenCancelledError: if the scope is cancelled and no file of the operation is started
        """
        with self._lock:
            if not self._started and self._scope.cancelled:
                raise TokenCancelledError("The write is cancelled before any of its files was started", token)
            self._started = True


class WriteToken(ConditionToken):
    """Token of a write operation: cancelled while its scope is cancelled and none of its files is started.

    :param scope: token of the cancellation scope the operation was started in
    """

    def __init__(self, scope: AbstractToken) -> None:
        self._write_start = _WriteStart(scope)
        super().__init__(self._write_start.cancelled)

    def start(self) -> None:
        """Close the start flag before the first change of a file.

        :raises TokenCancelledError: if the operation is cancelled before any of its files was started
        """
        self._write_start.start(self)


@contextmanager
def cancellation_scope(token: AbstractToken) -> Iterator[None]:
    """Run the operations started inside with the token as their cancellation scope.

    The scope is an integration API: a server cancels the token when its client goes away. The scope is read when
    an operation starts, in the thread that starts it; the tokens of the operation carry it further.

    :param token: any cantok token
    """
    previous = _scope.set(token)
    try:
        yield
    finally:
        _scope.reset(previous)


def read_token(timeout: int) -> AbstractToken:
    """Build the token of a read: its timeout and the cancellation scope, if there is one.

    :param timeout: seconds the whole read may take
    :returns: the token of the read
    """
    timeout_token = TimeoutToken(timeout)
    scope = _scope.get()
    if is_sentinel(scope):
        return timeout_token
    return timeout_token + scope


def write_token() -> AbstractToken:
    """Build the token of a write: the cancellation scope with the start flag, or a token never cancelled.

    :returns: a ``WriteToken`` inside a cancellation scope, a ``DefaultToken`` outside any
    """
    scope = _scope.get()
    if is_sentinel(scope):
        return DefaultToken()
    return WriteToken(scope)


def mark_started(token: AbstractToken) -> None:
    """Close the start flag of a write before the first change of a file; other tokens are left as they are.

    :param token: token of the operation
    :raises TokenCancelledError: if the write is cancelled before any of its files was started
    """
    if isinstance(token, WriteToken):
        token.start()


def check(token: AbstractToken) -> None:
    """Stop the operation if its token is cancelled.

    :param token: token of the operation
    :raises TokenTimeoutError: if the token that fired is a timeout token
    :raises TokenCancelledError: if any other token fired
    """
    try:
        token.check()
    except CancellationError as e:
        if isinstance(e.token, TimeoutToken):
            raise TokenTimeoutError(str(e), e.token) from None
        raise TokenCancelledError(str(e), e.token) from None


def cancellable(items: Iterable[T], token: AbstractToken) -> Iterator[T]:
    """Give the items one by one, checking the token before each.

    :param items: the units of work of the operation, one after another
    :param token: token of the operation
    :yields: the next item, once the token is checked
    :raises TokenTimeoutError: if the token that fired is a timeout token
    :raises TokenCancelledError: if any other token fired
    """
    for item in items:
        check(token)
        yield item


def resolve_timeout(timeout: int | SentinelType, default: int) -> int:
    """Take the timeout of the call, or the default when it is not given.

    :param timeout: seconds given to the call, or ``Sentinel``
    :param default: seconds of the setting of the client
    :returns: the seconds to use
    """
    if is_sentinel(timeout):
        return default
    return timeout
