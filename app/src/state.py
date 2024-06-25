import hashlib
from streamlit.runtime.scriptrunner import get_script_run_ctx
from streamlit.runtime.legacy_caching.hashing import _CodeHasher
from streamlit.runtime import Runtime
from streamlit.web.server.server import Server
import streamlit as st
from typing import Dict, Any
from streamlit.runtime import get_instance
from copy import deepcopy


class _SessionState:
    def __init__(self, session, hash_funcs):
        """Initialize SessionState instance."""
        self.__dict__["_state"] = {
            "data": {},
            "hash": None,
            "hasher": _CodeHasher(hash_funcs),
            "is_rerun": False,
            "session": session,
        }

    def __call__(self, **kwargs):
        """Initialize state data once."""
        for item, value in kwargs.items():
            if item not in self._state["data"]:
                self._state["data"][item] = value

    def __getitem__(self, item):
        """Return a saved state value, None if item is undefined."""
        return self._state["data"].get(item, None)

    def __getattr__(self, item):
        """Return a saved state value, None if item is undefined."""
        return self._state["data"].get(item, None)

    def __setitem__(self, item, value):
        """Set state value."""
        self._state["data"][item] = value

    def __setattr__(self, item, value):
        """Set state value."""
        self._state["data"][item] = value

    def clear(self):
        """Clear session state and request a rerun."""
        self._state["data"].clear()
        self._state["session"].request_rerun()

    def sync(self):
        """Rerun the app with all state values up to date from the beginning to fix rollbacks."""
        if self._state["is_rerun"]:
            self._state["is_rerun"] = False
        elif self._state["hash"] is not None:
            if self._state["hash"] != self._state["hasher"].to_bytes(self._state["data"], None):
                self._state["is_rerun"] = True
                self._state["session"].request_rerun(None)

        self._state["hash"] = self._state["hasher"].to_bytes(
            self._state["data"], None)


def _get_session():
    runtime = get_instance()
    session_id = get_script_run_ctx().session_id
    session_info = runtime._session_mgr.get_session_info(session_id)

    if session_info is None:
        raise RuntimeError("Couldn't get your Streamlit Session object.")
    return session_info.session


# Tạo hàm băm tùy chỉnh cho _SessionState
def session_state_hash_func(state):
    return hashlib.md5(str(state.__dict__).encode()).hexdigest()


def get_state(hash_funcs=None):
    session = _get_session()

    if not hasattr(session, "_custom_session_state"):
        session._custom_session_state = _SessionState(session, hash_funcs)

    return session._custom_session_state


CONFIG_DEFAULTS: Dict[str, Any] = {"slider_value": 0}


def provide_state(hash_funcs=None):
    # if hash_funcs is None:
    #     hash_funcs = {}
    # hash_funcs[_SessionState] = session_state_hash_func
    def inner(func):
        def wrapper(*args, **kwargs):
            state = get_state(hash_funcs=hash_funcs)
            if state.client_config is None:
                state.client_config = deepcopy(CONFIG_DEFAULTS)

            return_value = func(state=state, *args, **kwargs)
            state.sync()
            return return_value

        return wrapper
    return inner
