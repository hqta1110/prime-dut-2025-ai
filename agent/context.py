from contextvars import ContextVar

SESSION_STATE = ContextVar("SESSION_STATE", default=None)
