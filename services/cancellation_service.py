from threading import Event


_analysis_cancel_event = Event()


def begin_analysis():
    _analysis_cancel_event.clear()


def request_analysis_cancel():
    _analysis_cancel_event.set()


def is_analysis_cancelled():
    return _analysis_cancel_event.is_set()
