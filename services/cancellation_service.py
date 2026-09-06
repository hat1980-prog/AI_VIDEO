from threading import Event


_analysis_cancel_event = Event()
_skip_current_video_event = Event()


def begin_analysis():
    _analysis_cancel_event.clear()
    _skip_current_video_event.clear()


def request_analysis_cancel():
    _analysis_cancel_event.set()


def is_analysis_cancelled():
    return _analysis_cancel_event.is_set()


def request_current_video_skip():
    _skip_current_video_event.set()


def is_current_video_skip_requested():
    return _skip_current_video_event.is_set()


def should_stop_current_video():
    return is_analysis_cancelled() or is_current_video_skip_requested()


def consume_current_video_skip():
    requested = _skip_current_video_event.is_set()
    _skip_current_video_event.clear()
    return requested
