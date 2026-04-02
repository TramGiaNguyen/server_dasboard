# Services module initialization
from .parking_detection import process_video_stream
from .gate_camera import process_gate_video_stream

__all__ = ['process_video_stream', 'process_gate_video_stream']
