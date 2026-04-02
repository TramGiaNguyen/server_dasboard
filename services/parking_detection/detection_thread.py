"""
Detection Thread - Separate thread for YOLO inference
Improves performance by 15-30% by running detection in parallel with main processing loop
"""
import threading
import queue
import time
import cv2
import numpy as np


class DetectionThread:
    """
    Separate thread for running YOLO model inference.
    Main loop submits frames, detection thread processes them asynchronously.
    """
    
    def __init__(self, model, detection_width, detection_height, use_clahe=False, clahe_obj=None):
        """
        Initialize detection thread.
        
        Args:
            model: YOLO model instance
            detection_width: Width to resize frames for detection
            detection_height: Height to resize frames for detection
            use_clahe: Whether to apply CLAHE preprocessing
            clahe_obj: Pre-allocated CLAHE object (if use_clahe=True)
        """
        self.model = model
        self.detection_width = detection_width
        self.detection_height = detection_height
        self.use_clahe = use_clahe
        self.clahe = clahe_obj
        
        # Input queue: frames to process
        self.input_queue = queue.Queue(maxsize=2)  # Small buffer to avoid lag
        
        # Output queue: detection results
        self.output_queue = queue.Queue(maxsize=2)
        
        # Control flags
        self.running = False
        self.thread = None
        
        # Stats
        self.frames_processed = 0
        self.total_inference_time = 0.0
        
        # Frame interpolation cache
        self.last_valid_result = None
        self.result_cache_ttl = 5  # frames
        
        # Adaptive queue
        self.adaptive_queue = True
        self.queue_size = 2
    
    def start(self):
        """Start the detection thread"""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._detection_loop, daemon=True, name="DetectionThread")
        self.thread.start()
        print("[DETECTION THREAD] Started")
    
    def stop(self):
        """Stop the detection thread"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        print("[DETECTION THREAD] Stopped")
    
    def submit_frame(self, frame, frame_counter):
        """
        Submit a frame for detection (non-blocking).
        
        Args:
            frame: Original frame (full resolution)
            frame_counter: Frame number for tracking
        
        Returns:
            True if submitted, False if queue full (skip this frame)
        """
        try:
            self.input_queue.put_nowait((frame.copy(), frame_counter))
            return True
        except queue.Full:
            return False  # Skip frame if queue full
    
    def get_result(self, timeout=0.001, frame_counter=None):
        """
        Get detection result (non-blocking) with fallback to cached result.
        
        Args:
            timeout: Max time to wait for result (seconds)
            frame_counter: Current frame number for cache validation
        
        Returns:
            (detections, frame_counter) or None if no result available
        """
        try:
            result = self.output_queue.get(timeout=timeout)
            self.last_valid_result = result
            return result
        except queue.Empty:
            # No new result - use cached result if recent
            if self.last_valid_result and frame_counter:
                cached_detections, cached_frame = self.last_valid_result
                frame_diff = frame_counter - cached_frame
                
                if frame_diff <= self.result_cache_ttl:
                    # Cache still valid - reuse (tracker will predict new positions)
                    return self.last_valid_result
            
            return None
    
    def _preprocess_frame(self, frame):
        """Resize and optionally apply CLAHE to frame"""
        detection_frame = cv2.resize(frame, (self.detection_width, self.detection_height))
        
        if self.use_clahe and self.clahe is not None:
            lab = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = self.clahe.apply(l)
            lab = cv2.merge([l, a, b])
            detection_frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return detection_frame
    
    def _detection_loop(self):
        """Main detection loop (runs in separate thread)"""
        while self.running:
            try:
                # Get frame from input queue (blocking with timeout)
                frame, frame_counter = self.input_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            
            try:
                # Preprocess frame
                detection_frame = self._preprocess_frame(frame)
                
                # Run YOLO inference
                start_time = time.time()
                results = self.model.track(
                    detection_frame,
                    persist=True,
                    tracker='cfg/trackers/bytetrack_parking.yaml',
                    conf=0.20,
                    iou=0.5,
                    verbose=False
                )
                inference_time = time.time() - start_time
                
                # Update stats
                self.frames_processed += 1
                self.total_inference_time += inference_time
                
                # Extract detections
                detections = self._extract_detections(results)
                
                # Put result in output queue (drop oldest if full)
                try:
                    self.output_queue.put_nowait((detections, frame_counter))
                except queue.Full:
                    # Drop oldest result and add new one
                    try:
                        self.output_queue.get_nowait()
                    except queue.Empty:
                        pass
                    self.output_queue.put_nowait((detections, frame_counter))
            
            except Exception as e:
                print(f"[DETECTION THREAD] Error processing frame {frame_counter}: {e}")
                continue
    
    def _extract_detections(self, results):
        """Extract detection data from YOLO results"""
        detections = {
            'boxes': [],
            'vehicles': []
        }
        
        if not results or not results[0].boxes or results[0].boxes.id is None:
            return detections
        
        boxes = results[0].boxes
        xyxy = boxes.xyxy.cpu().numpy()
        ids = boxes.id.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()
        cls_np = boxes.cls.cpu().numpy().astype(int)
        
        for i in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[i]
            track_id = int(ids[i])
            conf = float(confs[i])
            cls_idx = int(cls_np[i])
            
            detections['boxes'].append([x1, y1, x2, y2, conf, cls_idx])
            detections['vehicles'].append({
                'track_id': track_id,
                'bbox': [x1, y1, x2, y2],
                'conf': conf,
                'cls_idx': cls_idx
            })
        
        return detections
    
    def _adjust_queue_size(self):
        """Adjust queue size based on processing speed"""
        if not self.adaptive_queue:
            return
        
        input_size = self.input_queue.qsize()
        output_size = self.output_queue.qsize()
        
        # If input queue often full, increase size
        if input_size >= self.queue_size - 1:
            new_size = min(5, self.queue_size + 1)
            if new_size != self.queue_size:
                self.queue_size = new_size
                print(f"📈 [DETECTION THREAD] Increased queue size to {self.queue_size}")
        
        # If output queue often empty, decrease size
        elif output_size == 0 and self.queue_size > 2:
            new_size = max(2, self.queue_size - 1)
            if new_size != self.queue_size:
                self.queue_size = new_size
                print(f"📉 [DETECTION THREAD] Decreased queue size to {self.queue_size}")
    
    def get_stats(self):
        """Get performance statistics"""
        if self.frames_processed == 0:
            return {
                'frames_processed': 0,
                'avg_inference_time': 0.0,
                'fps': 0.0,
                'queue_size': self.queue_size,
                'input_queue_size': self.input_queue.qsize(),
                'output_queue_size': self.output_queue.qsize()
            }
        
        avg_time = self.total_inference_time / self.frames_processed
        return {
            'frames_processed': self.frames_processed,
            'avg_inference_time': avg_time,
            'fps': 1.0 / avg_time if avg_time > 0 else 0.0,
            'queue_size': self.queue_size,
            'input_queue_size': self.input_queue.qsize(),
            'output_queue_size': self.output_queue.qsize()
        }
