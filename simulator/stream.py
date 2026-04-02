import subprocess
import os
import sys
import time
import shutil
import glob

# Gets the project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Paths to the video files
GATE_VIDEO = os.path.join(BASE_DIR, 'static', 'video', 'CAM_GATE.mp4')
PARKING_VIDEO = os.path.join(BASE_DIR, 'static', 'video', 'CAM_PARKING.mp4')

# RTSP destination URLs on local mediamtx server
GATE_RTSP_URL = "rtsp://localhost:8554/cam_gate"
PARKING_RTSP_URL = "rtsp://localhost:8554/cam_parking"

def resolve_ffmpeg_bin():
    """Find ffmpeg executable from env, PATH, or common install locations."""
    candidates = []
    env_bin = os.getenv("FFMPEG_BIN")
    if env_bin:
        candidates.append(env_bin)

    path_bin = shutil.which("ffmpeg")
    if path_bin:
        candidates.append(path_bin)

    user_profile = os.environ.get("USERPROFILE", "")
    local_app = os.environ.get("LOCALAPPDATA", "")
    common_paths = [
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe",
        os.path.join(user_profile, "scoop", "apps", "ffmpeg", "current", "bin", "ffmpeg.exe"),
    ]
    candidates.extend(common_paths)

    # Winget installs FFmpeg in a versioned directory under LocalAppData.
    winget_pattern = os.path.join(
        local_app,
        "Microsoft",
        "WinGet",
        "Packages",
        "Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe",
        "ffmpeg-*-full_build",
        "bin",
        "ffmpeg.exe",
    )
    winget_bins = sorted(glob.glob(winget_pattern), reverse=True)
    candidates.extend(winget_bins)

    for c in candidates:
        if not c:
            continue
        if os.path.exists(c):
            return c
    return None

def start_stream(video_path, rtsp_url):
    """Uses ffmpeg to stream a local video file to an RTSP server in an endless loop."""
    if not os.path.exists(video_path):
        print(f"[ERROR] Khong tim thay file video: {video_path}")
        return None

    ffmpeg_bin = resolve_ffmpeg_bin()
    if not ffmpeg_bin:
        print("[ERROR] Khong tim thay ffmpeg. Cai ffmpeg hoac set bien moi truong FFMPEG_BIN.")
        print("        Vi du: setx FFMPEG_BIN \"C:\\ffmpeg\\bin\\ffmpeg.exe\"")
        return None

    print(f"[INFO] Dang phat {os.path.basename(video_path)} lap lai lien tuc len: {rtsp_url}")
    
    # ffmpeg command: 
    # -re : read input at native frame rate (realtime)
    # -stream_loop -1 : loop infinitely
    # -c copy : copy codecs (no re-encoding, saves CPU)
    # -rtsp_transport tcp : use TCP to avoid UDP packet loss
    cmd = [
        ffmpeg_bin,
        '-re',
        '-stream_loop', '-1',
        '-i', video_path,
        '-c', 'copy',
        '-f', 'rtsp',
        '-rtsp_transport', 'tcp',
        rtsp_url
    ]
    
    # Hide ffmpeg output to keep terminal clean, only capture errors if needed
    process = subprocess.Popen(
        cmd, 
        stdout=subprocess.DEVNULL, 
        stderr=subprocess.DEVNULL
    )
    return process

if __name__ == '__main__':
    print("="*50)
    print(" Khoi dong RTSP Simulator bang FFmpeg")
    print("="*50)
    
    p1 = start_stream(GATE_VIDEO, GATE_RTSP_URL)
    p2 = start_stream(PARKING_VIDEO, PARKING_RTSP_URL)
    
    if not p1 and not p2:
        print("Khong co video nao duoc phat. Dang thoat...")
        sys.exit(1)
        
    print("\n[OK] He thong stream gia lap dang chay nen.")
    print(f"Luong Camera Cong : {GATE_RTSP_URL}")
    print(f"Luong Camera Bai  : {PARKING_RTSP_URL}")
    print("\n-> Bam Ctrl+C de dung phat stream.")
    
    try:
        # Keep main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n[INFO] Nhan lenh dung. Dang tat ffmpeg...")
        if p1: p1.terminate()
        if p2: p2.terminate()
        print("[OK] Da tat gia lap RTSP hoan tat.")
