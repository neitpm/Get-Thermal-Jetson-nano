import cv2
import requests
from requests.auth import HTTPDigestAuth
import threading
import time
import numpy as np
import json
import io

# --- CẤU HÌNH CAMERA ---
CAMERA_IP = "192.168.10.152"
USERNAME = "admin"
PASSWORD = "Demo@2024"

# --- BIẾN TOÀN CỤC ---
current_matrix = None # Ma trận nhiệt độ float32 (384x288)
display_frame = None  # Ảnh JPEG nhiệt để hiển thị
target_point = None   # (x, y) click trên màn hình
current_temp = "..."
data_lock = threading.Lock()
running = True

def parse_multipart(data, boundary):
    """Phân tách gói tin multipart từ Hikvision"""
    parts = data.split(boundary)
    result = {}
    for part in parts:
        if b'Content-Type: application/json' in part:
            header_end = part.find(b'\r\n\r\n')
            result['json'] = json.loads(part[header_end+4:].strip())
        elif b'Content-Type: image/pjpeg' in part or b'Content-Type: image/jpeg' in part:
            # Lưu ý: Hikvision có thể gửi nhiều ảnh (nhiệt và quang)
            # Chúng ta lấy ảnh đầu tiên làm ảnh nhiệt
            header_end = part.find(b'\r\n\r\n')
            if 'image' not in result:
                result['image'] = part[header_end+4:]
        elif b'Content-Type: application/octet-stream' in part or len(part) > 100000:
            # Phần dữ liệu nhị phân lớn thường là ma trận nhiệt độ
            header_end = part.find(b'\r\n\r\n')
            if header_end != -1:
                result['p2p'] = part[header_end+4:]
    return result

def fetch_thermal_matrix_loop():
    """Luồng liên tục tải ma trận nhiệt độ từ ISAPI"""
    global current_matrix, display_frame, current_temp, running
    url = f"http://{CAMERA_IP}/ISAPI/Thermal/channels/2/thermometry/jpegPicWithAppendData?format=json"
    auth = HTTPDigestAuth(USERNAME, PASSWORD)
    
    while running:
        try:
            response = requests.get(url, auth=auth, timeout=2)
            if response.status_code == 200:
                boundary = b'--boundary' # Boundary mặc định của Hikvision
                parts = parse_multipart(response.content, boundary)
                
                if 'json' in parts and 'p2p' in parts:
                    meta = parts['json']['JpegPictureWithAppendData']
                    w, h = meta['jpegPicWidth'], meta['jpegPicHeight']
                    
                    # Chuyển đổi dữ liệu nhị phân sang ma trận numpy float32
                    raw_data = parts['p2p']
                    # Thường dữ liệu P2P nằm ở cuối phần header, cần lấy đúng độ dài
                    p2p_len = meta['p2pDataLen']
                    matrix = np.frombuffer(raw_data[:p2p_len], dtype=np.float32).reshape(h, w)
                    
                    # Giải mã ảnh JPEG để hiển thị
                    nparr = np.frombuffer(parts['image'], np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    with data_lock:
                        current_matrix = matrix
                        display_frame = frame
                        # Cập nhật nhiệt độ nếu đang chọn một điểm
                        if target_point:
                            tx, ty = target_point
                            # Scale tọa độ hiển thị sang tọa độ ma trận (384x288)
                            fh, fw = frame.shape[:2]
                            mx = int((tx / fw) * w)
                            my = int((ty / fh) * h)
                            # Kiểm tra giới hạn
                            mx = max(0, min(w-1, mx))
                            my = max(0, min(h-1, my))
                            val = matrix[my, mx]
                            current_temp = f"{val:.1f}"
                            
        except Exception as e:
            print(f"Lỗi tải ma trận: {e}")
        time.sleep(0.1) # Tốc độ cập nhật ~10fps

def mouse_callback(event, x, y, flags, param):
    """Xử lý click chuột"""
    global target_point
    if event == cv2.EVENT_LBUTTONDOWN:
        with data_lock:
            target_point = (x, y)

def main():
    global running
    window_name = "Hikvision - Realtime Thermal Matrix"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)

    # Khởi chạy luồng tải dữ liệu
    thread = threading.Thread(target=fetch_thermal_matrix_loop, daemon=True)
    thread.start()

    print("--- ỨNG DỤNG MA TRẬN NHIỆT ĐÃ SẴN SÀNG ---")
    print("Click vào bất kỳ đâu để xem nhiệt độ tức thì từ Ma Trận.")

    while True:
        with data_lock:
            if display_frame is not None:
                frame = display_frame.copy()
                if target_point:
                    px, py = target_point
                    cv2.drawMarker(frame, (px, py), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
                    cv2.putText(frame, f"{current_temp} C", (px + 10, py - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.putText(frame, "CHEDO MA TRAN NHIET (PIXEL-TO-PIXEL)", (10, 25), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                cv2.imshow(window_name, frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    running = False
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
