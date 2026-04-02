# Smart Parking - Mobile App Backend

Backend API server riêng cho mobile app (Flutter). Server này chạy độc lập với web dashboard server.

## Tính năng

- **Authentication**: Login với Bearer token
- **Profile Management**: Xem và cập nhật thông tin user
- **Vehicle Management**: Quản lý danh sách xe
- **Slot Reservation**: Đặt chỗ, xem slot trống, hủy đặt chỗ
- **Notifications**: Nhận thông báo real-time
- **Camera Stream**: MJPEG stream cho gate và parking camera

## Cài đặt

1. Cài đặt dependencies:
```bash
cd backend_app
pip install -r requirements.txt
```

2. Copy và cấu hình file .env:
```bash
cp .env.example .env
# Chỉnh sửa .env với thông tin database và secret key
```

3. Chạy server:
```bash
python app.py
```

Server sẽ chạy trên port 5001 (mặc định).

## API Endpoints

### Authentication
- `POST /api/app/login` - Đăng nhập

### Profile
- `GET /api/app/profile` - Xem profile
- `PUT /api/app/profile` - Cập nhật profile

### Vehicles
- `GET /api/app/vehicles` - Danh sách xe
- `POST /api/app/vehicles` - Thêm xe mới
- `DELETE /api/app/vehicles/<id>` - Xóa xe

### Reservations
- `GET /api/app/slots/available` - Xem slot trống
- `POST /api/app/reservations` - Đặt chỗ
- `GET /api/app/reservations` - Danh sách đặt chỗ
- `POST /api/app/reservations/<id>/cancel` - Hủy đặt chỗ

### Notifications
- `GET /api/app/notifications` - Danh sách thông báo
- `POST /api/app/notifications/<id>/read` - Đánh dấu đã đọc

### Camera
- `GET /api/app/camera-stream?camera=gate|parking` - MJPEG stream

### Health Check
- `GET /health` - Kiểm tra server status

## Authentication

Tất cả endpoints (trừ `/api/app/login` và `/health`) yêu cầu Bearer token trong header:

```
Authorization: Bearer <token>
```

Token được trả về sau khi login thành công.

## CORS

Server đã enable CORS cho tất cả origins. Trong production nên giới hạn origins cụ thể.

## Production Deployment

1. Đổi `SECRET_KEY` trong .env
2. Set `FLASK_ENV=production`
3. Sử dụng production WSGI server (gunicorn):

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5001 --worker-class eventlet app:app
```

## Tích hợp với Main Server

Backend app này dùng chung:
- Database với main server
- `shared.state` module để đọc real-time camera frames và parking status
- `database` module cho models và operations
- `services.reservation_scheduler` cho tính toán remaining time

Cần chạy main server trước để có camera streams và parking detection.
