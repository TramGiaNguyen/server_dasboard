-- Schema Cơ Sở Dữ Liệu Hệ Thống Bãi Đỗ Xe Thông Minh (PostgreSQL)
-- Dự án: Smart Parking
-- Ngày cập nhật: 06/02/2026
-- Lưu ý: Đã loại bỏ các bảng liên quan đến PCCC (Fire/Smoke) theo yêu cầu mới.

-- Bật extension UUID để tạo định danh duy nhất (nếu chưa có)
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- =============================================
-- 1. Bảng Vị Trí Đỗ (Parking Slots)
-- Lưu trữ thông tin của 19 vị trí đỗ xe trong bãi
-- =============================================
CREATE TABLE IF NOT EXISTS parking_slots (
    slot_id SERIAL PRIMARY KEY,
    slot_number INTEGER UNIQUE NOT NULL, -- Số thứ tự logic (1-19)
    slot_name VARCHAR(50),               -- Tên hiển thị ví dụ: "Slot 1"
    status VARCHAR(20) DEFAULT 'free',   -- Trạng thái: 'free' (trống), 'occupied' (có xe), 'reserved' (đặt trước)
    is_occluded BOOLEAN DEFAULT FALSE,   -- Đánh dấu True cho các slot bị che khuất (như slot 19 bị cây che)
    is_vip BOOLEAN DEFAULT FALSE,        -- Đánh dấu True cho các slot dành riêng cho VIP
    coordinates JSONB,                   -- Tọa độ đa giác vùng đỗ (tùy chọn, dùng để tham khảo/vẽ)
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- =============================================
-- 2. Bảng Xe (Vehicles)
-- Kho lưu trữ thông tin tất cả các xe đã từng được hệ thống phát hiện
-- =============================================
CREATE TABLE IF NOT EXISTS vehicles (
    vehicle_uuid UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    plate_text VARCHAR(20) UNIQUE,       -- Biển số xe (nếu nhận diện được)
    vehicle_type VARCHAR(20),            -- Loại xe: 'car' (ô tô), 'bus' (xe buýt), 'truck' (xe tải)
    first_seen TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP, -- Thời điểm lần đầu thấy xe
    last_seen TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,  -- Thời điểm lần cuối thấy xe
    trust_score FLOAT DEFAULT 0.0        -- Độ tin cậy của biển số (dựa trên tần suất nhận diện đúng)
);

-- =============================================
-- 3. Bảng Phiên Đỗ Xe (Parking Sessions - Nhật Ký Vào/Ra)
-- Theo dõi vòng đời một lượt gửi xe: Cổng Vào -> Đỗ Xe -> Cổng Ra
-- =============================================
CREATE TABLE IF NOT EXISTS parking_sessions (
    session_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    vehicle_uuid UUID REFERENCES vehicles(vehicle_uuid),
    
    -- Thông tin Vào (Entry)
    entry_time TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    entry_gate_image_path TEXT,          -- Đường dẫn ảnh chụp xe tại cổng vào
    entry_plate_conf FLOAT,              -- Độ tin cậy OCR tại cổng vào
    
    -- Thông tin Đỗ (Parking)
    assigned_slot_id INTEGER REFERENCES parking_slots(slot_id),
    parked_time TIMESTAMP WITH TIME ZONE, -- Thời điểm xe ổn định vị trí trong slot
    
    -- Thông tin Ra (Exit)
    exit_time TIMESTAMP WITH TIME ZONE,
    exit_gate_image_path TEXT,
    exit_plate_conf FLOAT,
    
    -- Tài chính / Thống kê
    duration_minutes INTEGER,            -- Thời gian gửi xe (phút)
    status VARCHAR(20) DEFAULT 'active'  -- 'active' (đang gửi), 'completed' (đã xong), 'overnight' (qua đêm)
);

-- =============================================
-- 4. Bảng Sự Kiện Tracking (Tracking Events)
-- Log chi tiết di chuyển của xe để debug hoặc xem lại (Replay)
-- =============================================
CREATE TABLE IF NOT EXISTS tracking_events (
    event_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID REFERENCES parking_sessions(session_id),
    event_type VARCHAR(50),              -- Loại sự kiện: 'entered_gate', 'crossed_line_1', 'parked_slot_5'
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    details JSONB                        -- Metadata linh hoạt (độ tin cậy, tọa độ bbox, v.v...)
);

-- =============================================
-- 5. Bảng Lịch Sử Cổng (Gate Logs)
-- Bảng append-only chuyên dụng cho việc lưu lịch sử xe đi qua camera cổng
-- =============================================
CREATE TABLE IF NOT EXISTS gate_logs (
    log_id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    plate_text VARCHAR(20),       -- Biển số nhận diện được
    direction VARCHAR(10),        -- Hướng đi: 'IN' (Vào bãi), 'OUT' (Ra bãi)
    confidence FLOAT,             -- Độ tin cậy OCR
    image_path TEXT               -- Ảnh chụp lúc qua cổng
);

-- =============================================
-- 6. Bảng Users (App user + Web staff)
-- =============================================
CREATE TABLE IF NOT EXISTS users (
    user_id         SERIAL PRIMARY KEY,
    username        VARCHAR(50) UNIQUE NOT NULL,
    password_hash   VARCHAR(255) NOT NULL,
    role            VARCHAR(20) NOT NULL,         -- 'student' | 'guard' | 'manager' | 'staff'
    full_name       VARCHAR(100),
    email           VARCHAR(100),
    phone           VARCHAR(20),
    plate           VARCHAR(20),                 -- Biển số xe mặc định
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_users_role ON users(role);
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);

-- =============================================
-- 7. Bảng User Vehicles (Biển số xe đăng ký)
-- =============================================
CREATE TABLE IF NOT EXISTS user_vehicles (
    id              SERIAL PRIMARY KEY,
    user_id         INTEGER NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    plate_text      VARCHAR(20) NOT NULL,
    is_primary      BOOLEAN DEFAULT FALSE,
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, plate_text)
);

CREATE INDEX IF NOT EXISTS idx_user_vehicles_user ON user_vehicles(user_id);

-- =============================================
-- 8. Bảng Slot Reservations (Đặt trước slot)
-- =============================================
CREATE TABLE IF NOT EXISTS slot_reservations (
    reservation_id  SERIAL PRIMARY KEY,
    user_id         INTEGER NOT NULL REFERENCES users(user_id),
    slot_id         INTEGER NOT NULL REFERENCES parking_slots(slot_id),
    booking_date    DATE NOT NULL,
    time_from       TIME NOT NULL,
    time_to         TIME NOT NULL,
    arrival_time    TIME,
    plate_text      VARCHAR(20) NOT NULL,
    status          VARCHAR(20) DEFAULT 'pending',  -- pending | confirmed | completed | cancelled
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_reservations_user ON slot_reservations(user_id);
CREATE INDEX IF NOT EXISTS idx_reservations_slot_date ON slot_reservations(slot_id, booking_date);
CREATE INDEX IF NOT EXISTS idx_reservations_status ON slot_reservations(status);

-- =============================================
-- 9. Bảng Notifications (Thông báo cho app)
-- =============================================
CREATE TABLE IF NOT EXISTS notifications (
    notification_id SERIAL PRIMARY KEY,
    user_id         INTEGER REFERENCES users(user_id) ON DELETE CASCADE,
    title           VARCHAR(200) NOT NULL,
    body            TEXT,
    type            VARCHAR(30),
    related_id      INTEGER,
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    read_at         TIMESTAMP WITH TIME ZONE
);

CREATE INDEX IF NOT EXISTS idx_notifications_user ON notifications(user_id);
CREATE INDEX IF NOT EXISTS idx_notifications_created ON notifications(created_at DESC);

-- Bổ sung slot_number cho improper_parking_logs (chạy sau khi models đã tạo bảng)
-- ALTER TABLE improper_parking_logs ADD COLUMN IF NOT EXISTS slot_number INTEGER;

-- Tạo Index để tăng tốc truy vấn
CREATE INDEX IF NOT EXISTS idx_vehicle_plate ON vehicles(plate_text);
CREATE INDEX IF NOT EXISTS idx_session_active ON parking_sessions(status) WHERE status = 'active';
CREATE INDEX IF NOT EXISTS idx_session_entry ON parking_sessions(entry_time);

-- Khởi tạo Dữ liệu ban đầu cho các Slot (Seeding)
-- Sử dụng ON CONFLICT để tránh lỗi khi chạy lại script
INSERT INTO parking_slots (slot_number, slot_name, is_occluded) VALUES
(1, 'Slot 1 - A', FALSE),
(2, 'Slot 2 - A', FALSE),
(3, 'Slot 3 - A', FALSE),
(4, 'Slot 4 - A', FALSE),
(5, 'Slot 5 - A', FALSE),
(6, 'Slot 6 - A', FALSE),
(7, 'Slot 7 - B', FALSE),
(8, 'Slot 8 - B', FALSE),
(9, 'Slot 9 - B', FALSE),
(10, 'Slot 10 - B', FALSE),
(11, 'Slot 11 - C', FALSE),
(12, 'Slot 12 - C', FALSE),
(13, 'Slot 13 - C', FALSE),
(14, 'Slot 14 - C', FALSE),
(15, 'Slot 15 - D', FALSE),
(16, 'Slot 16 - D', FALSE),
(17, 'Slot 17 - D', FALSE),
(18, 'Slot 18 - E', FALSE),
(19, 'Slot 19 - E', FALSE)
ON CONFLICT (slot_number) DO NOTHING;

-- Seed default users
-- Passwords are hashed with werkzeug.security (scrypt)
-- Default passwords: guard123, manager123, staff123, 12345609876
INSERT INTO users (username, password_hash, role, full_name, email, phone, plate) VALUES
('guard', 'scrypt:32768:8:1$K9NcFzbZlmZvq0WT$42d33960a9f2502b3c2d8cf5f1680101c1bdb535212d7fd6c2311237b6571120fed08b1be0181773a20d55dad0fe061298be1d1fdb3dc7d317a0b24a3de38603', 'guard', 'Security Guard', 'guard@parking.local', '0123456789', NULL),
('manager', 'scrypt:32768:8:1$cgT8MSjZl8R3dGoq$ef7cfd5e8cc803c6346a0c95cee85a76a05b4cc14ce82240240be3368edcc9d078f8b1531e6d742c9807f054e4edb03ed742fcdd4278d540ebc0a236151ac5f4', 'manager', 'Parking Manager', 'manager@parking.local', '0987654321', NULL),
('staff', 'scrypt:32768:8:1$Dls2w4k8vxHF9QEe$45a344cddb6b77c01b732752f21ed949df6d0a1bfa4cef3eb318323a5b65dbd314a4d8a7800759b6ee68f2021e9fc1aeb983ec8c91bb6ef14ac6eb1ad08c80f7', 'staff', 'Parking Staff', 'staff@parking.local', '0111222333', NULL),
('22050026', 'scrypt:32768:8:1$T54gv1dXr06swBUi$db190dcb406f05a3b3d0ae6f62805bf7d6096732cec2c2a2c92a7a4900b22973030731851958e6d73bf08c069eb222f71df4db973276cd7ce8c0b40446d8e21c', 'student', 'Nguyen Van A', 'student@bdu.edu.vn', '0999888777', '61A93132')
ON CONFLICT (username) DO UPDATE SET
    password_hash = EXCLUDED.password_hash,
    role = EXCLUDED.role,
    full_name = EXCLUDED.full_name,
    email = EXCLUDED.email,
    phone = EXCLUDED.phone,
    plate = EXCLUDED.plate;
