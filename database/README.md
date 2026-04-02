# Database Module - Smart Parking System

## Tổng Quan

Module database chứa tất cả các file liên quan đến PostgreSQL database:
- Models (SQLAlchemy ORM)
- Database connection
- Operations (CRUD)
- Initialization script
- SQL schema

## Files

```
database/
├── __init__.py              # Package init
├── db.py                    # Database connection & session
├── models.py                # SQLAlchemy ORM models
├── operations.py            # Database operations (CRUD)
├── init.sql                 # SQL schema definition
└── create_db.py             # Auto-initialization script
```

## Auto-Initialization (Docker)

Khi chạy với Docker, file `create_db.py` sẽ tự động được thực thi khi container khởi động lần đầu.

### Chức năng

1. **Tạo Tables**: Tạo tất cả các bảng nếu chưa tồn tại
2. **Migrations**: Thêm các cột mới nếu cần
3. **Seed Parking Slots**: Tạo 19 parking slots
4. **Seed Users**: Tạo default users

### Default Users

| Username | Password | Role | Mô tả |
|----------|----------|------|-------|
| guard | guard123 | guard | Bảo vệ |
| manager | manager123 | manager | Quản lý |
| staff | staff123 | staff | Nhân viên |
| 22050026 | 12345609876 | student | Sinh viên mẫu |

### Parking Slots

19 slots được tạo tự động:
- Slot 1-6: Zone A
- Slot 7-10: Zone B
- Slot 11-14: Zone C
- Slot 15-17: Zone D
- Slot 18-19: Zone E

## Manual Usage

### Chạy initialization script
```bash
# Từ project root
python database/create_db.py

# Hoặc từ database folder
cd database
python create_db.py
```

### Reset database
```bash
# Drop all tables (cẩn thận!)
psql -U postgres -d PARKING_PLATE -c "DROP SCHEMA public CASCADE; CREATE SCHEMA public;"

# Reinitialize
python database/create_db.py
```

## Database Schema

### Core Tables

1. **parking_slots** - 19 vị trí đỗ xe
2. **vehicles** - Thông tin xe (biển số, loại xe)
3. **parking_sessions** - Phiên gửi xe (vào/ra)
4. **tracking_events** - Log di chuyển xe
5. **gate_logs** - Lịch sử qua cổng
6. **improper_parking_logs** - Vi phạm đỗ xe

### User Management Tables

7. **users** - Tài khoản người dùng
8. **user_vehicles** - Xe đăng ký của user
9. **slot_reservations** - Đặt chỗ trước
10. **notifications** - Thông báo

## Models (SQLAlchemy)

### Example Usage

```python
from database.db import get_db
from database.models import ParkingSlot, Vehicle, User

# Get database session
db = get_db()

# Query parking slots
slots = db.query(ParkingSlot).filter_by(status='free').all()

# Create new vehicle
vehicle = Vehicle(
    plate_text='51K12345',
    vehicle_type='car'
)
db.add(vehicle)
db.commit()

# Close session
db.close()
```

## Operations

Module `operations.py` cung cấp các hàm CRUD tiện lợi:

```python
from database.operations import (
    get_all_slots,
    update_slot_status,
    create_parking_session,
    get_active_sessions
)

# Get all parking slots
slots = get_all_slots()

# Update slot status
update_slot_status(slot_id=1, status='occupied')

# Create parking session
session = create_parking_session(
    vehicle_uuid=vehicle.vehicle_uuid,
    slot_id=1
)
```

## Migrations

Script `create_db.py` bao gồm các migrations:

### 1. users.plate
Thêm cột `plate` vào bảng `users` để lưu biển số xe mặc định.

### 2. improper_parking_logs.slot_number
Thêm cột `slot_number` để track slot bị vi phạm.

### 3. parking_slots.is_vip
Thêm cột `is_vip` để đánh dấu slot VIP.

## Connection String

Format: `postgresql://user:password@host:port/database`

### Development
```
postgresql://postgres:1412@localhost:5432/PARKING_PLATE
```

### Docker
```
postgresql://postgres:1412@postgres:5432/PARKING_PLATE
```

### Production
```
postgresql://user:password@production-host:5432/PARKING_PLATE
```

## Troubleshooting

### Connection refused
```bash
# Check PostgreSQL is running
sudo systemctl status postgresql

# Check connection
psql -U postgres -d PARKING_PLATE
```

### Tables not created
```bash
# Run initialization manually
python database/create_db.py

# Check tables
psql -U postgres -d PARKING_PLATE -c "\dt"
```

### Migration errors
```bash
# Check current schema
psql -U postgres -d PARKING_PLATE -c "\d users"

# Run migrations manually
psql -U postgres -d PARKING_PLATE -c "ALTER TABLE users ADD COLUMN IF NOT EXISTS plate VARCHAR(20)"
```

## Best Practices

1. **Always use sessions properly**
   ```python
   db = get_db()
   try:
       # ... operations ...
       db.commit()
   except:
       db.rollback()
       raise
   finally:
       db.close()
   ```

2. **Use context managers** (if available)
   ```python
   with get_db() as db:
       # ... operations ...
       db.commit()
   ```

3. **Index frequently queried columns**
   - Already indexed: `plate_text`, `status`, `entry_time`

4. **Use transactions for multiple operations**
   ```python
   db.begin()
   try:
       # Multiple operations
       db.commit()
   except:
       db.rollback()
   ```

## Performance Tips

1. Use connection pooling (already configured in `db.py`)
2. Add indexes for frequently queried columns
3. Use `EXPLAIN ANALYZE` to optimize queries
4. Regular `VACUUM` and `ANALYZE`
5. Monitor slow queries

## Security

1. Never commit `.env` with real credentials
2. Use strong passwords in production
3. Restrict database access by IP
4. Enable SSL for database connections
5. Regular backups
6. Audit logging for sensitive operations

---

**Note**: File này được tự động chạy trong Docker. Không cần chạy manual khi deploy với Docker.
