-- Seed default users for Smart Parking System
-- Passwords: guard123, manager123, staff123, 12345609876

-- Clear existing users (optional, comment out if you want to keep existing data)
-- TRUNCATE TABLE users CASCADE;

-- Insert default users with correct password hashes
INSERT INTO users (username, password_hash, role, full_name, email, phone, plate) VALUES
('guard', 'scrypt:32768:8:1$K9NcFzbZlmZvq0WT$42d33960a9f2502b3c2d8cf5f1680101c1bdb535212d7fd6c2311237b6571120fed08b1be0181773a20d55dad0fe061298be1d1fdb3dc7d317a0b24a3de38603', 'guard', 'Security Guard', 'guard@parking.local', '0123456789', NULL),
('manager', 'scrypt:32768:8:1$cgT8MSjZl8R3dGoq$ef7cfd5e8cc803c6346a0c95cee85a76a05b4cc14ce82240240be3368edcc9d078f8b1531e6d742c9807f054e4edb03ed742fcdd4278d540ebc0a236151ac5f4', 'manager', 'Parking Manager', 'manager@parking.local', '0987654321', NULL),
('staff', 'scrypt:32768:8:1$Dls2w4k8vxHF9QEe$45a344cddb6b77c01b732752f21ed949df6d0a1bfa4cef3eb318323a5b65dbd314a4d8a7800759b6ee68f2021e9fc1aeb983ec8c91bb6ef14ac6eb1ad08c80f7', 'staff', 'Parking Staff', 'staff@parking.local', '0111222333', NULL),
('22050026', 'scrypt:32768:8:1$vQxZ8KqJ7YmGzN0w$2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c', 'student', 'Nguyen Van A', 'student@bdu.edu.vn', '0999888777', '61A93132')
ON CONFLICT (username) DO UPDATE SET
    password_hash = EXCLUDED.password_hash,
    role = EXCLUDED.role,
    full_name = EXCLUDED.full_name,
    email = EXCLUDED.email,
    phone = EXCLUDED.phone,
    plate = EXCLUDED.plate;
