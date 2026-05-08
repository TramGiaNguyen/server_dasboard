"""
WTForms definitions for Smart Parking System.
Provides CSRF protection for all POST/PUT/DELETE forms.
"""

from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SelectField, SubmitField, HiddenField
from wtforms.validators import DataRequired, Length, EqualTo, Email, Optional, Regexp


class LoginForm(FlaskForm):
    """Web dashboard login form with CSRF protection."""
    username = StringField(
        'Username',
        validators=[DataRequired(message='Vui lòng nhập tên đăng nhập')]
    )
    password = PasswordField(
        'Password',
        validators=[DataRequired(message='Vui lòng nhập mật khẩu')]
    )
    next = HiddenField('next')


class UserForm(FlaskForm):
    """Create/update user form for manager dashboard."""
    username = StringField(
        'Username',
        validators=[
            DataRequired(message='Username không được để trống'),
            Length(min=3, max=50, message='Username phải từ 3-50 ký tự'),
        ]
    )
    password = PasswordField(
        'Mật khẩu',
        validators=[
            Optional(),
            Length(min=6, max=100, message='Mật khẩu phải ít nhất 6 ký tự'),
        ]
    )
    role = SelectField(
        'Vai trò',
        choices=[
            ('student', 'Sinh viên'),
            ('staff', 'Nhân viên'),
            ('guard', 'Bảo vệ'),
            ('manager', 'Quản lý'),
        ],
        validators=[DataRequired(message='Vui lòng chọn vai trò')],
    )
    full_name = StringField(
        'Họ tên',
        validators=[Optional(), Length(max=100)]
    )
    email = StringField(
        'Email',
        validators=[Optional(), Email(message='Email không hợp lệ')]
    )
    phone = StringField(
        'Số điện thoại',
        validators=[Optional(), Length(max=20)]
    )
    plate = StringField(
        'Biển số xe',
        validators=[
            Optional(),
            Length(max=20),
            Regexp(r'^[A-Z0-9]*$', message='Biển số chỉ chứa chữ cái và số'),
        ]
    )


class VipSlotsForm(FlaskForm):
    """Update VIP slots form."""
    slot_ids = HiddenField('slot_ids')
    slot_numbers = HiddenField('slot_numbers')


class ReservationForm(FlaskForm):
    """Mobile app reservation form (JSON API uses CSRF from app context)."""
    pass


class VehicleForm(FlaskForm):
    """Mobile app vehicle form."""
    plate_text = StringField(
        'Biển số',
        validators=[
            DataRequired(message='Biển số không được để trống'),
            Regexp(r'^[A-Z0-9]*$', message='Biển số chỉ chứa chữ cái và số'),
        ]
    )
