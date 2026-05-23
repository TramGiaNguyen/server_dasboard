import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

time = np.arange(0, 60, 0.5)

# ===== GIAI DOAN 1: Dao động không có hysteresis =====
# Detection signal (bị nhiễu)
detection_no_hyst = []
status_no_hyst = []
current = 1
for t in time:
    # Giả lập: xen kẽ có/không xe mỗi 3-5 giây
    cycle = int(t / 4) % 2
    detection_no_hyst.append(cycle)
    # Không có hysteresis -> nhảy ngay
    status_no_hyst.append(cycle)

ax1.step(time, detection_no_hyst, 'gray', where='post', linewidth=1, label='Detection Signal', alpha=0.5)
ax1.step(time, status_no_hyst, 'r-', where='post', linewidth=2, label='Status Output')
ax1.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Threshold=0.5')
ax1.fill_between(time, status_no_hyst, alpha=0.2, step='post')
ax1.set_ylabel('Trạng thái', fontsize=11)
ax1.set_title('KHÔNG có Hysteresis: Dao động trạng thái', fontsize=12, color='red')
ax1.set_yticks([0, 1])
ax1.set_yticklabels(['KhongXe', 'CoXe'])
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)
ax1.set_ylim(-0.2, 1.3)

# Annotate oscillation
for i, t in enumerate(time):
    if i > 0 and status_no_hyst[i] != status_no_hyst[i-1]:
        ax1.annotate('!', (t, 0.5), fontsize=14, color='red', ha='center')
        ax1.axvline(x=t, color='red', linestyle=':', alpha=0.3)

# ===== GIAI DOAN 2: Có hysteresis =====
# Counter-based với threshold 45 frames
THRESHOLD = 45
H_FRAME = 0.5  # mỗi frame = 0.5s

detection_hyst = []
status_hyst = []
counter = 0
current_state = 1
for t in time:
    cycle = int(t / 4) % 2
    detection_hyst.append(cycle)
    
    if cycle == current_state:
        counter = min(counter + 1, THRESHOLD)
    else:
        counter = max(counter - 1, 0)
    
    if current_state == 0 and counter >= THRESHOLD:
        current_state = 1
        counter = 0
    elif current_state == 1 and counter >= THRESHOLD:
        current_state = 0
        counter = 0
    
    status_hyst.append(current_state)

ax2.step(time, detection_hyst, 'gray', where='post', linewidth=1, label='Detection Signal', alpha=0.5)
ax2.step(time, status_hyst, 'g-', where='post', linewidth=2, label='Status Output')
ax2.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label=f'Hysteresis Zone (Δ={THRESHOLD} frames)')
ax2.fill_between(time, status_hyst, alpha=0.2, step='post', color='green')
ax2.set_xlabel('Thời gian (giây)', fontsize=11)
ax2.set_ylabel('Trạng thái', fontsize=11)
ax2.set_title('CÓ Hysteresis: Trạng thái ổn định', fontsize=12, color='green')
ax2.set_yticks([0, 1])
ax2.set_yticklabels(['KhongXe', 'CoXe'])
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)
ax2.set_ylim(-0.2, 1.3)

# Highlight stable period
ax2.annotate('ỔN ĐỊNH', xy=(25, 0.5), fontsize=12, color='green', ha='center',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

# Vẽ counter bar ở giữa hai subplot
ax_counter = ax2.twinx()
counter_vals = []
c = 0
cs = 1
for t in time:
    cycle = int(t / 4) % 2
    if cycle == cs:
        c = min(c + 1, THRESHOLD)
    else:
        c = max(c - 1, 0)
    if cs == 0 and c >= THRESHOLD:
        cs = 1; c = 0
    elif cs == 1 and c >= THRESHOLD:
        cs = 0; c = 0
    counter_vals.append(c)

ax_counter.fill_between(time, counter_vals, alpha=0.15, color='blue', step='post')
ax_counter.step(time, counter_vals, 'b-', where='post', linewidth=1, label='Hysteresis Counter')
ax_counter.axhline(y=THRESHOLD, color='blue', linestyle='--', linewidth=1.5, label=f'Threshold={THRESHOLD}')
ax_counter.set_ylabel('Counter (frames)', fontsize=10, color='blue')
ax_counter.set_ylim(0, THRESHOLD * 1.5)
ax_counter.legend(loc='center right')

plt.suptitle('Hình 8: Ổn định theo thời gian bằng Hysteresis nhằm hạn chế dao động trạng thái bãi đỗ', 
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('hinh8_hysteresis_parking.png', dpi=150, bbox_inches='tight')
plt.show()