// VIP Slot Notification System
// Displays popup notification when admin marks VIP slots

class VIPNotification {
    constructor() {
        this.container = null;
        this.init();
    }

    init() {
        // Create container for notifications
        this.container = document.createElement('div');
        this.container.id = 'vip-notification-container';
        this.container.style.cssText = `
            position: fixed;
            bottom: 20px;
            right: 20px;
            z-index: 9999;
            max-width: 400px;
        `;
        document.body.appendChild(this.container);

        // Inject CSS
        this.injectStyles();
    }

    injectStyles() {
        const style = document.createElement('style');
        style.textContent = `
            .vip-notification {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 16px 20px;
                border-radius: 8px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.3);
                margin-bottom: 12px;
                animation: slideIn 0.3s ease-out;
                position: relative;
                overflow: hidden;
            }

            .vip-notification::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                width: 4px;
                height: 100%;
                background: #ffd700;
            }

            .vip-notification-title {
                font-weight: bold;
                font-size: 16px;
                margin-bottom: 8px;
                display: flex;
                align-items: center;
                gap: 8px;
            }

            .vip-notification-message {
                font-size: 14px;
                line-height: 1.5;
                opacity: 0.95;
            }

            .vip-notification-close {
                position: absolute;
                top: 12px;
                right: 12px;
                background: rgba(255,255,255,0.2);
                border: none;
                color: white;
                width: 24px;
                height: 24px;
                border-radius: 50%;
                cursor: pointer;
                font-size: 16px;
                line-height: 1;
                transition: background 0.2s;
            }

            .vip-notification-close:hover {
                background: rgba(255,255,255,0.3);
            }

            @keyframes slideIn {
                from {
                    transform: translateX(400px);
                    opacity: 0;
                }
                to {
                    transform: translateX(0);
                    opacity: 1;
                }
            }

            @keyframes slideOut {
                from {
                    transform: translateX(0);
                    opacity: 1;
                }
                to {
                    transform: translateX(400px);
                    opacity: 0;
                }
            }

            .vip-notification.removing {
                animation: slideOut 0.3s ease-in forwards;
            }

            .vip-icon {
                display: inline-block;
                width: 20px;
                height: 20px;
                background: #ffd700;
                border-radius: 50%;
                text-align: center;
                line-height: 20px;
                color: #667eea;
                font-weight: bold;
            }
        `;
        document.head.appendChild(style);
    }

    show(slotNumbers) {
        if (!Array.isArray(slotNumbers) || slotNumbers.length === 0) return;

        const notification = document.createElement('div');
        notification.className = 'vip-notification';

        const slotList = slotNumbers.map(n => `Slot ${n}`).join(', ');
        const message = slotNumbers.length === 1
            ? `Slot ${slotNumbers[0]} has been marked for VIP guests. Please ensure no one parks there except VIPs.`
            : `Slots ${slotList} have been marked for VIP guests. Please ensure no one parks there except VIPs.`;

        notification.innerHTML = `
            <button class="vip-notification-close" onclick="this.parentElement.remove()">×</button>
            <div class="vip-notification-title">
                <span class="vip-icon">★</span>
                <span>VIP Slot Notification</span>
            </div>
            <div class="vip-notification-message">${message}</div>
        `;

        this.container.appendChild(notification);

        // Persistent notification - does not auto-dismiss
        // User must click close button to dismiss
    }
}

// Initialize and expose globally
window.vipNotification = new VIPNotification();
