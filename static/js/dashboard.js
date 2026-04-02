// Initialize Chart.js defaults for dark theme
Chart.defaults.color = '#94A3B8';
Chart.defaults.font.family = "'Fira Sans', sans-serif";

let parkingChart;

function initChart() {
    const ctx = document.getElementById('parkingChart').getContext('2d');
    
    // Custom plugin for ChartDataLabels will be loaded via script tag in HTML
    parkingChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Đậu Đúng', 'Đậu Lấn Ô', 'Khả Dụng'],
            datasets: [{
                data: [0, 0, 0],
                backgroundColor: [
                    'rgba(34, 197, 94, 0.8)',  // Green (CTA)
                    'rgba(245, 158, 11, 0.8)', // Amber (Warning)
                    'rgba(59, 130, 246, 0.8)'  // Blue (Info)
                ],
                borderColor: [
                    'rgba(34, 197, 94, 1)',
                    'rgba(245, 158, 11, 1)',
                    'rgba(59, 130, 246, 1)'
                ],
                borderWidth: 2,
                hoverOffset: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: '65%',
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        padding: 20,
                        usePointStyle: true,
                        pointStyle: 'circle'
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(15, 23, 42, 0.9)',
                    titleColor: '#F8FAFC',
                    bodyColor: '#F8FAFC',
                    borderColor: 'rgba(255,255,255,0.1)',
                    borderWidth: 1,
                    padding: 12,
                    boxPadding: 6
                },
                datalabels: {
                    color: '#FFF',
                    font: {
                        weight: 'bold',
                        size: 14,
                        family: "'Fira Code', monospace"
                    },
                    formatter: (value) => {
                        return value > 0 ? value : '';
                    }
                }
            }
        },
        plugins: [ChartDataLabels]
    });
}

function initSocketIO() {
    const socket = io.connect(location.protocol + '//' + document.domain + ':' + location.port);
    const statusDot = document.getElementById('wsStatusDot');
    const statusText = document.getElementById('wsStatusText');
    
    socket.on('connect', () => {
        statusDot.className = 'status-dot online';
        statusText.textContent = 'Đã kết nối';
    });
    
    socket.on('disconnect', () => {
        statusDot.className = 'status-dot offline';
        statusText.textContent = 'Mất kết nối';
    });

    socket.on('update_status', function (data) {
        // Update DOM elements using nice animation or direct set
        document.getElementById('total_spaces').innerText = data.total_spaces || 0;
        document.getElementById('occupied_spaces').innerText = data.occupied_spaces || 0;
        document.getElementById('available_spaces').innerText = data.available_spaces || 0;
        document.getElementById('overlapping_vehicles_count').innerText = data.overlapping_vehicles_count || 0;
        document.getElementById('outside_vehicles_count').innerText = data.outside_vehicles_count || 0;

        // Calculate properly parked vehicles
        var properly_parked = (data.occupied_spaces || 0) - (data.overlapping_vehicles_count || 0);

        // Update chart
        if (parkingChart) {
            parkingChart.data.datasets[0].data = [
                properly_parked,
                data.overlapping_vehicles_count || 0,
                data.available_spaces || 0
            ];
            parkingChart.update();
        }
    });
}

// Document Ready
document.addEventListener('DOMContentLoaded', () => {
    // Initialize Lucide Icons
    lucide.createIcons();
    
    // Initialize Chart
    initChart();
    
    // Initialize WebSockets
    initSocketIO();
});
