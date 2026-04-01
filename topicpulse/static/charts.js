document.addEventListener('DOMContentLoaded', function() {
    const dataElement = document.getElementById('chartData');
    if (!dataElement) return;

    try {
        const data = JSON.parse(dataElement.textContent);
        const ctx = document.getElementById('sentimentChart').getContext('2d');
        
        new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Positive', 'Negative', 'Neutral'],
                datasets: [{
                    data: [data.positive, data.negative, data.neutral],
                    backgroundColor: [
                        '#10b981', // green
                        '#ef4444', // red
                        '#94a3b8'  // grey
                    ],
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                cutout: '70%',
                plugins: {
                    legend: {
                        position: 'right',
                        labels: { boxWidth: 12 }
                    }
                }
            }
        });
    } catch(e) {
        console.error("Error creating chart", e);
    }
});
