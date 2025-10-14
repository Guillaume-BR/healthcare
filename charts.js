// Simple chart rendering without external libraries
// This creates basic visualizations using HTML5 Canvas

document.addEventListener('DOMContentLoaded', function() {
    // Data for severity chart
    const severityData = {
        labels: ['Faible', 'Moyen', 'Élevé'],
        values: [1.6, 4.5, 10.8],
        colors: ['#38ef7d', '#ffd93d', '#ff6b6b']
    };
    
    // Data for age distribution
    const ageData = {
        labels: ['<30', '30-50', '50-65', '65+'],
        values: [2.5, 4.0, 5.5, 8.5],
        colors: ['#667eea', '#764ba2', '#f093fb', '#4facfe']
    };
    
    renderBarChart('severityChart', severityData, 'Durée moyenne (jours)');
    renderBarChart('ageChart', ageData, 'Durée moyenne (jours)');
});

function renderBarChart(canvasId, data, yLabel) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    const width = canvas.width = canvas.offsetWidth;
    const height = canvas.height = 300;
    
    // Clear canvas
    ctx.clearRect(0, 0, width, height);
    
    // Chart settings
    const padding = { top: 30, right: 30, bottom: 60, left: 60 };
    const chartWidth = width - padding.left - padding.right;
    const chartHeight = height - padding.top - padding.bottom;
    
    // Calculate max value for scaling
    const maxValue = Math.max(...data.values);
    const scale = chartHeight / (maxValue * 1.1); // 1.1 for some top padding
    
    // Draw bars
    const barWidth = chartWidth / data.values.length * 0.7;
    const barSpacing = chartWidth / data.values.length;
    
    data.values.forEach((value, index) => {
        const barHeight = value * scale;
        const x = padding.left + (index * barSpacing) + (barSpacing - barWidth) / 2;
        const y = padding.top + chartHeight - barHeight;
        
        // Draw bar with gradient
        const gradient = ctx.createLinearGradient(x, y, x, y + barHeight);
        gradient.addColorStop(0, data.colors[index]);
        gradient.addColorStop(1, adjustBrightness(data.colors[index], -20));
        
        ctx.fillStyle = gradient;
        ctx.fillRect(x, y, barWidth, barHeight);
        
        // Draw value on top of bar
        ctx.fillStyle = '#333';
        ctx.font = 'bold 14px Arial';
        ctx.textAlign = 'center';
        ctx.fillText(value.toFixed(1), x + barWidth / 2, y - 5);
        
        // Draw label below bar
        ctx.fillStyle = '#666';
        ctx.font = '12px Arial';
        ctx.fillText(data.labels[index], x + barWidth / 2, padding.top + chartHeight + 20);
    });
    
    // Draw axes
    ctx.strokeStyle = '#ccc';
    ctx.lineWidth = 2;
    
    // Y-axis
    ctx.beginPath();
    ctx.moveTo(padding.left, padding.top);
    ctx.lineTo(padding.left, padding.top + chartHeight);
    ctx.stroke();
    
    // X-axis
    ctx.beginPath();
    ctx.moveTo(padding.left, padding.top + chartHeight);
    ctx.lineTo(padding.left + chartWidth, padding.top + chartHeight);
    ctx.stroke();
    
    // Y-axis label
    ctx.fillStyle = '#666';
    ctx.font = '12px Arial';
    ctx.textAlign = 'center';
    ctx.save();
    ctx.translate(15, padding.top + chartHeight / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText(yLabel, 0, 0);
    ctx.restore();
    
    // Y-axis ticks
    const numTicks = 5;
    for (let i = 0; i <= numTicks; i++) {
        const tickValue = (maxValue * 1.1 / numTicks) * i;
        const y = padding.top + chartHeight - (tickValue * scale);
        
        ctx.fillStyle = '#666';
        ctx.font = '11px Arial';
        ctx.textAlign = 'right';
        ctx.fillText(tickValue.toFixed(1), padding.left - 10, y + 4);
        
        // Tick mark
        ctx.strokeStyle = '#ddd';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(padding.left - 5, y);
        ctx.lineTo(padding.left, y);
        ctx.stroke();
        
        // Grid line
        ctx.strokeStyle = '#f0f0f0';
        ctx.beginPath();
        ctx.moveTo(padding.left, y);
        ctx.lineTo(padding.left + chartWidth, y);
        ctx.stroke();
    }
}

function adjustBrightness(color, percent) {
    // Simple color adjustment
    const num = parseInt(color.replace('#', ''), 16);
    const amt = Math.round(2.55 * percent);
    const R = (num >> 16) + amt;
    const G = (num >> 8 & 0x00FF) + amt;
    const B = (num & 0x0000FF) + amt;
    
    return '#' + (0x1000000 + 
        (R < 255 ? R < 1 ? 0 : R : 255) * 0x10000 +
        (G < 255 ? G < 1 ? 0 : G : 255) * 0x100 +
        (B < 255 ? B < 1 ? 0 : B : 255))
        .toString(16).slice(1);
}

// Responsive resize
window.addEventListener('resize', function() {
    const severityData = {
        labels: ['Faible', 'Moyen', 'Élevé'],
        values: [1.6, 4.5, 10.8],
        colors: ['#38ef7d', '#ffd93d', '#ff6b6b']
    };
    
    const ageData = {
        labels: ['<30', '30-50', '50-65', '65+'],
        values: [2.5, 4.0, 5.5, 8.5],
        colors: ['#667eea', '#764ba2', '#f093fb', '#4facfe']
    };
    
    renderBarChart('severityChart', severityData, 'Durée moyenne (jours)');
    renderBarChart('ageChart', ageData, 'Durée moyenne (jours)');
});
