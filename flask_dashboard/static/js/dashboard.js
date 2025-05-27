/**
 * Dashboard JavaScript for HeteroFL Bidirectional System
 * Handles real-time updates, visualizations, and user interactions
 */

// Initialize Socket.IO connection
const socket = io();

// Global variables
let currentStatus = 'idle';
let currentRound = 0;
let totalRounds = 0;
let modelEvolutionChart = null;
let globalEvolutionChart = null;
let classDistributionChart = null;

// DOM elements
const statusIndicator = document.getElementById('statusIndicator');
const statusText = document.getElementById('statusText');
const progressCircle = document.getElementById('progressCircle');
const progressText = document.getElementById('progressText');
const currentRoundEl = document.getElementById('currentRound');
const totalRoundsEl = document.getElementById('totalRounds');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const roundsInput = document.getElementById('roundsInput');
const systemLogs = document.getElementById('systemLogs');
const modelStatus = document.getElementById('modelStatus');

// Metric elements
const globalAccuracy = document.getElementById('globalAccuracy');
const globalF1 = document.getElementById('globalF1');
const modelsActive = document.getElementById('modelsActive');
const avgConfidence = document.getElementById('avgConfidence');

// Initialize dashboard
document.addEventListener('DOMContentLoaded', function() {
    initializeDashboard();
    setupEventListeners();
    updateCurrentTime();
    setInterval(updateCurrentTime, 1000);
});

function initializeDashboard() {
    addLog('Dashboard initialized', 'info');
    
    // Initialize empty plots
    initializeModelEvolutionPlot();
    initializeGlobalEvolutionPlot();
    initializeClassDistributionPlot();
    
    // Request initial status
    socket.emit('request_metrics');
}

function setupEventListeners() {
    // Control buttons
    startBtn.addEventListener('click', startFLProcess);
    stopBtn.addEventListener('click', stopFLProcess);
    
    // Export buttons
    document.getElementById('exportMetricsBtn').addEventListener('click', exportMetrics);
    document.getElementById('exportModelsBtn').addEventListener('click', exportModels);
    
    // Refresh buttons
    document.getElementById('refreshModelEvolution').addEventListener('click', refreshModelEvolution);
    document.getElementById('refreshGlobalEvolution').addEventListener('click', refreshGlobalEvolution);
    document.getElementById('refreshClassDistribution').addEventListener('click', refreshClassDistribution);
    
    // Clear logs button
    document.getElementById('clearLogs').addEventListener('click', clearLogs);
}

// Socket.IO event handlers
socket.on('connect', function() {
    addLog('Connected to server', 'info');
});

socket.on('disconnect', function() {
    addLog('Disconnected from server', 'warning');
});

socket.on('status_update', function(data) {
    updateDashboardStatus(data);
});

socket.on('round_start', function(data) {
    addLog(`Round ${data.round} started`, 'info');
    addFlowAnimation();
});

socket.on('round_complete', function(data) {
    if (data.success) {
        addLog(`Round ${data.round} completed successfully`, 'info');
    } else {
        addLog(`Round ${data.round} failed`, 'error');
    }
});

socket.on('metrics_update', function(data) {
    updateMetrics(data);
    refreshAllPlots();
});

socket.on('fl_complete', function(data) {
    addLog(`Federated learning completed! Total rounds: ${data.total_rounds}`, 'info');
    showCompletionAnimation();
});

socket.on('fl_error', function(data) {
    addLog(`FL Error: ${data.error}`, 'error');
});

// Dashboard update functions
function updateDashboardStatus(data) {
    currentStatus = data.fl_status;
    currentRound = data.current_round;
    totalRounds = data.total_rounds;
    
    // Update status indicator
    statusIndicator.className = `status-indicator status-${currentStatus}`;
    statusText.textContent = currentStatus.charAt(0).toUpperCase() + currentStatus.slice(1);
    
    // Update progress
    const progress = totalRounds > 0 ? (currentRound / totalRounds) * 100 : 0;
    updateProgressRing(progress);
    
    // Update round counters
    currentRoundEl.textContent = currentRound;
    totalRoundsEl.textContent = totalRounds;
    
    // Update button states
    if (currentStatus === 'running') {
        startBtn.disabled = true;
        stopBtn.disabled = false;
        roundsInput.disabled = true;
    } else {
        startBtn.disabled = false;
        stopBtn.disabled = true;
        roundsInput.disabled = false;
    }
    
    // Update model status
    if (data.models_loaded && data.models_loaded.length > 0) {
        updateModelStatus(data.models_loaded);
    }
}

function updateProgressRing(progress) {
    const circumference = 2 * Math.PI * 45;
    const offset = circumference - (progress / 100) * circumference;
    
    progressCircle.style.strokeDashoffset = offset;
    progressText.textContent = `${Math.round(progress)}%`;
}

function updateModelStatus(models) {
    modelStatus.innerHTML = '';
    
    models.forEach(modelName => {
        const modelDiv = document.createElement('div');
        modelDiv.className = 'model-status';
        modelDiv.innerHTML = `
            <i class="fas fa-circle text-success me-2"></i>
            <span>${modelName}</span>
        `;
        modelStatus.appendChild(modelDiv);
    });
    
    modelsActive.textContent = models.length;
}

function updateMetrics(data) {
    if (data.global_model) {
        globalAccuracy.textContent = data.global_model.accuracy ? 
            (data.global_model.accuracy * 100).toFixed(1) + '%' : '--';
        globalF1.textContent = data.global_model.f1 ? 
            (data.global_model.f1 * 100).toFixed(1) + '%' : '--';
    }
    
    if (data.local_models) {
        const confidences = Object.values(data.local_models)
            .map(model => model.mean_confidence)
            .filter(conf => conf !== undefined);
        
        if (confidences.length > 0) {
            const avgConf = confidences.reduce((a, b) => a + b, 0) / confidences.length;
            avgConfidence.textContent = (avgConf * 100).toFixed(1) + '%';
        }
    }
}

// Control functions
function startFLProcess() {
    const numRounds = parseInt(roundsInput.value);
    
    if (numRounds < 1 || numRounds > 20) {
        alert('Please enter a valid number of rounds (1-20)');
        return;
    }
    
    addLog(`Starting FL process with ${numRounds} rounds`, 'info');
    
    fetch('/api/control/start', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            num_rounds: numRounds
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            addLog(`Error: ${data.error}`, 'error');
        } else {
            addLog(data.message, 'info');
        }
    })
    .catch(error => {
        addLog(`Error starting FL process: ${error}`, 'error');
    });
}

function stopFLProcess() {
    addLog('Stopping FL process', 'warning');
    
    fetch('/api/control/stop', {
        method: 'POST'
    })
    .then(response => response.json())
    .then(data => {
        addLog(data.message, 'info');
    })
    .catch(error => {
        addLog(`Error stopping FL process: ${error}`, 'error');
    });
}

function exportMetrics() {
    addLog('Exporting metrics...', 'info');
    window.open('/api/export/metrics', '_blank');
}

function exportModels() {
    addLog('Exporting models...', 'info');
    window.open('/api/export/models', '_blank');
}

// Plotting functions
function initializeModelEvolutionPlot() {
    const layout = {
        title: 'Model Evolution: Accuracy Across Rounds',
        xaxis: { title: 'Round Number' },
        yaxis: { title: 'Accuracy' },
        showlegend: true,
        responsive: true
    };
    
    Plotly.newPlot('modelEvolutionPlot', [], layout);
}

function initializeGlobalEvolutionPlot() {
    const layout = {
        title: 'Global Model Performance Evolution',
        xaxis: { title: 'Round Number' },
        yaxis: { title: 'Score' },
        showlegend: true,
        responsive: true
    };
    
    Plotly.newPlot('globalEvolutionPlot', [], layout);
}

function initializeClassDistributionPlot() {
    const layout = {
        title: 'Class Distribution Heatmap',
        xaxis: { title: 'Classes' },
        yaxis: { title: 'Models' },
        responsive: true
    };
    
    Plotly.newPlot('classDistributionPlot', [], layout);
}

function refreshModelEvolution() {
    fetch('/api/plots/model_evolution')
        .then(response => response.json())
        .then(plotData => {
            if (plotData.error) {
                addLog(`Error loading model evolution plot: ${plotData.error}`, 'error');
                return;
            }
            
            const fig = JSON.parse(plotData);
            Plotly.react('modelEvolutionPlot', fig.data, fig.layout);
        })
        .catch(error => {
            addLog(`Error refreshing model evolution plot: ${error}`, 'error');
        });
}

function refreshGlobalEvolution() {
    fetch('/api/plots/global_evolution')
        .then(response => response.json())
        .then(plotData => {
            if (plotData.error) {
                addLog(`Error loading global evolution plot: ${plotData.error}`, 'error');
                return;
            }
            
            const fig = JSON.parse(plotData);
            Plotly.react('globalEvolutionPlot', fig.data, fig.layout);
        })
        .catch(error => {
            addLog(`Error refreshing global evolution plot: ${error}`, 'error');
        });
}

function refreshClassDistribution() {
    fetch('/api/plots/class_distribution')
        .then(response => response.json())
        .then(plotData => {
            if (plotData.error) {
                addLog(`Error loading class distribution plot: ${plotData.error}`, 'error');
                return;
            }
            
            const fig = JSON.parse(plotData);
            Plotly.react('classDistributionPlot', fig.data, fig.layout);
        })
        .catch(error => {
            addLog(`Error refreshing class distribution plot: ${error}`, 'error');
        });
}

function refreshAllPlots() {
    // Only refresh if the tab is visible to avoid unnecessary API calls
    const activeTab = document.querySelector('.nav-link.active');
    
    if (activeTab) {
        const tabId = activeTab.getAttribute('data-bs-target');
        
        switch (tabId) {
            case '#model-evolution':
                refreshModelEvolution();
                break;
            case '#global-evolution':
                refreshGlobalEvolution();
                break;
            case '#class-distribution':
                refreshClassDistribution();
                break;
        }
    }
}

// Utility functions
function addLog(message, type = 'info') {
    const timestamp = new Date().toLocaleTimeString();
    const logEntry = document.createElement('div');
    logEntry.className = `log-entry log-${type}`;
    logEntry.textContent = `[${timestamp}] [${type.toUpperCase()}] ${message}`;
    
    systemLogs.appendChild(logEntry);
    systemLogs.scrollTop = systemLogs.scrollHeight;
    
    // Limit log entries to prevent memory issues
    const logEntries = systemLogs.children;
    if (logEntries.length > 100) {
        systemLogs.removeChild(logEntries[0]);
    }
}

function clearLogs() {
    systemLogs.innerHTML = '';
    addLog('Logs cleared', 'info');
}

function updateCurrentTime() {
    const now = new Date();
    document.getElementById('currentTime').textContent = now.toLocaleString();
}

function addFlowAnimation() {
    // Add visual flow animation between models and global model
    const flowElements = document.querySelectorAll('.flow-animation');
    flowElements.forEach(element => {
        const arrow = document.createElement('div');
        arrow.className = 'flow-arrow';
        element.appendChild(arrow);
        
        setTimeout(() => {
            if (element.contains(arrow)) {
                element.removeChild(arrow);
            }
        }, 2000);
    });
}

function showCompletionAnimation() {
    // Show completion animation
    const statusCard = document.querySelector('.status-card');
    statusCard.style.animation = 'pulse 0.5s ease-in-out 3';
    
    setTimeout(() => {
        statusCard.style.animation = '';
    }, 1500);
}

// Tab change event listener to refresh plots when switching tabs
document.addEventListener('shown.bs.tab', function(event) {
    const targetTab = event.target.getAttribute('data-bs-target');
    
    // Small delay to ensure tab content is visible
    setTimeout(() => {
        switch (targetTab) {
            case '#model-evolution':
                refreshModelEvolution();
                break;
            case '#global-evolution':
                refreshGlobalEvolution();
                break;
            case '#class-distribution':
                refreshClassDistribution();
                break;
        }
    }, 100);
});

// Auto-refresh plots every 30 seconds when FL is running
setInterval(() => {
    if (currentStatus === 'running') {
        refreshAllPlots();
    }
}, 30000);