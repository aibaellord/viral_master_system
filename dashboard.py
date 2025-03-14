import os
import json
import threading
import importlib
from flask import Flask, render_template, jsonify, request

app = Flask(__name__, 
           template_folder="templates",
           static_folder="static")

# Global reference to system launcher
system = None

@app.route('/')
def index():
    """Main dashboard page."""
    return render_template('index.html')

@app.route('/api/status')
def system_status():
    """Get overall system status."""
    if not system or not system.components:
        return jsonify({"error": "System not initialized"})
        
    status = {
        "components": [comp.get_status() for comp in system.components],
        "gpu": system.gpu_config
    }
    return jsonify(status)

@app.route('/api/start_component', methods=['POST'])
def start_component():
    """Start a specific component."""
    data = request.json
    component_name = data.get('name')
    
    for component in system.components:
        if component.name == component_name and not component.running:
            component.start()
            return jsonify({"status": "started", "name": component_name})
            
    return jsonify({"error": f"Component {component_name} not found or already running"})

@app.route('/api/stop_component', methods=['POST'])
def stop_component():
    """Stop a specific component."""
    data = request.json
    component_name = data.get('name')
    
    for component in system.components:
        if component.name == component_name and component.running:
            component.stop()
            return jsonify({"status": "stopped", "name": component_name})
            
    return jsonify({"error": f"Component {component_name} not found or not running"})

def start_dashboard(launcher):
    """Start the web dashboard with a reference to the system launcher."""
    global system
    system = launcher
    # Create necessary directories for templates and static files
    os.makedirs("templates", exist_ok=True)
    os.makedirs("static", exist_ok=True)
    
    # Create basic HTML template
    with open("templates/index.html", "w") as f:
        f.write("""<!DOCTYPE html>
<html>
<head>
    <title>Viral Master System Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
    <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.5.1/jquery.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.16.0/umd/popper.min.js"></script>
    <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js"></script>
    <style>
        .component-card {
            margin-bottom: 15px;
        }
        .status-running {
            color: green;
        }
        .status-stopped {
            color: red;
        }
    </style>
</head>
<body>
    <nav class="navbar navbar-expand-sm bg-dark navbar-dark">
        <a class="navbar-brand" href="#">Viral Master System</a>
    </nav>
    
    <div class="container mt-4">
        <div class="jumbotron">
            <h1>System Dashboard</h1>
            <p>Monitor and control the Viral Master System components</p>
        </div>
        
        <div class="row">
            <div class="col-md-8">
                <h2>Components</h2>
                <div id="components-container">
                    <p>Loading components...</p>
                </div>
            </div>
            <div class="col-md-4">
                <h2>System Info</h2>
                <div class="card">
                    <div class="card-body">
                        <h5 class="card-title">GPU Information</h5>
                        <div id="gpu-info">
                            <p>Loading GPU information...</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // Function to refresh the system status
        function refreshStatus() {
            $.get('/api/status', function(data) {
                // Update components section
                if (data.error) {
                    $('#components-container').html('<div class="alert alert-danger">' + data.error + '</div>');
                    return;
                }
                
                let componentsHtml = '';
                data.components.forEach(function(component) {
                    let statusClass = component.running ? 'status-running' : 'status-stopped';
                    let statusText = component.running ? 'Running' : 'Stopped';
                    let actionButton = component.running ? 
                        '<button class="btn btn-danger btn-sm stop-btn" data-name="' + component.name + '">Stop</button>' :
                        '<button class="btn btn-success btn-sm start-btn" data-name="' + component.name + '">Start</button>';
                    
                    componentsHtml += `
                        <div class="card component-card">
                            <div class="card-body">
                                <h5 class="card-title">${component.name}</h5>
                                <p class="card-text">Status: <span class="${statusClass}">${statusText}</span></p>
                                <p class="card-text">GPU: ${component.using_gpu ? 'Yes' : 'No'}</p>
                                ${actionButton}
                            </div>
                        </div>
                    `;
                });
                
                $('#components-container').html(componentsHtml);
                
                // Update GPU info section
                let gpuHtml = '';
                if (data.gpu && data.gpu.has_cuda) {
                    gpuHtml += `
                        <p><strong>CUDA Available:</strong> Yes</p>
                        <p><strong>GPU Count:</strong> ${data.gpu.device_count}</p>
                        <p><strong>RTX Optimizations:</strong> ${data.gpu.rtx_specific_optimizations ? 'Enabled' : 'Disabled'}</p>
                    `;
                } else {
                    gpuHtml = '<p>No GPU acceleration available</p>';
                }
                
                $('#gpu-info').html(gpuHtml);
                
                // Add event listeners for start/stop buttons
                $('.start-btn').click(function() {
                    let name = $(this).data('name');
                    $.ajax({
                        url: '/api/start_component',
                        type: 'POST',
                        contentType: 'application/json',
                        data: JSON.stringify({name: name}),
                        success: function() {
                            refreshStatus();
                        }
                    });
                });
                
                $('.stop-btn').click(function() {
                    let name = $(this).data('name');
                    $.ajax({
                        url: '/api/stop_component',
                        type: 'POST',
                        contentType: 'application/json',
                        data: JSON.stringify({name: name}),
                        success: function() {
                            refreshStatus();
                        }
                    });
                });
            });
        }
        
        // Initial refresh
        $(document).ready(function() {
            refreshStatus();
            // Refresh every 5 seconds
            setInterval(refreshStatus, 5000);
        });
    </script>
</body>
</html>""")
    
    # Create a basic CSS file
    with open("static/style.css", "w") as f:
        f.write("""
/* Base styles for the dashboard */
body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background-color: #f5f5f5;
}

.navbar-brand {
    font-weight: bold;
}

.jumbotron {
    background-color: #343a40;
    color: white;
}

.card {
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    transition: 0.3s;
}

.card:hover {
    box-shadow: 0 8px 16px rgba(0,0,0,0.2);
}

.status-running {
    font-weight: bold;
}

.status-stopped {
    font-weight: bold;
}
""")
    
    # Start Flask in a separate thread
    threading.Thread(target=lambda: app.run(host='0.0.0.0', port=5000, debug=False)).start()
    print("Dashboard started at http://0.0.0.0:5000")

if __name__ == "__main__":
    try:
        # Attempt to import and use the system launcher
        from main_launcher import SystemLauncher
        launcher = SystemLauncher()
        launcher.initialize_system()
        launcher.start_system()
        
        # Start the dashboard with the launcher
        start_dashboard(launcher)
    except ImportError:
        print("Warning: Could not import SystemLauncher. Running dashboard in standalone mode.")
        # Start dashboard without system integration
        app.run(host='0.0.0.0', port=5000, debug=True)

