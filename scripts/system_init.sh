#!/bin/bash

# System Initialization Script for Viral Master System
# Sets up and configures all system components

set -e
source "$(dirname "$0")/lib/logging.sh"
source "$(dirname "$0")/lib/validation.sh"

# Configuration
CONFIG_DIR="/etc/viral_master"
DATA_DIR="/var/lib/viral_master"
LOG_DIR="/var/log/viral_master"

# System Dependencies
install_dependencies() {
    log_info "Installing system dependencies..."
    
    # Base Dependencies
    install_base_deps
    
    # Python Dependencies
    install_python_deps
    
    # Node.js Dependencies
    install_node_deps
    
    # Database Dependencies
    install_db_deps
    
    log_info "Dependencies installed successfully"
}

# Configuration Setup
setup_configuration() {
    log_info "Setting up system configuration..."
    
    # Create Config Directory
    setup_config_dir
    
    # Generate Config Files
    generate_configs
    
    # Set Permissions
    set_permissions
    
    # Validate Config
    validate_configs
    
    log_info "Configuration setup completed"
}

# Initialize Services
init_services() {
    log_info "Initializing services..."
    
    # Database Setup
    init_database
    
    # Cache Setup
    init_cache
    
    # Queue Setup
    init_queue
    
    # API Setup
    init_api
    
    log_info "Services initialized successfully"
}

# Security Setup
setup_security() {
    log_info "Configuring security..."
    
    # SSL/TLS Setup
    setup_ssl
    
    # Firewall Rules
    setup_firewall
    
    # Authentication
    setup_auth
    
    # Access Control
    setup_acl
    
    log_info "Security configuration completed"
}

# Main Execution
main() {
    log_info "Starting system initialization..."
    
    # Install Dependencies
    install_dependencies
    
    # Setup Configuration
    setup_configuration
    
    # Initialize Services
    init_services
    
    # Setup Security
    setup_security
    
    # Validate Setup
    validate_system
    
    log_info "System initialization completed successfully"
}

# Execute with Error Handling
if main; then
    log_success "System initialization completed successfully"
    exit 0
else
    log_error "System initialization failed"
    alert_admin "Initialization script failed"
    exit 1
fi

