import asyncio
import logging
from logging.handlers import RotatingFileHandler
import sys
from typing import Dict, Optional
import aiohttp
import psutil
import yaml
from prometheus_client import start_http_server

class EngineCore:
    def __init__(self):
        self.logger = self._setup_logging()
        self.config = self._load_config()
        self.connection_pools: Dict[str, aiohttp.ClientSession] = {}
        self.health_checks = {}
        self.event_loop = None

    def _setup_logging(self) -> logging.Logger:
        """Configure comprehensive logging system"""
        logger = logging.getLogger('viral_engine')
        logger.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        
        # File handler with rotation
        file_handler = RotatingFileHandler(
            'viral_engine.log', maxBytes=10485760, backupCount=5
        )
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        return logger

    def _load_config(self) -> dict:
        """Load and validate system configuration"""
        try:
            with open('config/engine_config.yaml', 'r') as f:
                config = yaml.safe_load(f)
            self._validate_config(config)
            return config
        except Exception as e:
            self.logger.critical(f"Failed to load configuration: {str(e)}")
            raise

    async def initialize(self):
        """Initialize core engine components"""
        try:
            self.event_loop = asyncio.get_event_loop()
            await self._setup_connection_pools()
            await self._initialize_monitoring()
            await self._setup_health_checks()
            start_http_server(8000)  # Prometheus metrics
            self.logger.info("Engine core initialized successfully")
        except Exception as e:
            self.logger.critical(f"Failed to initialize engine: {str(e)}")
            raise

    async def _setup_connection_pools(self):
        """Initialize and configure connection pools"""
        for service, config in self.config['services'].items():
            self.connection_pools[service] = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
                connector=aiohttp.TCPConnector(
                    limit=config.get('connection_limit', 100),
                    enable_cleanup_closed=True
                )
            )

    async def _initialize_monitoring(self):
        """Setup system monitoring"""
        self.monitoring = {
            'cpu_usage': psutil.cpu_percent(),
            'memory_usage': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent,
            'network_connections': len(psutil.net_connections())
        }

    async def _setup_health_checks(self):
        """Configure system health checks"""
        self.health_checks = {
            'database': self._check_database_health,
            'api_services': self._check_api_health,
            'resource_usage': self._check_resource_usage
        }
        
        # Start health check loop
        asyncio.create_task(self._run_health_checks())

    async def _run_health_checks(self):
        """Execute periodic health checks"""
        while True:
            try:
                results = await asyncio.gather(
                    *[check() for check in self.health_checks.values()],
                    return_exceptions=True
                )
                self._process_health_results(results)
                await asyncio.sleep(60)  # Check every minute
            except Exception as e:
                self.logger.error(f"Health check failed: {str(e)}")

    async def shutdown(self):
        """Graceful system shutdown"""
        self.logger.info("Initiating engine shutdown")
        for session in self.connection_pools.values():
            await session.close()
        self.logger.info("Engine shutdown complete")

