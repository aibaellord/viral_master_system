import asyncio
import aiohttp
import pytest
import time
from typing import List, Dict, Any
from dataclasses import dataclass
import numpy as np

@dataclass
class APITestMetrics:
    endpoint: str
    response_times: List[float]
    status_codes: List[int]
    errors: List[str]

    def get_summary(self) -> Dict[str, Any]:
        return {
            "endpoint": self.endpoint,
            "response_times": {
                "mean": np.mean(self.response_times),
                "p50": np.percentile(self.response_times, 50),
                "p95": np.percentile(self.response_times, 95),
                "p99": np.percentile(self.response_times, 99)
            },
            "success_rate": sum(1 for code in self.status_codes if 200 <= code < 300) / len(self.status_codes),
            "error_count": len(self.errors)
        }

class APILoadTest:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session = None
        self.metrics: Dict[str, APITestMetrics] = {}

    async def setup(self):
        self.session = aiohttp.ClientSession()

    async def cleanup(self):
        if self.session:
            await self.session.close()

    async def make_request(self, method: str, endpoint: str, data: Dict = None) -> APITestMetrics:
        if endpoint not in self.metrics:
            self.metrics[endpoint] = APITestMetrics(endpoint, [], [], [])
        
        try:
            start_time = time.time()
            async with self.session.request(method, f"{self.base_url}{endpoint}", json=data) as response:
                await response.read()
                response_time = time.time() - start_time
                
                self.metrics[endpoint].response_times.append(response_time)
                self.metrics[endpoint].status_codes.append(response.status)
                
                if response.status >= 400:
                    error_text = await response.text()
                    self.metrics[endpoint].errors.append(error_text)
        except Exception as e:
            self.metrics[endpoint].errors.append(str(e))
            self.metrics[endpoint].status_codes.append(500)

        return self.metrics[endpoint]

@pytest.fixture
async def api_load_test():
    test = APILoadTest("http://localhost:8000")
    await test.setup()
    yield test
    await test.cleanup()

@pytest.mark.asyncio
async def test_campaign_api_load(api_load_test: APILoadTest):
    """Test campaign API endpoints under load"""
    
    async def create_campaign():
        return await api_load_test.make_request("POST", "/api/campaigns", {
            "name": f"Load Test Campaign {time.time()}",
            "platforms": ["instagram", "tiktok", "youtube"],
            "optimization_targets": ["viral_coefficient", "engagement_rate"]
        })

    async def get_campaign(campaign_id: str):
        return await api_load_test.make_request("GET", f"/api/campaigns/{campaign_id}")

    async def update_campaign(campaign_id: str):
        return await api_load_test.make_request("PUT", f"/api/campaigns/{campaign_id}", {
            "optimization_targets": ["shares", "comments", "viral_spread"]
        })

    # Run concurrent API requests
    tasks = []
    for _ in range(100):
        tasks.extend([
            create_campaign(),
            get_campaign("test-campaign-1"),
            update_campaign("test-campaign-1")
        ])
    
    await asyncio.gather(*tasks)
    
    # Assert API performance
    for endpoint, metrics in api_load_test.metrics.items():
        summary = metrics.get_summary()
        assert summary["response_times"]["p95"] < 1.0  # 95% under 1 second
        assert summary["success_rate"] > 0.99          # 99% success rate

@pytest.mark.asyncio
async def test_optimization_api_load(api_load_test: APILoadTest):
    """Test optimization API endpoints under load"""
    
    async def optimize_content(campaign_id: str):
        return await api_load_test.make_request("POST", f"/api/optimization/content", {
            "campaign_id": campaign_id,
            "content_type": "video",
            "platforms": ["tiktok", "instagram"]
        })

    async def get_optimization_status(optimization_id: str):
        return await api_load_test.make_request("GET", f"/api/optimization/{optimization_id}/status")

    # Run concurrent optimization requests
    tasks = []
    for i in range(50):
        campaign_id = f"test-campaign-{i}"
        tasks.extend([
            optimize_content(campaign_id),
            get_optimization_status(f"opt-{i}")
        ])
    
    await asyncio.gather(*tasks)
    
    # Assert optimization API performance
    for endpoint, metrics in api_load_test.metrics.items():
        summary = metrics.get_summary()
        assert summary["response_times"]["p99"] < 2.0  

