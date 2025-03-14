import React, { useEffect, useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  Grid,
  Typography,
  Tab,
  Tabs,
  MenuItem,
  Select,
  FormControl,
  InputLabel,
} from '@mui/material';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts';
import { useWebSocket } from '../hooks/useWebSocket';
import { useMetricsStore } from '../stores/metricsStore';
import { formatNumber, calculateGrowthRate } from '../utils/metrics';

export interface ViralMetric {
  timestamp: number;
  viralCoefficient: number;
  growthRate: number;
  engagement: number;
  shares: number;
  reach: number;
  platformMetrics: Record<string, PlatformMetrics>;
}

interface PlatformMetrics {
  engagement: number;
  shares: number;
  reach: number;
  virality: number;
}

interface ViralMetricsProps {
  campaignId?: string;
  refreshInterval?: number;
  platforms: string[];
}

export const ViralMetrics: React.FC<ViralMetricsProps> = ({
  campaignId,
  refreshInterval = 5000,
  platforms,
}) => {
  const [selectedTab, setSelectedTab] = useState(0);
  const [timeRange, setTimeRange] = useState('24h');
  const [selectedPlatform, setSelectedPlatform] = useState('all');
  const { metrics, predictions } = useMetricsStore();
  const { lastMessage } = useWebSocket(`/ws/metrics/${campaignId}`);

  useEffect(() => {
    if (lastMessage) {
      const data = JSON.parse(lastMessage.data);
      useMetricsStore.getState().updateMetrics(data);
    }
  }, [lastMessage]);

  const renderGrowthChart = () => (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Growth Trends
        </Typography>
        <ResponsiveContainer width="100%" height={300}>
          <AreaChart data={metrics}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="timestamp" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Area
              type="monotone"
              dataKey="growthRate"
              stroke="#8884d8"
              fill="#8884d8"
              name="Growth Rate"
            />
          </AreaChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );

  const renderViralCoefficients = () => (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Viral Coefficients
        </Typography>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={metrics}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="timestamp" />
            <YAxis />
            <Tooltip />
            <Legend />
            {platforms.map((platform, index) => (
              <Line
                key={platform}
                type="monotone"
                dataKey={`platformMetrics.${platform}.virality`}
                stroke={`#${((index + 1) * 4321) % 16777215}`}
                name={`${platform} Virality`}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );

  const renderEngagementMetrics = () => (
    <Grid container spacing={2}>
      {platforms.map((platform) => (
        <Grid item xs={12} md={6} key={platform}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                {platform} Engagement
              </Typography>
              <Typography variant="h4">
                {formatNumber(metrics[metrics.length - 1]?.platformMetrics[platform]?.engagement)}
              </Typography>
              <Typography variant="body2" color="textSecondary">
                Growth: {calculateGrowthRate(metrics, platform, 'engagement')}%
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      ))}
    </Grid>
  );

  const renderPredictions = () => (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Growth Predictions
        </Typography>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={predictions}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="timestamp" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Line
              type="monotone"
              dataKey="predicted"
              stroke="#82ca9d"
              name="Predicted Growth"
            />
            <Line
              type="monotone"
              dataKey="actual"
              stroke="#8884d8"
              name="Actual Growth"
            />
          </LineChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );

  return (
    <Box sx={{ width: '100%' }}>
      <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 2 }}>
        <Tabs value={selectedTab} onChange={(_, newValue) => setSelectedTab(newValue)}>
          <Tab label="Growth" />
          <Tab label="Virality" />
          <Tab label="Engagement" />
          <Tab label="Predictions" />
        </Tabs>
      </Box>

      <Box sx={{ mb: 2 }}>
        <Grid container spacing={2} alignItems="center">
          <Grid item xs={12} md={6}>
            <FormControl fullWidth>
              <InputLabel>Time Range</InputLabel>
              <Select
                value={timeRange}
                onChange={(e) => setTimeRange(e.target.value as string)}
              >
                <MenuItem value="1h">Last Hour</MenuItem>
                <MenuItem value="24h">Last 24 Hours</MenuItem>
                <MenuItem value="7d">Last 7 Days</MenuItem>
                <MenuItem value="30d">Last 30 Days</MenuItem>
              </Select>
            </FormControl>
          </Grid>
          <Grid item xs={12} md={6}>
            <FormControl fullWidth>
              <InputLabel>Platform</InputLabel>
              <Select
                value={selectedPlatform}
                onChange={(e) => setSelectedPlatform(e.target.value as string)}
              >
                <MenuItem value="all">All Platforms</MenuItem>
                {platforms.map((platform) => (
                  <MenuItem key={platform} value={platform}>
                    {platform}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>
        </Grid>
      </Box>

      <Box>
        {selectedTab === 0 && renderGrowthChart()}
        {selectedTab === 1 && renderViralCoefficients()}
        {selectedTab === 2 && renderEngagementMetrics()}
        {selectedTab === 3 && renderPredictions()}
      </Box>
    </Box>
  );
};

