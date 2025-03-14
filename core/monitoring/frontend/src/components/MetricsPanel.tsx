import React, { useState, useEffect, useCallback } from 'react';
import {
  Grid,
  Paper,
  Typography,
  Box,
  CircularProgress,
  Button,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Snackbar,
  Alert,
} from '@mui/material';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  AreaChart,
  Area,
} from 'recharts';
import { useWebSocket } from '../hooks/useWebSocket';
import { useMetrics } from '../hooks/useMetrics';
import { MetricsService } from '../services/MetricsService';
import { ExportService } from '../services/ExportService';

// Types
interface MetricData {
  timestamp: number;
  value: number;
  platform?: string;
  category?: string;
}

interface MetricsState {
  performanceKPIs: MetricData[];
  systemMetrics: MetricData[];
  viralMetrics: MetricData[];
  platformMetrics: MetricData[];
  businessMetrics: MetricData[];
}

interface Alert {
  message: string;
  severity: 'error' | 'warning' | 'info' | 'success';
}

// Component
export const MetricsPanel: React.FC = () => {
  // State
  const [timeRange, setTimeRange] = useState<string>('1h');
  const [selectedMetrics, setSelectedMetrics] = useState<string[]>([]);
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [metrics, setMetrics] = useState<MetricsState>({
    performanceKPIs: [],
    systemMetrics: [],
    viralMetrics: [],
    platformMetrics: [],
    businessMetrics: [],
  });

  // Hooks
  const { data: wsData, error: wsError } = useWebSocket('/metrics');
  const { fetchMetrics, isLoading: isFetching } = useMetrics();

  // Effects
  useEffect(() => {
    loadMetrics();
  }, [timeRange]);

  useEffect(() => {
    if (wsData) {
      updateMetrics(wsData);
    }
  }, [wsData]);

  useEffect(() => {
    if (wsError) {
      setAlerts([...alerts, { message: 'WebSocket error: ' + wsError, severity: 'error' }]);
    }
  }, [wsError]);

  // Methods
  const loadMetrics = async () => {
    setIsLoading(true);
    try {
      const data = await fetchMetrics(timeRange);
      setMetrics(data);
    } catch (error) {
      setAlerts([...alerts, { message: 'Failed to load metrics: ' + error, severity: 'error' }]);
    } finally {
      setIsLoading(false);
    }
  };

  const updateMetrics = (newData: any) => {
    setMetrics(prev => ({
      ...prev,
      ...newData,
    }));
  };

  const handleExport = async () => {
    try {
      await ExportService.exportMetrics(metrics, selectedMetrics);
      setAlerts([...alerts, { message: 'Metrics exported successfully', severity: 'success' }]);
    } catch (error) {
      setAlerts([...alerts, { message: 'Export failed: ' + error, severity: 'error' }]);
    }
  };

  const handleTimeRangeChange = (event: React.ChangeEvent<{ value: unknown }>) => {
    setTimeRange(event.target.value as string);
  };

  // Render methods
  const renderMetricChart = (data: MetricData[], title: string) => (
    <Box p={2}>
      <Typography variant="h6" gutterBottom>
        {title}
      </Typography>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis
            dataKey="timestamp"
            tickFormatter={(timestamp) => new Date(timestamp).toLocaleTimeString()}
          />
          <YAxis />
          <Tooltip
            labelFormatter={(timestamp) => new Date(timestamp).toLocaleString()}
          />
          <Legend />
          <Line type="monotone" dataKey="value" stroke="#8884d8" />
        </LineChart>
      </ResponsiveContainer>
    </Box>
  );

  return (
    <Paper>
      <Box p={3}>
        <Grid container spacing={3}>
          {/* Controls */}
          <Grid item xs={12}>
            <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
              <FormControl variant="outlined" style={{ minWidth: 120 }}>
                <InputLabel>Time Range</InputLabel>
                <Select
                  value={timeRange}
                  onChange={handleTimeRangeChange}
                  label="Time Range"
                >
                  <MenuItem value="1h">Last Hour</MenuItem>
                  <MenuItem value="24h">Last 24 Hours</MenuItem>
                  <MenuItem value="7d">Last 7 Days</MenuItem>
                  <MenuItem value="30d">Last 30 Days</MenuItem>
                </Select>
              </FormControl>
              <Button
                variant="contained"
                color="primary"
                onClick={handleExport}
                disabled={isLoading}
              >
                Export Metrics
              </Button>
            </Box>
          </Grid>

          {/* Metric Charts */}
          {isLoading ? (
            <Grid item xs={12} display="flex" justifyContent="center">
              <CircularProgress />
            </Grid>
          ) : (
            <>
              <Grid item xs={12} md={6}>
                {renderMetricChart(metrics.performanceKPIs, 'Performance KPIs')}
              </Grid>
              <Grid item xs={12} md={6}>
                {renderMetricChart(metrics.systemMetrics, 'System Metrics')}
              </Grid>
              <Grid item xs={12} md={6}>
                {renderMetricChart(metrics.viralMetrics, 'Viral Metrics')}
              </Grid>
              <Grid item xs={12} md={6}>
                {renderMetricChart(metrics.platformMetrics, 'Platform Metrics')}
              </Grid>
              <Grid item xs={12}>
                {renderMetricChart(metrics.businessMetrics, 'Business Metrics')}
              </Grid>
            </>
          )}
        </Grid>

        {/* Alerts */}
        <Snackbar
          open={alerts.length > 0}
          autoHideDuration={6000}
          onClose={() => setAlerts(alerts.slice(1))}
        >
          <Alert
            severity={alerts[0]?.severity}
            onClose={() => setAlerts(alerts.slice(1))}
          >
            {alerts[0]?.message}
          </Alert>
        </Snackbar>
      </Box>
    </Paper>
  );
};

