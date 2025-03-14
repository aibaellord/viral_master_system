import React, { useState, useEffect } from 'react';
import Plot from 'react-plotly.js';
import {
  Grid,
  Paper,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Button,
  Typography,
  Box,
  CircularProgress,
} from '@mui/material';
import { DateRangePicker } from '@mui/lab';
import { format } from 'date-fns';

interface DashboardProps {
  refreshInterval?: number;
  defaultPlatform?: string;
  defaultMetricType?: string;
}

interface MetricData {
  figure: any;
  loading: boolean;
  error: string | null;
}

const Dashboard: React.FC<DashboardProps> = ({
  refreshInterval = 60,
  defaultPlatform = 'instagram',
  defaultMetricType = 'engagement',
}) => {
  const [platform, setPlatform] = useState(defaultPlatform);
  const [metricType, setMetricType] = useState(defaultMetricType);
  const [dateRange, setDateRange] = useState([
    new Date(Date.now() - 7 * 24 * 60 * 60 * 1000),
    new Date(),
  ]);
  const [metrics, setMetrics] = useState<MetricData>({
    figure: null,
    loading: true,
    error: null,
  });
  const [viralPotential, setViralPotential] = useState<MetricData>({
    figure: null,
    loading: true,
    error: null,
  });
  const [abTestResults, setAbTestResults] = useState<MetricData>({
    figure: null,
    loading: true,
    error: null,
  });

  const fetchMetrics = async () => {
    try {
      setMetrics(prev => ({ ...prev, loading: true, error: null }));
      const response = await fetch(
        `/api/metrics/${platform}/${metricType}?` +
        `start_date=${format(dateRange[0], 'yyyy-MM-dd')}` +
        `&end_date=${format(dateRange[1], 'yyyy-MM-dd')}`
      );
      if (!response.ok) throw new Error('Failed to fetch metrics');
      const data = await response.json();
      setMetrics({ figure: data, loading: false, error: null });
    } catch (error) {
      setMetrics(prev => ({
        ...prev,
        loading: false,
        error: error.message,
      }));
    }
  };

  const fetchViralPotential = async () => {
    try {
      setViralPotential(prev => ({ ...prev, loading: true, error: null }));
      const response = await fetch(`/api/viral-potential/latest`);
      if (!response.ok) throw new Error('Failed to fetch viral potential');
      const data = await response.json();
      setViralPotential({ figure: data, loading: false, error: null });
    } catch (error) {
      setViralPotential(prev => ({
        ...prev,
        loading: false,
        error: error.message,
      }));
    }
  };

  const fetchAbTestResults = async () => {
    try {
      setAbTestResults(prev => ({ ...prev, loading: true, error: null }));
      const response = await fetch(`/api/ab-test/latest`);
      if (!response.ok) throw new Error('Failed to fetch A/B test results');
      const data = await response.json();
      setAbTestResults({ figure: data, loading: false, error: null });
    } catch (error) {
      setAbTestResults(prev => ({
        ...prev,
        loading: false,
        error: error.message,
      }));
    }
  };

  const handleExport = async (format: 'csv' | 'json') => {
    try {
      const response = await fetch('/api/export', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          platform,
          metric_type: metricType,
          start_date: format(dateRange[0], 'yyyy-MM-dd'),
          end_date: format(dateRange[1], 'yyyy-MM-dd'),
          format,
        }),
      });
      if (!response.ok) throw new Error('Export failed');
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `metrics_export.${format}`;
      a.click();
    } catch (error) {
      console.error('Export failed:', error);
    }
  };

  useEffect(() => {
    fetchMetrics();
    fetchViralPotential();
    fetchAbTestResults();

    const interval = setInterval(() => {
      fetchMetrics();
      fetchViralPotential();
      fetchAbTestResults();
    }, refreshInterval * 1000);

    return () => clearInterval(interval);
  }, [platform, metricType, dateRange]);

  return (
    <Box sx={{ flexGrow: 1, p: 3 }}>
      <Grid container spacing={3}>
        {/* Controls */}
        <Grid item xs={12}>
          <Paper sx={{ p: 2 }}>
            <Grid container spacing={2} alignItems="center">
              <Grid item>
                <FormControl>
                  <InputLabel>Platform</InputLabel>
                  <Select
                    value={platform}
                    onChange={(e) => setPlatform(e.target.value)}
                  >
                    <MenuItem value="instagram">Instagram</MenuItem>
                    <MenuItem value="tiktok">TikTok</MenuItem>
                    <MenuItem value="youtube">YouTube</MenuItem>
                  </Select>
                </FormControl>
              </Grid>
              <Grid item>
                <FormControl>
                  <InputLabel>Metric Type</InputLabel>
                  <Select
                    value={metricType}
                    onChange={(e) => setMetricType(e.target.value)}
                  >
                    <MenuItem value="engagement">Engagement</MenuItem>
                    <MenuItem value="reach">Reach</MenuItem>
                    <MenuItem value="conversion">Conversion</MenuItem>
                  </Select>
                </FormControl>
              </Grid>
              <Grid item>
                <DateRangePicker
                  value={dateRange}
                  onChange={(newValue) => setDateRange(newValue)}
                />
              </Grid>
              <Grid item>
                <Button
                  variant="contained"
                  onClick={() => handleExport('csv')}
                >
                  Export CSV
                </Button>
              </Grid>
              <Grid item>
                <Button
                  variant="contained"

