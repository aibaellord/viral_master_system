import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  CircularProgress,
  Tabs,
  Tab,
  Paper,
  Tooltip,
} from '@mui/material';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  Legend,
  ResponsiveContainer,
  BarChart,
  Bar,
} from 'recharts';
import { useWebSocket } from '../hooks/useWebSocket';
import { ModelMetrics, OptimizationMetrics, FeatureImportance } from '../types/ai';

interface AIPerformanceProps {
  modelId: string;
  refreshInterval?: number;
}

export const AIPerformance: React.FC<AIPerformanceProps> = ({
  modelId,
  refreshInterval = 5000,
}) => {
  const [activeTab, setActiveTab] = useState(0);
  const [modelMetrics, setModelMetrics] = useState<ModelMetrics | null>(null);
  const [optimizationMetrics, setOptimizationMetrics] = useState<OptimizationMetrics | null>(null);
  const [featureImportance, setFeatureImportance] = useState<FeatureImportance[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  const wsEndpoint = `ws://api/v1/ai/performance/${modelId}/stream`;
  const { lastMessage } = useWebSocket(wsEndpoint);

  useEffect(() => {
    if (lastMessage) {
      const data = JSON.parse(lastMessage.data);
      setModelMetrics(data.modelMetrics);
      setOptimizationMetrics(data.optimizationMetrics);
      setFeatureImportance(data.featureImportance);
      setIsLoading(false);
    }
  }, [lastMessage]);

  const renderModelMetrics = () => (
    <Card>
      <CardContent>
        <Typography variant="h6">Model Performance</Typography>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={modelMetrics?.accuracyHistory || []}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="timestamp" />
            <YAxis />
            <RechartsTooltip />
            <Legend />
            <Line
              type="monotone"
              dataKey="accuracy"
              stroke="#8884d8"
              name="Accuracy"
            />
            <Line
              type="monotone"
              dataKey="confidence"
              stroke="#82ca9d"
              name="Confidence"
            />
          </LineChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );

  const renderOptimizationMetrics = () => (
    <Card>
      <CardContent>
        <Typography variant="h6">Optimization Impact</Typography>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={optimizationMetrics?.impactHistory || []}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="timestamp" />
            <YAxis />
            <RechartsTooltip />
            <Legend />
            <Line
              type="monotone"
              dataKey="viralCoefficient"
              stroke="#8884d8"
              name="Viral Coefficient"
            />
            <Line
              type="monotone"
              dataKey="engagement"
              stroke="#82ca9d"
              name="Engagement"
            />
          </LineChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );

  const renderFeatureImportance = () => (
    <Card>
      <CardContent>
        <Typography variant="h6">Feature Importance</Typography>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={featureImportance}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="feature" />
            <YAxis />
            <RechartsTooltip />
            <Legend />
            <Bar dataKey="importance" fill="#8884d8" name="Importance" />
          </BarChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );

  const renderModelHealth = () => (
    <Grid container spacing={2}>
      <Grid item xs={12} md={6}>
        <Card>
          <CardContent>
            <Typography variant="h6">Model Health</Typography>
            <Box display="flex" justifyContent="space-between" mt={2}>
              <Tooltip title="Model Drift Status">
                <Box textAlign="center">
                  <CircularProgress
                    variant="determinate"
                    value={modelMetrics?.driftScore || 0}
                    color={modelMetrics?.driftScore < 30 ? 'success' : 'warning'}
                  />
                  <Typography>Drift Score</Typography>
                </Box>
              </Tooltip>
              <Tooltip title="Training Status">
                <Box textAlign="center">
                  <CircularProgress
                    variant="determinate"
                    value={modelMetrics?.trainingProgress || 0}
                    color="primary"
                  />
                  <Typography>Training Progress</Typography>
                </Box>
              </Tooltip>
            </Box>
          </CardContent>
        </Card>
      </Grid>
      <Grid item xs={12} md={6}>
        <Card>
          <CardContent>
            <Typography variant="h6">Optimization Status</Typography>
            <Box display="flex" justifyContent="space-between" mt={2}>
              <Typography>
                Latest Accuracy: {modelMetrics?.currentAccuracy.toFixed(2)}%
              </Typography>
              <Typography>
                Optimization Impact: {optimizationMetrics?.currentImpact.toFixed(2)}x
              </Typography>
            </Box>
          </CardContent>
        </Card>
      </Grid>
    </Grid>
  );

  if (isLoading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight={400}>
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Box>
      <Paper sx={{ mb: 2 }}>
        <Tabs
          value={activeTab}
          onChange={(_, newValue) => setActiveTab(newValue)}
          indicatorColor="primary"
          textColor="primary"
          variant="fullWidth"
        >
          <Tab label="Model Metrics" />
          <Tab label="Optimization" />
          <Tab label="Features" />
          <Tab label="Health" />
        </Tabs>
      </Paper>

      <Box mt={2}>
        {activeTab === 0 && renderModelMetrics()}
        {activeTab === 1 && renderOptimizationMetrics()}
        {activeTab === 2 && renderFeatureImportance()}
        {activeTab === 3 && renderModelHealth()}
      </Box>
    </Box>
  );
};

