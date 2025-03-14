import React, { useEffect, useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  Grid,
  Typography,
  LinearProgress,
  Alert,
  IconButton,
  Chip,
} from '@mui/material';
import {
  Timeline,
  Assessment,
  Warning,
  CheckCircle,
  Error,
  Refresh,
} from '@mui/icons-material';
import { useTheme } from '@mui/material/styles';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';

// Types for system health data
interface SystemMetrics {
  cpu: number;
  memory: number;
  disk: number;
  network: number;
}

interface ComponentStatus {
  name: string;
  status: 'healthy' | 'warning' | 'error';
  responseTime: number;
  errorRate: number;
}

interface AlertNotification {
  id: string;
  type: 'error' | 'warning' | 'info';
  message: string;
  timestamp: string;
}

interface SystemHealthProps {
  websocketUrl: string;
  onAlertAcknowledge: (alertId: string) => void;
  onRefreshMetrics: () => void;
}

const SystemHealth: React.FC<SystemHealthProps> = ({
  websocketUrl,
  onAlertAcknowledge,
  onRefreshMetrics,
}) => {
  const theme = useTheme();
  const [metrics, setMetrics] = useState<SystemMetrics>({
    cpu: 0,
    memory: 0,
    disk: 0,
    network: 0,
  });
  const [components, setComponents] = useState<ComponentStatus[]>([]);
  const [alerts, setAlerts] = useState<AlertNotification[]>([]);
  const [socket, setSocket] = useState<WebSocket | null>(null);

  // Initialize WebSocket connection
  useEffect(() => {
    const ws = new WebSocket(websocketUrl);
    
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      switch (data.type) {
        case 'metrics':
          setMetrics(data.metrics);
          break;
        case 'components':
          setComponents(data.components);
          break;
        case 'alert':
          setAlerts(prev => [...prev, data.alert]);
          break;
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    setSocket(ws);

    return () => {
      ws.close();
    };
  }, [websocketUrl]);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy':
        return theme.palette.success.main;
      case 'warning':
        return theme.palette.warning.main;
      case 'error':
        return theme.palette.error.main;
      default:
        return theme.palette.grey[500];
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'healthy':
        return <CheckCircle color="success" />;
      case 'warning':
        return <Warning color="warning" />;
      case 'error':
        return <Error color="error" />;
      default:
        return null;
    }
  };

  return (
    <Box sx={{ p: 3 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 3 }}>
        <Typography variant="h4">System Health</Typography>
        <IconButton onClick={onRefreshMetrics}>
          <Refresh />
        </IconButton>
      </Box>

      {/* Resource Utilization */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Resource Utilization
              </Typography>
              <Box sx={{ my: 2 }}>
                <Typography variant="subtitle1">CPU Usage</Typography>
                <LinearProgress
                  variant="determinate"
                  value={metrics.cpu}
                  color={metrics.cpu > 80 ? 'error' : 'primary'}
                />
                <Typography variant="body2">{`${Math.round(metrics.cpu)}%`}</Typography>
              </Box>
              <Box sx={{ my: 2 }}>
                <Typography variant="subtitle1">Memory Usage</Typography>
                <LinearProgress
                  variant="determinate"
                  value={metrics.memory}
                  color={metrics.memory > 80 ? 'error' : 'primary'}
                />
                <Typography variant="body2">{`${Math.round(metrics.memory)}%`}</Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Component Status Grid */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Component Status
              </Typography>
              <Grid container spacing={2}>
                {components.map((component) => (
                  <Grid item xs={12} key={component.name}>
                    <Box
                      sx={{
                        display: 'flex',
                        alignItems: 'center',
                        p: 1,
                        border: 1,
                        borderColor: 'divider',
                        borderRadius: 1,
                      }}
                    >
                      {getStatusIcon(component.status)}
                      <Typography sx={{ ml: 1 }}>{component.name}</Typography>
                      <Chip
                        label={`${component.responseTime}ms`}
                        size="small"
                        sx={{ ml: 'auto' }}
                      />
                    </Box>
                  </Grid>
                ))}
              </Grid>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Alert Notifications */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Active Alerts
          </Typography>
          {alerts.map((alert) => (
            <Alert
              key={alert.id}
              severity={alert.type}
              onClose={() => onAlertAcknowledge(alert.id)}
              sx={{ mb: 1 }}
            >
              <Typography variant="body2">
                {alert.message}
                <Typography
                  component="span"
                  variant="caption"
                  sx={{ ml: 2, color: 'text.secondary' }}
                >
                  {new Date(alert.timestamp).toLocaleString()}
                </Typography>
              </Typography>
            </Alert>
          ))}
        </CardContent>
      </Card>

      {/* Performance Metrics */}
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Performance Trends
          </Typography>
          <Box sx={{ height: 300 }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={[metrics]}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="timestamp" />
                <YAxis />
                <Tooltip />
                <Line
                  type="monotone"
                  dataKey="cpu"
                  stroke={theme.palette.primary.main}
                  name="CPU"
                />
                <Line
                  type="monotone"
                  dataKey="memory"
                  stroke={theme.palette.secondary.main}
                  name="Memory"
                />
              </LineChart>
            </ResponsiveContainer>
          </Box>
        </CardContent>
      </Card>
    </Box>
  );
};

export default SystemHealth;

import React, { useState } from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  Grid,
  Chip,
  LinearProgress,
  List,
  ListItem,
  ListItemText,
  IconButton,
  Badge,
  Alert,

