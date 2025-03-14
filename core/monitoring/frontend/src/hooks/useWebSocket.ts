import { useEffect, useRef, useState, useCallback } from 'react';

// Types for WebSocket configuration and state
interface WebSocketConfig {
    url: string;
    protocols?: string | string[];
    reconnectInterval?: number;
    maxReconnectAttempts?: number;
    heartbeatInterval?: number;
    compressionEnabled?: boolean;
    messageQueueSize?: number;
}

interface WebSocketMessage {
    id: string;
    type: string;
    payload: any;
    timestamp: number;
    priority: number;
}

interface WebSocketState {
    isConnected: boolean;
    lastError: Error | null;
    reconnectAttempts: number;
    latency: number;
    messageCount: number;
}

// Custom hook for WebSocket management
export const useWebSocket = (config: WebSocketConfig) => {
    const ws = useRef<WebSocket | null>(null);
    const messageQueue = useRef<Map<string, WebSocketMessage>>(new Map());
    const subscriptions = useRef<Map<string, Set<(data: any) => void>>>(new Map());
    const connectionPool = useRef<WebSocket[]>([]);
    
    const [state, setState] = useState<WebSocketState>({
        isConnected: false,
        lastError: null,
        reconnectAttempts: 0,
        latency: 0,
        messageCount: 0
    });

    // Message handling with deduplication and prioritization
    const handleMessage = useCallback((message: WebSocketMessage) => {
        // Deduplication check
        if (messageQueue.current.has(message.id)) {
            return;
        }

        // Add to queue with priority
        messageQueue.current.set(message.id, message);

        // Process queue based on priority
        const processQueue = () => {
            const sortedMessages = Array.from(messageQueue.current.values())
                .sort((a, b) => b.priority - a.priority);

            for (const msg of sortedMessages) {
                // Notify subscribers
                const handlers = subscriptions.current.get(msg.type) || new Set();
                handlers.forEach(handler => handler(msg.payload));

                // Remove processed message
                messageQueue.current.delete(msg.id);
            }
        };

        // Batch process messages
        requestAnimationFrame(processQueue);
    }, []);

    // Connection management with automatic reconnection
    const connect = useCallback(() => {
        try {
            const socket = new WebSocket(config.url, config.protocols);

            // Connection monitoring
            socket.onopen = () => {
                setState(prev => ({ ...prev, isConnected: true, reconnectAttempts: 0 }));
                startHeartbeat(socket);
            };

            socket.onclose = () => {
                setState(prev => ({ 
                    ...prev, 
                    isConnected: false,
                    reconnectAttempts: prev.reconnectAttempts + 1
                }));
                handleReconnection();
            };

            socket.onerror = (error) => {
                setState(prev => ({ ...prev, lastError: error as Error }));
                handleError(error);
            };

            socket.onmessage = (event) => {
                const startTime = performance.now();
                
                // Handle compression if enabled
                const data = config.compressionEnabled 
                    ? decompressMessage(event.data)
                    : JSON.parse(event.data);

                handleMessage(data);
                
                // Track latency
                const latency = performance.now() - startTime;
                setState(prev => ({ 
                    ...prev, 
                    latency: (prev.latency + latency) / 2,
                    messageCount: prev.messageCount + 1
                }));
            };

            ws.current = socket;
            connectionPool.current.push(socket);

        } catch (error) {
            handleError(error as Error);
        }
    }, [config.url, config.protocols, handleMessage]);

    // Heartbeat mechanism
    const startHeartbeat = (socket: WebSocket) => {
        const interval = setInterval(() => {
            if (socket.readyState === WebSocket.OPEN) {
                socket.send(JSON.stringify({ type: 'heartbeat' }));
            }
        }, config.heartbeatInterval || 30000);

        return () => clearInterval(interval);
    };

    // Reconnection logic
    const handleReconnection = useCallback(() => {
        if (state.reconnectAttempts < (config.maxReconnectAttempts || 5)) {
            setTimeout(() => {
                connect();
            }, (config.reconnectInterval || 1000) * Math.pow(2, state.reconnectAttempts));
        }
    }, [state.reconnectAttempts, config.maxReconnectAttempts, config.reconnectInterval, connect]);

    // Error handling
    const handleError = (error: Error) => {
        console.error('WebSocket error:', error);
        // Implement custom error reporting here
    };

    // Message compression
    const decompressMessage = (data: any) => {
        // Implement message decompression logic here
        return JSON.parse(data);
    };

    // Subscription management
    const subscribe = useCallback((type: string, handler: (data: any) => void) => {
        const handlers = subscriptions.current.get(type) || new Set();
        handlers.add(handler);
        subscriptions.current.set(type, handlers);

        return () => {
            const handlers = subscriptions.current.get(type);
            if (handlers) {
                handlers.delete(handler);
                if (handlers.size === 0) {
                    subscriptions.current.delete(type);
                }
            }
        };
    }, []);

    // Send message with retry logic
    const sendMessage = useCallback((message: Omit<WebSocketMessage, 'id' | 'timestamp'>) => {
        const send = (retries = 0) => {
            if (!ws.current || ws.current.readyState !== WebSocket.OPEN) {
                if (retries < 3) {
                    setTimeout(() => send(retries + 1), 1000);
                }
                return;
            }

            const fullMessage = {
                ...message,
                id: Math.random().toString(36).substr(2, 9),
                timestamp: Date.now()
            };

            try {
                const data = config.compressionEnabled 
                    ? compressMessage(fullMessage)
                    : JSON.stringify(fullMessage);
                ws.current.send(data);
            } catch (error) {
                handleError(error as Error);
                if (retries < 3) {
                    setTimeout(() => send(retries + 1), 1000);
                }
            }
        };

        send();
    }, [config.compressionEnabled]);

    // Message compression
    const compressMessage = (message: any) => {
        // Implement message compression logic here
        return JSON.stringify(message);
    };

    // Initialize connection
    useEffect(() => {
        connect();
        return () => {
            // Cleanup connections
            connectionPool.current.forEach(socket => socket.close());
            connectionPool.current = [];
            ws.current?.close();
        };
    }, [connect]);

    return {
        state,
        sendMessage,
        subscribe,
        messageQueue: messageQueue.current,
        latency: state.latency,
        connectionPool: connectionPool.current,
    };
};

