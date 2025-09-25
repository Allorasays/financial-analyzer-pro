import React, { createContext, useContext, useEffect, useState, ReactNode } from 'react';
import { useAuth } from './AuthContext';
import { API_CONFIG } from '../config/api';

interface WebSocketContextType {
  isConnected: boolean;
  lastMessage: any;
  sendMessage: (message: any) => void;
  subscribeToSymbols: (symbols: string[]) => void;
}

const WebSocketContext = createContext<WebSocketContextType | undefined>(undefined);

export const useWebSocket = () => {
  const context = useContext(WebSocketContext);
  if (context === undefined) {
    throw new Error('useWebSocket must be used within a WebSocketProvider');
  }
  return context;
};

interface WebSocketProviderProps {
  children: ReactNode;
}

export const WebSocketProvider: React.FC<WebSocketProviderProps> = ({ children }) => {
  const { user } = useAuth();
  const [ws, setWs] = useState<WebSocket | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState<any>(null);
  const [reconnectAttempts, setReconnectAttempts] = useState(0);
  const maxReconnectAttempts = 5;

  useEffect(() => {
    if (user) {
      connect();
    } else {
      disconnect();
    }

    return () => {
      disconnect();
    };
  }, [user]);

  const connect = () => {
    if (ws?.readyState === WebSocket.OPEN) {
      return;
    }

    try {
      const wsUrl = API_CONFIG.WS_URL.replace('http', 'ws');
      const newWs = new WebSocket(`${wsUrl}/ws/${user?.id}`);

      newWs.onopen = () => {
        console.log('WebSocket connected');
        setIsConnected(true);
        setReconnectAttempts(0);
      };

      newWs.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          setLastMessage(data);
          
          // Handle different message types
          switch (data.type) {
            case 'market_data':
              console.log('Market data update:', data.data);
              break;
            case 'portfolio_update':
              console.log('Portfolio update:', data.message);
              break;
            case 'price_alert':
              console.log('Price alert:', data.data);
              break;
            case 'symbol_update':
              console.log('Symbol update:', data.data);
              break;
            case 'pong':
              console.log('Pong received');
              break;
            default:
              console.log('Unknown message type:', data.type);
          }
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };

      newWs.onclose = () => {
        console.log('WebSocket disconnected');
        setIsConnected(false);
        
        // Attempt to reconnect
        if (reconnectAttempts < maxReconnectAttempts) {
          setTimeout(() => {
            setReconnectAttempts(prev => prev + 1);
            connect();
          }, 5000 * (reconnectAttempts + 1)); // Exponential backoff
        }
      };

      newWs.onerror = (error) => {
        console.error('WebSocket error:', error);
        setIsConnected(false);
      };

      setWs(newWs);
    } catch (error) {
      console.error('Failed to create WebSocket connection:', error);
    }
  };

  const disconnect = () => {
    if (ws) {
      ws.close();
      setWs(null);
      setIsConnected(false);
    }
  };

  const sendMessage = (message: any) => {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(message));
    } else {
      console.warn('WebSocket is not connected');
    }
  };

  const subscribeToSymbols = (symbols: string[]) => {
    sendMessage({
      type: 'subscribe',
      symbols: symbols
    });
  };

  // Send ping every 30 seconds to keep connection alive
  useEffect(() => {
    if (isConnected) {
      const interval = setInterval(() => {
        sendMessage({ type: 'ping' });
      }, 30000);

      return () => clearInterval(interval);
    }
  }, [isConnected]);

  const value = {
    isConnected,
    lastMessage,
    sendMessage,
    subscribeToSymbols
  };

  return (
    <WebSocketContext.Provider value={value}>
      {children}
    </WebSocketContext.Provider>
  );
};

