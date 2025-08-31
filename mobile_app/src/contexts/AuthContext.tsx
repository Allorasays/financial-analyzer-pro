import React, { createContext, useContext, useState, useEffect } from 'react';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { api } from '../services/api';

interface User {
  username: string;
  email: string;
}

interface AuthContextType {
  user: User | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  login: (username: string, password: string) => Promise<void>;
  register: (username: string, email: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
  updateUser: (userData: User) => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    checkAuthStatus();
  }, []);

  const checkAuthStatus = async () => {
    try {
      const token = await AsyncStorage.getItem('authToken');
      const userData = await AsyncStorage.getItem('userData');
      
      if (token && userData) {
        api.defaults.headers.common['Authorization'] = `Bearer ${token}`;
        setUser(JSON.parse(userData));
        setIsAuthenticated(true);
      }
    } catch (error) {
      console.error('Error checking auth status:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const login = async (username: string, password: string) => {
    try {
      setIsLoading(true);
      const response = await api.post('/api/auth/login', {
        username,
        password,
      });

      const { access_token, username: userUsername } = response.data;
      
      // Store token and user data
      await AsyncStorage.setItem('authToken', access_token);
      await AsyncStorage.setItem('userData', JSON.stringify({ username: userUsername }));
      
      // Set API header
      api.defaults.headers.common['Authorization'] = `Bearer ${access_token}`;
      
      setUser({ username: userUsername, email: '' });
      setIsAuthenticated(true);
    } catch (error: any) {
      throw new Error(error.response?.data?.detail || 'Login failed');
    } finally {
      setIsLoading(false);
    }
  };

  const register = async (username: string, email: string, password: string) => {
    try {
      setIsLoading(true);
      const response = await api.post('/api/auth/register', {
        username,
        email,
        password,
      });

      const { access_token } = response.data;
      
      // Store token and user data
      await AsyncStorage.setItem('authToken', access_token);
      await AsyncStorage.setItem('userData', JSON.stringify({ username, email }));
      
      // Set API header
      api.defaults.headers.common['Authorization'] = `Bearer ${access_token}`;
      
      setUser({ username, email });
      setIsAuthenticated(true);
    } catch (error: any) {
      throw new Error(error.response?.data?.detail || 'Registration failed');
    } finally {
      setIsLoading(false);
    }
  };

  const logout = async () => {
    try {
      await AsyncStorage.removeItem('authToken');
      await AsyncStorage.removeItem('userData');
      
      delete api.defaults.headers.common['Authorization'];
      
      setUser(null);
      setIsAuthenticated(false);
    } catch (error) {
      console.error('Error during logout:', error);
    }
  };

  const updateUser = (userData: User) => {
    setUser(userData);
    AsyncStorage.setItem('userData', JSON.stringify(userData));
  };

  const value: AuthContextType = {
    user,
    isAuthenticated,
    isLoading,
    login,
    register,
    logout,
    updateUser,
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
};
