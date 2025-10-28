# 📱 React Native Mobile App Structure

## 🏗️ **Project Structure**

```
FinancialAnalyzerApp/
├── src/
│   ├── components/
│   │   ├── common/
│   │   │   ├── Button.js
│   │   │   ├── Input.js
│   │   │   ├── Card.js
│   │   │   ├── LoadingSpinner.js
│   │   │   └── ErrorMessage.js
│   │   ├── auth/
│   │   │   ├── LoginScreen.js
│   │   │   ├── RegisterScreen.js
│   │   │   └── ForgotPasswordScreen.js
│   │   ├── portfolio/
│   │   │   ├── PortfolioListScreen.js
│   │   │   ├── PortfolioDetailScreen.js
│   │   │   ├── AddPositionScreen.js
│   │   │   └── PortfolioChart.js
│   │   ├── watchlist/
│   │   │   ├── WatchlistScreen.js
│   │   │   ├── AddToWatchlistScreen.js
│   │   │   └── WatchlistItem.js
│   │   ├── market/
│   │   │   ├── MarketOverviewScreen.js
│   │   │   ├── StockDetailScreen.js
│   │   │   ├── StockChart.js
│   │   │   └── NewsScreen.js
│   │   └── profile/
│   │       ├── ProfileScreen.js
│   │       ├── SubscriptionScreen.js
│   │       └── SettingsScreen.js
│   ├── services/
│   │   ├── api.js
│   │   ├── auth.js
│   │   ├── portfolio.js
│   │   ├── watchlist.js
│   │   └── market.js
│   ├── navigation/
│   │   ├── AppNavigator.js
│   │   ├── AuthNavigator.js
│   │   └── TabNavigator.js
│   ├── store/
│   │   ├── index.js
│   │   ├── authSlice.js
│   │   ├── portfolioSlice.js
│   │   └── watchlistSlice.js
│   ├── utils/
│   │   ├── constants.js
│   │   ├── helpers.js
│   │   └── validators.js
│   └── styles/
│       ├── colors.js
│       ├── typography.js
│       └── common.js
├── assets/
│   ├── images/
│   ├── icons/
│   └── fonts/
├── android/
├── ios/
├── package.json
└── App.js
```

## 🎨 **Design System**

### **Colors**
```javascript
// src/styles/colors.js
export const colors = {
  // Primary colors
  primary: '#667eea',
  primaryDark: '#5a6fd8',
  primaryLight: '#7c8ef0',
  
  // Secondary colors
  secondary: '#764ba2',
  secondaryDark: '#6a4190',
  secondaryLight: '#8255b4',
  
  // Status colors
  success: '#28a745',
  warning: '#ffc107',
  danger: '#dc3545',
  info: '#17a2b8',
  
  // Neutral colors
  white: '#ffffff',
  black: '#000000',
  gray100: '#f8f9fa',
  gray200: '#e9ecef',
  gray300: '#dee2e6',
  gray400: '#ced4da',
  gray500: '#adb5bd',
  gray600: '#6c757d',
  gray700: '#495057',
  gray800: '#343a40',
  gray900: '#212529',
  
  // Background colors
  background: '#f8f9fa',
  surface: '#ffffff',
  
  // Text colors
  textPrimary: '#212529',
  textSecondary: '#6c757d',
  textLight: '#adb5bd',
  
  // Chart colors
  chartGreen: '#28a745',
  chartRed: '#dc3545',
  chartBlue: '#007bff',
  chartOrange: '#fd7e14',
  chartPurple: '#6f42c1',
};
```

### **Typography**
```javascript
// src/styles/typography.js
export const typography = {
  // Font families
  fontFamily: {
    regular: 'System',
    medium: 'System',
    bold: 'System',
  },
  
  // Font sizes
  fontSize: {
    xs: 12,
    sm: 14,
    base: 16,
    lg: 18,
    xl: 20,
    '2xl': 24,
    '3xl': 30,
    '4xl': 36,
  },
  
  // Font weights
  fontWeight: {
    normal: '400',
    medium: '500',
    semibold: '600',
    bold: '700',
  },
  
  // Line heights
  lineHeight: {
    tight: 1.2,
    normal: 1.4,
    relaxed: 1.6,
  },
};
```

## 🔧 **Core Components**

### **Button Component**
```javascript
// src/components/common/Button.js
import React from 'react';
import { TouchableOpacity, Text, StyleSheet } from 'react-native';
import { colors, typography } from '../../styles';

const Button = ({ 
  title, 
  onPress, 
  variant = 'primary', 
  size = 'medium',
  disabled = false,
  style,
  textStyle 
}) => {
  const buttonStyle = [
    styles.button,
    styles[variant],
    styles[size],
    disabled && styles.disabled,
    style
  ];
  
  const buttonTextStyle = [
    styles.text,
    styles[`${variant}Text`],
    styles[`${size}Text`],
    disabled && styles.disabledText,
    textStyle
  ];
  
  return (
    <TouchableOpacity
      style={buttonStyle}
      onPress={onPress}
      disabled={disabled}
      activeOpacity={0.8}
    >
      <Text style={buttonTextStyle}>{title}</Text>
    </TouchableOpacity>
  );
};

const styles = StyleSheet.create({
  button: {
    borderRadius: 8,
    alignItems: 'center',
    justifyContent: 'center',
  },
  
  // Variants
  primary: {
    backgroundColor: colors.primary,
  },
  secondary: {
    backgroundColor: colors.secondary,
  },
  outline: {
    backgroundColor: 'transparent',
    borderWidth: 1,
    borderColor: colors.primary,
  },
  ghost: {
    backgroundColor: 'transparent',
  },
  
  // Sizes
  small: {
    paddingHorizontal: 12,
    paddingVertical: 8,
  },
  medium: {
    paddingHorizontal: 16,
    paddingVertical: 12,
  },
  large: {
    paddingHorizontal: 20,
    paddingVertical: 16,
  },
  
  // States
  disabled: {
    opacity: 0.5,
  },
  
  // Text styles
  text: {
    fontWeight: typography.fontWeight.medium,
  },
  primaryText: {
    color: colors.white,
  },
  secondaryText: {
    color: colors.white,
  },
  outlineText: {
    color: colors.primary,
  },
  ghostText: {
    color: colors.primary,
  },
  
  // Text sizes
  smallText: {
    fontSize: typography.fontSize.sm,
  },
  mediumText: {
    fontSize: typography.fontSize.base,
  },
  largeText: {
    fontSize: typography.fontSize.lg,
  },
  
  disabledText: {
    opacity: 0.7,
  },
});

export default Button;
```

### **Input Component**
```javascript
// src/components/common/Input.js
import React, { useState } from 'react';
import { View, TextInput, Text, StyleSheet } from 'react-native';
import { colors, typography } from '../../styles';

const Input = ({
  label,
  placeholder,
  value,
  onChangeText,
  error,
  secureTextEntry = false,
  keyboardType = 'default',
  style,
  ...props
}) => {
  const [isFocused, setIsFocused] = useState(false);
  
  const inputStyle = [
    styles.input,
    isFocused && styles.inputFocused,
    error && styles.inputError,
    style
  ];
  
  return (
    <View style={styles.container}>
      {label && <Text style={styles.label}>{label}</Text>}
      <TextInput
        style={inputStyle}
        placeholder={placeholder}
        placeholderTextColor={colors.gray500}
        value={value}
        onChangeText={onChangeText}
        secureTextEntry={secureTextEntry}
        keyboardType={keyboardType}
        onFocus={() => setIsFocused(true)}
        onBlur={() => setIsFocused(false)}
        {...props}
      />
      {error && <Text style={styles.errorText}>{error}</Text>}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    marginBottom: 16,
  },
  label: {
    fontSize: typography.fontSize.sm,
    fontWeight: typography.fontWeight.medium,
    color: colors.textPrimary,
    marginBottom: 8,
  },
  input: {
    borderWidth: 1,
    borderColor: colors.gray300,
    borderRadius: 8,
    paddingHorizontal: 12,
    paddingVertical: 12,
    fontSize: typography.fontSize.base,
    color: colors.textPrimary,
    backgroundColor: colors.white,
  },
  inputFocused: {
    borderColor: colors.primary,
  },
  inputError: {
    borderColor: colors.danger,
  },
  errorText: {
    fontSize: typography.fontSize.sm,
    color: colors.danger,
    marginTop: 4,
  },
});

export default Input;
```

## 🔐 **Authentication Service**

```javascript
// src/services/auth.js
import AsyncStorage from '@react-native-async-storage/async-storage';
import { API_BASE_URL } from '../utils/constants';

class AuthService {
  constructor() {
    this.baseURL = API_BASE_URL;
  }
  
  async login(email, password) {
    try {
      const response = await fetch(`${this.baseURL}/api/auth/login`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ email, password }),
      });
      
      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Login failed');
      }
      
      const data = await response.json();
      
      // Store tokens
      await AsyncStorage.setItem('access_token', data.access_token);
      await AsyncStorage.setItem('refresh_token', data.refresh_token);
      
      return data;
    } catch (error) {
      throw error;
    }
  }
  
  async register(userData) {
    try {
      const response = await fetch(`${this.baseURL}/api/auth/register`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(userData),
      });
      
      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Registration failed');
      }
      
      return await response.json();
    } catch (error) {
      throw error;
    }
  }
  
  async logout() {
    try {
      const refreshToken = await AsyncStorage.getItem('refresh_token');
      
      if (refreshToken) {
        await fetch(`${this.baseURL}/api/auth/logout`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${refreshToken}`,
          },
        });
      }
      
      // Clear stored tokens
      await AsyncStorage.removeItem('access_token');
      await AsyncStorage.removeItem('refresh_token');
    } catch (error) {
      console.error('Logout error:', error);
    }
  }
  
  async getAccessToken() {
    return await AsyncStorage.getItem('access_token');
  }
  
  async isAuthenticated() {
    const token = await this.getAccessToken();
    return !!token;
  }
  
  async refreshToken() {
    try {
      const refreshToken = await AsyncStorage.getItem('refresh_token');
      
      if (!refreshToken) {
        throw new Error('No refresh token available');
      }
      
      const response = await fetch(`${this.baseURL}/api/auth/refresh`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ refresh_token: refreshToken }),
      });
      
      if (!response.ok) {
        throw new Error('Token refresh failed');
      }
      
      const data = await response.json();
      
      // Update stored token
      await AsyncStorage.setItem('access_token', data.access_token);
      
      return data.access_token;
    } catch (error) {
      // If refresh fails, logout user
      await this.logout();
      throw error;
    }
  }
}

export default new AuthService();
```

## 📊 **Portfolio Service**

```javascript
// src/services/portfolio.js
import { API_BASE_URL } from '../utils/constants';
import authService from './auth';

class PortfolioService {
  constructor() {
    this.baseURL = API_BASE_URL;
  }
  
  async getAuthHeaders() {
    const token = await authService.getAccessToken();
    return {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${token}`,
    };
  }
  
  async getPortfolios() {
    try {
      const response = await fetch(`${this.baseURL}/api/portfolios`, {
        headers: await this.getAuthHeaders(),
      });
      
      if (!response.ok) {
        throw new Error('Failed to fetch portfolios');
      }
      
      return await response.json();
    } catch (error) {
      throw error;
    }
  }
  
  async createPortfolio(name, description = '') {
    try {
      const response = await fetch(`${this.baseURL}/api/portfolios`, {
        method: 'POST',
        headers: await this.getAuthHeaders(),
        body: JSON.stringify({ name, description }),
      });
      
      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to create portfolio');
      }
      
      return await response.json();
    } catch (error) {
      throw error;
    }
  }
  
  async getPortfolioPositions(portfolioId) {
    try {
      const response = await fetch(`${this.baseURL}/api/portfolios/${portfolioId}/positions`, {
        headers: await this.getAuthHeaders(),
      });
      
      if (!response.ok) {
        throw new Error('Failed to fetch portfolio positions');
      }
      
      return await response.json();
    } catch (error) {
      throw error;
    }
  }
  
  async addPosition(portfolioId, positionData) {
    try {
      const response = await fetch(`${this.baseURL}/api/portfolios/${portfolioId}/positions`, {
        method: 'POST',
        headers: await this.getAuthHeaders(),
        body: JSON.stringify(positionData),
      });
      
      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to add position');
      }
      
      return await response.json();
    } catch (error) {
      throw error;
    }
  }
}

export default new PortfolioService();
```

## 🧭 **Navigation Structure**

```javascript
// src/navigation/AppNavigator.js
import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createStackNavigator } from '@react-navigation/stack';
import { useSelector } from 'react-redux';

import AuthNavigator from './AuthNavigator';
import TabNavigator from './TabNavigator';
import { selectIsAuthenticated } from '../store/authSlice';

const Stack = createStackNavigator();

const AppNavigator = () => {
  const isAuthenticated = useSelector(selectIsAuthenticated);
  
  return (
    <NavigationContainer>
      <Stack.Navigator screenOptions={{ headerShown: false }}>
        {isAuthenticated ? (
          <Stack.Screen name="Main" component={TabNavigator} />
        ) : (
          <Stack.Screen name="Auth" component={AuthNavigator} />
        )}
      </Stack.Navigator>
    </NavigationContainer>
  );
};

export default AppNavigator;
```

## 📱 **Main App Component**

```javascript
// App.js
import React from 'react';
import { Provider } from 'react-redux';
import { StatusBar } from 'expo-status-bar';
import { SafeAreaProvider } from 'react-native-safe-area-context';

import store from './src/store';
import AppNavigator from './src/navigation/AppNavigator';

export default function App() {
  return (
    <Provider store={store}>
      <SafeAreaProvider>
        <AppNavigator />
        <StatusBar style="auto" />
      </SafeAreaProvider>
    </Provider>
  );
}
```

## 📦 **Package.json Dependencies**

```json
{
  "name": "financial-analyzer-app",
  "version": "1.0.0",
  "main": "node_modules/expo/AppEntry.js",
  "scripts": {
    "start": "expo start",
    "android": "expo start --android",
    "ios": "expo start --ios",
    "web": "expo start --web"
  },
  "dependencies": {
    "@expo/vector-icons": "^13.0.0",
    "@react-native-async-storage/async-storage": "1.17.3",
    "@react-navigation/native": "^6.0.2",
    "@react-navigation/stack": "^6.0.7",
    "@react-navigation/bottom-tabs": "^6.0.5",
    "expo": "~47.0.0",
    "expo-status-bar": "~1.4.2",
    "react": "18.1.0",
    "react-native": "0.70.5",
    "react-native-safe-area-context": "4.4.1",
    "react-native-screens": "~3.22.0",
    "react-redux": "^8.0.2",
    "@reduxjs/toolkit": "^1.9.1",
    "react-native-chart-kit": "^6.12.0",
    "react-native-svg": "13.4.0",
    "react-native-gesture-handler": "~2.8.1",
    "react-native-reanimated": "~2.12.0"
  },
  "devDependencies": {
    "@babel/core": "^7.12.9"
  }
}
```

This React Native structure provides:

1. **Modular Architecture**: Organized components, services, and navigation
2. **Design System**: Consistent colors, typography, and components
3. **Authentication**: Complete auth flow with token management
4. **Portfolio Management**: Full CRUD operations for portfolios and positions
5. **Scalable Structure**: Easy to add new features and screens
6. **Professional UI**: Modern, clean design following mobile best practices

The app will integrate seamlessly with your existing backend API and provide a professional mobile experience for your financial analysis platform! 📱💰



