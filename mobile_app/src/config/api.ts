// API Configuration for different environments
export const API_CONFIG = {
  // Development URLs
  development: {
    // Android emulator
    android: 'http://10.0.2.2:8000',
    // iOS simulator
    ios: 'http://localhost:8000',
    // Physical device (replace with your computer's IP)
    device: 'http://192.168.1.100:8000', // Change this to your actual IP
  },
  
  // Production URL (when you deploy your backend)
  production: 'https://your-backend-url.com',
  
  // Current environment
  current: 'development' as keyof typeof API_CONFIG,
};

// Get the appropriate API URL based on platform
export const getApiUrl = () => {
  const { current } = API_CONFIG;
  
  if (current === 'production') {
    return API_CONFIG.production;
  }
  
  // For development, you can manually set which URL to use
  // or detect the platform automatically
  return API_CONFIG.development.android; // Default to Android emulator
};

// Alternative: Auto-detect platform
export const getApiUrlAuto = () => {
  const { current } = API_CONFIG;
  
  if (current === 'production') {
    return API_CONFIG.production;
  }
  
  // This would need Platform detection in React Native
  // For now, return Android emulator URL
  return API_CONFIG.development.android;
};




