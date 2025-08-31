import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  RefreshControl,
  Alert,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { useAuth } from '../contexts/AuthContext';
import { apiService } from '../services/api';

interface MarketData {
  symbol: string;
  name: string;
  value: number;
  change: number;
  change_pct: number;
}

interface TrendingStock {
  ticker: string;
  price: number;
  change: number;
  change_pct: number;
  volume: number;
}

const DashboardScreen: React.FC = () => {
  const { user, logout } = useAuth();
  const [marketData, setMarketData] = useState<MarketData[]>([]);
  const [trendingStocks, setTrendingStocks] = useState<TrendingStock[]>([]);
  const [refreshing, setRefreshing] = useState(false);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    loadDashboardData();
  }, []);

  const loadDashboardData = async () => {
    try {
      setIsLoading(true);
      const [marketOverview] = await Promise.all([
        apiService.getMarketOverview(),
      ]);
      
      setMarketData(marketOverview.indices || []);
      setTrendingStocks(marketOverview.trending_stocks || []);
    } catch (error: any) {
      Alert.alert('Error', 'Failed to load market data');
      console.error('Dashboard data error:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const onRefresh = async () => {
    setRefreshing(true);
    await loadDashboardData();
    setRefreshing(false);
  };

  const handleLogout = () => {
    Alert.alert(
      'Logout',
      'Are you sure you want to logout?',
      [
        { text: 'Cancel', style: 'cancel' },
        { text: 'Logout', onPress: logout },
      ]
    );
  };

  const renderMarketCard = (data: MarketData) => (
    <View key={data.symbol} style={styles.marketCard}>
      <Text style={styles.marketName}>{data.name}</Text>
      <Text style={styles.marketValue}>${data.value.toLocaleString()}</Text>
      <View style={[
        styles.changeContainer,
        { backgroundColor: data.change >= 0 ? '#d4edda' : '#f8d7da' }
      ]}>
        <Ionicons
          name={data.change >= 0 ? 'trending-up' : 'trending-down'}
          size={16}
          color={data.change >= 0 ? '#155724' : '#721c24'}
        />
        <Text style={[
          styles.changeText,
          { color: data.change >= 0 ? '#155724' : '#721c24' }
        ]}>
          {data.change >= 0 ? '+' : ''}{data.change.toFixed(2)} ({data.change_pct.toFixed(2)}%)
        </Text>
      </View>
    </View>
  );

  const renderTrendingStock = (stock: TrendingStock) => (
    <View key={stock.ticker} style={styles.trendingCard}>
      <View style={styles.stockHeader}>
        <Text style={styles.stockTicker}>{stock.ticker}</Text>
        <Text style={styles.stockPrice}>${stock.price.toFixed(2)}</Text>
      </View>
      <View style={[
        styles.changeContainer,
        { backgroundColor: stock.change >= 0 ? '#d4edda' : '#f8d7da' }
      ]}>
        <Ionicons
          name={stock.change >= 0 ? 'trending-up' : 'trending-down'}
          size={14}
          color={stock.change >= 0 ? '#155724' : '#721c24'}
        />
        <Text style={[
          styles.changeText,
          { color: stock.change >= 0 ? '#155724' : '#721c24' }
        ]}>
          {stock.change >= 0 ? '+' : ''}{stock.change.toFixed(2)} ({stock.change_pct.toFixed(2)}%)
        </Text>
      </View>
      <Text style={styles.stockVolume}>Vol: {stock.volume.toLocaleString()}</Text>
    </View>
  );

  const renderQuickAction = (icon: string, title: string, onPress: () => void) => (
    <TouchableOpacity key={title} style={styles.quickAction} onPress={onPress}>
      <View style={styles.quickActionIcon}>
        <Ionicons name={icon as any} size={24} color="#667eea" />
      </View>
      <Text style={styles.quickActionText}>{title}</Text>
    </TouchableOpacity>
  );

  return (
    <ScrollView
      style={styles.container}
      refreshControl={
        <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
      }
    >
      {/* Header */}
      <LinearGradient
        colors={['#667eea', '#764ba2']}
        style={styles.header}
      >
        <View style={styles.headerContent}>
          <View>
            <Text style={styles.welcomeText}>Welcome back,</Text>
            <Text style={styles.userName}>{user?.username}</Text>
          </View>
          <TouchableOpacity onPress={handleLogout} style={styles.logoutButton}>
            <Ionicons name="log-out-outline" size={24} color="white" />
          </TouchableOpacity>
        </View>
      </LinearGradient>

      {/* Quick Actions */}
      <View style={styles.quickActionsContainer}>
        <Text style={styles.sectionTitle}>Quick Actions</Text>
        <View style={styles.quickActionsGrid}>
          {renderQuickAction('search', 'Search Stocks', () => {})}
          {renderQuickAction('pie-chart', 'Portfolio', () => {})}
          {renderQuickAction('star', 'Watchlist', () => {})}
          {renderQuickAction('trending-up', 'Market', () => {})}
          {renderQuickAction('analytics', 'Technical', () => {})}
          {renderQuickAction('brain', 'ML Predictions', () => {})}
        </View>
      </View>

      {/* Market Overview */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Market Overview</Text>
        <ScrollView horizontal showsHorizontalScrollIndicator={false}>
          <View style={styles.marketCardsContainer}>
            {marketData.map(renderMarketCard)}
          </View>
        </ScrollView>
      </View>

      {/* Trending Stocks */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Trending Stocks</Text>
        <View style={styles.trendingContainer}>
          {trendingStocks.map(renderTrendingStock)}
        </View>
      </View>

      {/* Portfolio Summary */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Portfolio Summary</Text>
        <View style={styles.portfolioCard}>
          <Text style={styles.portfolioTitle}>Total Value</Text>
          <Text style={styles.portfolioValue}>$0.00</Text>
          <Text style={styles.portfolioChange}>+0.00 (0.00%)</Text>
          <TouchableOpacity style={styles.viewPortfolioButton}>
            <Text style={styles.viewPortfolioButtonText}>View Portfolio</Text>
          </TouchableOpacity>
        </View>
      </View>
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f8f9fa',
  },
  header: {
    paddingTop: 50,
    paddingBottom: 30,
    paddingHorizontal: 20,
  },
  headerContent: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  welcomeText: {
    color: 'rgba(255, 255, 255, 0.8)',
    fontSize: 16,
  },
  userName: {
    color: 'white',
    fontSize: 24,
    fontWeight: 'bold',
  },
  logoutButton: {
    padding: 8,
  },
  quickActionsContainer: {
    padding: 20,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 15,
  },
  quickActionsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
  },
  quickAction: {
    width: '30%',
    alignItems: 'center',
    backgroundColor: 'white',
    padding: 15,
    borderRadius: 12,
    marginBottom: 15,
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.1,
    shadowRadius: 3.84,
    elevation: 5,
  },
  quickActionIcon: {
    width: 50,
    height: 50,
    borderRadius: 25,
    backgroundColor: '#f8f9fa',
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 10,
  },
  quickActionText: {
    fontSize: 12,
    color: '#666',
    textAlign: 'center',
  },
  section: {
    paddingHorizontal: 20,
    marginBottom: 20,
  },
  marketCardsContainer: {
    flexDirection: 'row',
    paddingRight: 20,
  },
  marketCard: {
    backgroundColor: 'white',
    padding: 15,
    borderRadius: 12,
    marginRight: 15,
    minWidth: 120,
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.1,
    shadowRadius: 3.84,
    elevation: 5,
  },
  marketName: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 5,
  },
  marketValue: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#667eea',
    marginBottom: 8,
  },
  changeContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 6,
  },
  changeText: {
    fontSize: 12,
    fontWeight: '500',
    marginLeft: 4,
  },
  trendingContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
  },
  trendingCard: {
    backgroundColor: 'white',
    padding: 15,
    borderRadius: 12,
    width: '48%',
    marginBottom: 15,
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.1,
    shadowRadius: 3.84,
    elevation: 5,
  },
  stockHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  stockTicker: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#333',
  },
  stockPrice: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#667eea',
  },
  stockVolume: {
    fontSize: 12,
    color: '#666',
    marginTop: 8,
  },
  portfolioCard: {
    backgroundColor: 'white',
    padding: 20,
    borderRadius: 12,
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.1,
    shadowRadius: 3.84,
    elevation: 5,
  },
  portfolioTitle: {
    fontSize: 16,
    color: '#666',
    marginBottom: 5,
  },
  portfolioValue: {
    fontSize: 32,
    fontWeight: 'bold',
    color: '#667eea',
    marginBottom: 5,
  },
  portfolioChange: {
    fontSize: 14,
    color: '#28a745',
    marginBottom: 20,
  },
  viewPortfolioButton: {
    backgroundColor: '#667eea',
    paddingHorizontal: 20,
    paddingVertical: 10,
    borderRadius: 8,
  },
  viewPortfolioButtonText: {
    color: 'white',
    fontSize: 14,
    fontWeight: 'bold',
  },
});

export default DashboardScreen;
