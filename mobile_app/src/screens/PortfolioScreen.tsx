import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  RefreshControl,
  Alert,
  Modal,
  TextInput,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { apiService } from '../services/api';

interface PortfolioItem {
  ticker: string;
  shares: number;
  avg_price: number;
  current_price: number;
  total_value: number;
  total_cost: number;
  gain_loss: number;
  gain_loss_pct: number;
  added_at: string;
}

interface PortfolioSummary {
  total_value: number;
  total_cost: number;
  total_gain_loss: number;
  total_gain_loss_pct: number;
  num_positions: number;
}

const PortfolioScreen: React.FC = () => {
  const [portfolio, setPortfolio] = useState<PortfolioItem[]>([]);
  const [summary, setSummary] = useState<PortfolioSummary | null>(null);
  const [refreshing, setRefreshing] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [showAddModal, setShowAddModal] = useState(false);
  const [newTicker, setNewTicker] = useState('');
  const [newShares, setNewShares] = useState('');
  const [newPrice, setNewPrice] = useState('');

  useEffect(() => {
    loadPortfolio();
  }, []);

  const loadPortfolio = async () => {
    try {
      setIsLoading(true);
      const portfolioData = await apiService.getPortfolio();
      setPortfolio(portfolioData.portfolio || []);
      setSummary(portfolioData.summary || null);
    } catch (error: any) {
      Alert.alert('Error', 'Failed to load portfolio');
      console.error('Portfolio error:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const onRefresh = async () => {
    setRefreshing(true);
    await loadPortfolio();
    setRefreshing(false);
  };

  const handleAddToPortfolio = async () => {
    if (!newTicker.trim() || !newShares.trim() || !newPrice.trim()) {
      Alert.alert('Error', 'Please fill in all fields');
      return;
    }

    const shares = parseFloat(newShares);
    const price = parseFloat(newPrice);

    if (isNaN(shares) || isNaN(price) || shares <= 0 || price <= 0) {
      Alert.alert('Error', 'Please enter valid numbers');
      return;
    }

    try {
      await apiService.addToPortfolio(newTicker.trim().toUpperCase(), shares, price);
      Alert.alert('Success', 'Stock added to portfolio');
      setShowAddModal(false);
      setNewTicker('');
      setNewShares('');
      setNewPrice('');
      loadPortfolio();
    } catch (error: any) {
      Alert.alert('Error', error.message || 'Failed to add stock');
    }
  };

  const renderPortfolioItem = (item: PortfolioItem) => (
    <View key={item.ticker} style={styles.portfolioItem}>
      <View style={styles.itemHeader}>
        <Text style={styles.ticker}>{item.ticker}</Text>
        <Text style={styles.currentPrice}>${item.current_price.toFixed(2)}</Text>
      </View>
      
      <View style={styles.itemDetails}>
        <View style={styles.detailRow}>
          <Text style={styles.detailLabel}>Shares:</Text>
          <Text style={styles.detailValue}>{item.shares}</Text>
        </View>
        <View style={styles.detailRow}>
          <Text style={styles.detailLabel}>Avg Price:</Text>
          <Text style={styles.detailValue}>${item.avg_price.toFixed(2)}</Text>
        </View>
        <View style={styles.detailRow}>
          <Text style={styles.detailLabel}>Total Value:</Text>
          <Text style={styles.detailValue}>${item.total_value.toFixed(2)}</Text>
        </View>
        <View style={styles.detailRow}>
          <Text style={styles.detailLabel}>Total Cost:</Text>
          <Text style={styles.detailValue}>${item.total_cost.toFixed(2)}</Text>
        </View>
      </View>

      <View style={[
        styles.gainLossContainer,
        { backgroundColor: item.gain_loss >= 0 ? '#d4edda' : '#f8d7da' }
      ]}>
        <Ionicons
          name={item.gain_loss >= 0 ? 'trending-up' : 'trending-down'}
          size={16}
          color={item.gain_loss >= 0 ? '#155724' : '#721c24'}
        />
        <Text style={[
          styles.gainLossText,
          { color: item.gain_loss >= 0 ? '#155724' : '#721c24' }
        ]}>
          {item.gain_loss >= 0 ? '+' : ''}${item.gain_loss.toFixed(2)} ({item.gain_loss_pct.toFixed(2)}%)
        </Text>
      </View>
    </View>
  );

  const renderSummaryCard = () => {
    if (!summary) return null;

    return (
      <View style={styles.summaryCard}>
        <Text style={styles.summaryTitle}>Portfolio Summary</Text>
        <View style={styles.summaryRow}>
          <Text style={styles.summaryLabel}>Total Value:</Text>
          <Text style={styles.summaryValue}>${summary.total_value.toFixed(2)}</Text>
        </View>
        <View style={styles.summaryRow}>
          <Text style={styles.summaryLabel}>Total Cost:</Text>
          <Text style={styles.summaryValue}>${summary.total_cost.toFixed(2)}</Text>
        </View>
        <View style={styles.summaryRow}>
          <Text style={styles.summaryLabel}>Total Gain/Loss:</Text>
          <Text style={[
            styles.summaryValue,
            { color: summary.total_gain_loss >= 0 ? '#28a745' : '#dc3545' }
          ]}>
            {summary.total_gain_loss >= 0 ? '+' : ''}${summary.total_gain_loss.toFixed(2)}
          </Text>
        </View>
        <View style={styles.summaryRow}>
          <Text style={styles.summaryLabel}>Gain/Loss %:</Text>
          <Text style={[
            styles.summaryValue,
            { color: summary.total_gain_loss >= 0 ? '#28a745' : '#dc3545' }
          ]}>
            {summary.total_gain_loss >= 0 ? '+' : ''}{summary.total_gain_loss_pct.toFixed(2)}%
          </Text>
        </View>
        <View style={styles.summaryRow}>
          <Text style={styles.summaryLabel}>Positions:</Text>
          <Text style={styles.summaryValue}>{summary.num_positions}</Text>
        </View>
      </View>
    );
  };

  return (
    <View style={styles.container}>
      <ScrollView
        style={styles.scrollView}
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
        }
      >
        {/* Header */}
        <View style={styles.header}>
          <Text style={styles.headerTitle}>My Portfolio</Text>
          <TouchableOpacity
            style={styles.addButton}
            onPress={() => setShowAddModal(true)}
          >
            <Ionicons name="add" size={24} color="white" />
          </TouchableOpacity>
        </View>

        {/* Summary Card */}
        {renderSummaryCard()}

        {/* Portfolio Items */}
        <View style={styles.portfolioContainer}>
          <Text style={styles.sectionTitle}>Holdings</Text>
          {portfolio.length === 0 ? (
            <View style={styles.emptyState}>
              <Ionicons name="pie-chart-outline" size={64} color="#ccc" />
              <Text style={styles.emptyStateText}>No stocks in portfolio</Text>
              <Text style={styles.emptyStateSubtext}>
                Add your first stock to start tracking your investments
              </Text>
            </View>
          ) : (
            portfolio.map(renderPortfolioItem)
          )}
        </View>
      </ScrollView>

      {/* Add Stock Modal */}
      <Modal
        visible={showAddModal}
        animationType="slide"
        transparent={true}
        onRequestClose={() => setShowAddModal(false)}
      >
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <Text style={styles.modalTitle}>Add Stock to Portfolio</Text>
            
            <View style={styles.inputContainer}>
              <Text style={styles.inputLabel}>Ticker Symbol</Text>
              <TextInput
                style={styles.input}
                value={newTicker}
                onChangeText={setNewTicker}
                placeholder="e.g., AAPL"
                autoCapitalize="characters"
                autoCorrect={false}
              />
            </View>

            <View style={styles.inputContainer}>
              <Text style={styles.inputLabel}>Number of Shares</Text>
              <TextInput
                style={styles.input}
                value={newShares}
                onChangeText={setNewShares}
                placeholder="e.g., 100"
                keyboardType="numeric"
              />
            </View>

            <View style={styles.inputContainer}>
              <Text style={styles.inputLabel}>Average Price per Share</Text>
              <TextInput
                style={styles.input}
                value={newPrice}
                onChangeText={setNewPrice}
                placeholder="e.g., 150.50"
                keyboardType="numeric"
              />
            </View>

            <View style={styles.modalButtons}>
              <TouchableOpacity
                style={styles.cancelButton}
                onPress={() => setShowAddModal(false)}
              >
                <Text style={styles.cancelButtonText}>Cancel</Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={styles.addButton}
                onPress={handleAddToPortfolio}
              >
                <Text style={styles.addButtonText}>Add Stock</Text>
              </TouchableOpacity>
            </View>
          </View>
        </View>
      </Modal>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f8f9fa',
  },
  scrollView: {
    flex: 1,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 20,
    paddingTop: 60,
    backgroundColor: '#667eea',
  },
  headerTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: 'white',
  },
  addButton: {
    backgroundColor: 'rgba(255, 255, 255, 0.2)',
    padding: 10,
    borderRadius: 8,
  },
  summaryCard: {
    backgroundColor: 'white',
    margin: 20,
    padding: 20,
    borderRadius: 12,
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.1,
    shadowRadius: 3.84,
    elevation: 5,
  },
  summaryTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 15,
    textAlign: 'center',
  },
  summaryRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 8,
    borderBottomWidth: 1,
    borderBottomColor: '#f0f0f0',
  },
  summaryLabel: {
    fontSize: 14,
    color: '#666',
  },
  summaryValue: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#333',
  },
  portfolioContainer: {
    paddingHorizontal: 20,
    marginBottom: 20,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 15,
  },
  portfolioItem: {
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
  itemHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 15,
  },
  ticker: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#333',
  },
  currentPrice: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#667eea',
  },
  itemDetails: {
    marginBottom: 15,
  },
  detailRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 4,
  },
  detailLabel: {
    fontSize: 14,
    color: '#666',
  },
  detailValue: {
    fontSize: 14,
    fontWeight: '500',
    color: '#333',
  },
  gainLossContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 8,
  },
  gainLossText: {
    fontSize: 14,
    fontWeight: 'bold',
    marginLeft: 8,
  },
  emptyState: {
    alignItems: 'center',
    paddingVertical: 60,
  },
  emptyStateText: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#666',
    marginTop: 20,
  },
  emptyStateSubtext: {
    fontSize: 14,
    color: '#999',
    textAlign: 'center',
    marginTop: 10,
    paddingHorizontal: 40,
  },
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  modalContent: {
    backgroundColor: 'white',
    borderRadius: 12,
    padding: 20,
    width: '90%',
    maxWidth: 400,
  },
  modalTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 20,
    textAlign: 'center',
  },
  inputContainer: {
    marginBottom: 15,
  },
  inputLabel: {
    fontSize: 14,
    fontWeight: '500',
    color: '#333',
    marginBottom: 5,
  },
  input: {
    borderWidth: 1,
    borderColor: '#ddd',
    borderRadius: 8,
    paddingHorizontal: 12,
    paddingVertical: 10,
    fontSize: 16,
  },
  modalButtons: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: 20,
  },
  cancelButton: {
    flex: 1,
    backgroundColor: '#6c757d',
    paddingVertical: 12,
    borderRadius: 8,
    marginRight: 10,
    alignItems: 'center',
  },
  cancelButtonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: 'bold',
  },
  addButtonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: 'bold',
  },
});

export default PortfolioScreen;
