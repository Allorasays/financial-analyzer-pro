import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  TextInput,
  Alert,
  ActivityIndicator,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { apiService } from '../services/api';

interface MLPrediction {
  day: number;
  predicted_price: number;
  date: string;
}

interface MLData {
  ticker: string;
  current_price: number;
  predicted_price_1d: number;
  confidence_score: number;
  future_predictions: MLPrediction[];
  model_accuracy: number;
  timestamp: string;
}

const MLPredictionsScreen: React.FC = () => {
  const [ticker, setTicker] = useState('');
  const [mlData, setMlData] = useState<MLData | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [daysAhead, setDaysAhead] = useState('30');

  const getPredictions = async () => {
    if (!ticker.trim()) {
      Alert.alert('Error', 'Please enter a ticker symbol');
      return;
    }

    try {
      setIsLoading(true);
      const data = await apiService.getMLPredictions(ticker.trim().toUpperCase(), parseInt(daysAhead));
      setMlData(data);
    } catch (error: any) {
      Alert.alert('Error', error.message || 'Failed to get predictions');
    } finally {
      setIsLoading(false);
    }
  };

  const renderPredictionItem = (prediction: MLPrediction) => (
    <View key={prediction.day} style={styles.predictionItem}>
      <View style={styles.predictionHeader}>
        <Text style={styles.predictionDay}>Day {prediction.day}</Text>
        <Text style={styles.predictionDate}>{prediction.date}</Text>
      </View>
      <Text style={styles.predictedPrice}>
        ${prediction.predicted_price.toFixed(2)}
      </Text>
    </View>
  );

  const renderMLResults = () => {
    if (!mlData) return null;

    const priceChange = mlData.predicted_price_1d - mlData.current_price;
    const priceChangePct = (priceChange / mlData.current_price) * 100;

    return (
      <View style={styles.resultsContainer}>
        {/* Current vs Predicted */}
        <View style={styles.comparisonCard}>
          <Text style={styles.comparisonTitle}>Price Prediction (1 Day)</Text>
          <View style={styles.priceComparison}>
            <View style={styles.priceItem}>
              <Text style={styles.priceLabel}>Current Price</Text>
              <Text style={styles.currentPrice}>${mlData.current_price.toFixed(2)}</Text>
            </View>
            <View style={styles.priceItem}>
              <Text style={styles.priceLabel}>Predicted Price</Text>
              <Text style={styles.predictedPrice}>${mlData.predicted_price_1d.toFixed(2)}</Text>
            </View>
          </View>
          <View style={[
            styles.changeContainer,
            { backgroundColor: priceChange >= 0 ? '#d4edda' : '#f8d7da' }
          ]}>
            <Ionicons
              name={priceChange >= 0 ? 'trending-up' : 'trending-down'}
              size={16}
              color={priceChange >= 0 ? '#155724' : '#721c24'}
            />
            <Text style={[
              styles.changeText,
              { color: priceChange >= 0 ? '#155724' : '#721c24' }
            ]}>
              {priceChange >= 0 ? '+' : ''}${priceChange.toFixed(2)} ({priceChangePct.toFixed(2)}%)
            </Text>
          </View>
        </View>

        {/* Model Confidence */}
        <View style={styles.confidenceCard}>
          <Text style={styles.confidenceTitle}>Model Confidence</Text>
          <View style={styles.confidenceBar}>
            <View style={styles.confidenceFill}>
              <View 
                style={[
                  styles.confidenceProgress, 
                  { width: `${mlData.confidence_score * 100}%` }
                ]} 
              />
            </View>
            <Text style={styles.confidenceText}>
              {mlData.confidence_score.toFixed(3)}
            </Text>
          </View>
          <Text style={styles.accuracyText}>
            Model Accuracy: {mlData.model_accuracy}%
          </Text>
        </View>

        {/* Future Predictions */}
        <View style={styles.predictionsCard}>
          <Text style={styles.predictionsTitle}>Future Predictions</Text>
          <View style={styles.predictionsGrid}>
            {mlData.future_predictions.slice(0, 12).map(renderPredictionItem)}
          </View>
        </View>

        {/* Disclaimer */}
        <View style={styles.disclaimerCard}>
          <Ionicons name="warning" size={20} color="#856404" />
          <Text style={styles.disclaimerText}>
            These predictions are generated using machine learning algorithms and should not be considered as financial advice. 
            Always do your own research and consult with financial professionals before making investment decisions.
          </Text>
        </View>
      </View>
    );
  };

  return (
    <ScrollView style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <Text style={styles.headerTitle}>ML Price Predictions</Text>
        <Text style={styles.headerSubtitle}>
          Machine learning-powered stock price forecasting
        </Text>
      </View>

      {/* Input Section */}
      <View style={styles.inputSection}>
        <View style={styles.inputRow}>
          <View style={styles.inputContainer}>
            <Text style={styles.inputLabel}>Stock Ticker</Text>
            <TextInput
              style={styles.input}
              value={ticker}
              onChangeText={setTicker}
              placeholder="e.g., AAPL"
              autoCapitalize="characters"
              autoCorrect={false}
            />
          </View>
          <View style={styles.inputContainer}>
            <Text style={styles.inputLabel}>Days Ahead</Text>
            <TextInput
              style={styles.input}
              value={daysAhead}
              onChangeText={setDaysAhead}
              placeholder="30"
              keyboardType="numeric"
            />
          </View>
        </View>
        
        <TouchableOpacity
          style={[styles.predictButton, isLoading && styles.disabledButton]}
          onPress={getPredictions}
          disabled={isLoading}
        >
          {isLoading ? (
            <ActivityIndicator color="white" />
          ) : (
            <>
              <Ionicons name="brain" size={20} color="white" />
              <Text style={styles.predictButtonText}>Get Predictions</Text>
            </>
          )}
        </TouchableOpacity>
      </View>

      {/* Results */}
      {renderMLResults()}

      {/* Features Info */}
      <View style={styles.featuresCard}>
        <Text style={styles.featuresTitle}>How It Works</Text>
        <View style={styles.featureItem}>
          <Ionicons name="analytics" size={16} color="#667eea" />
          <Text style={styles.featureText}>
            Analyzes historical price data, volume, and technical indicators
          </Text>
        </View>
        <View style={styles.featureItem}>
          <Ionicons name="trending-up" size={16} color="#667eea" />
          <Text style={styles.featureText}>
            Uses Random Forest algorithm for price prediction
          </Text>
        </View>
        <View style={styles.featureItem}>
          <Ionicons name="shield-checkmark" size={16} color="#667eea" />
          <Text style={styles.featureText}>
            Provides confidence scores and accuracy metrics
          </Text>
        </View>
        <View style={styles.featureItem}>
          <Ionicons name="calendar" size={16} color="#667eea" />
          <Text style={styles.featureText}>
            Generates short-term and long-term price forecasts
          </Text>
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
    backgroundColor: '#667eea',
    padding: 20,
    paddingTop: 60,
  },
  headerTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: 'white',
    marginBottom: 5,
  },
  headerSubtitle: {
    fontSize: 14,
    color: 'rgba(255, 255, 255, 0.8)',
  },
  inputSection: {
    padding: 20,
    backgroundColor: 'white',
    margin: 20,
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
  inputRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 20,
  },
  inputContainer: {
    flex: 1,
    marginRight: 15,
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
  predictButton: {
    backgroundColor: '#667eea',
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 15,
    borderRadius: 8,
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.25,
    shadowRadius: 3.84,
    elevation: 5,
  },
  disabledButton: {
    backgroundColor: '#ccc',
  },
  predictButtonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: 'bold',
    marginLeft: 8,
  },
  resultsContainer: {
    paddingHorizontal: 20,
  },
  comparisonCard: {
    backgroundColor: 'white',
    padding: 20,
    borderRadius: 12,
    marginBottom: 20,
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.1,
    shadowRadius: 3.84,
    elevation: 5,
  },
  comparisonTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 15,
    textAlign: 'center',
  },
  priceComparison: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 15,
  },
  priceItem: {
    alignItems: 'center',
    flex: 1,
  },
  priceLabel: {
    fontSize: 14,
    color: '#666',
    marginBottom: 5,
  },
  currentPrice: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#667eea',
  },
  predictedPrice: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#28a745',
  },
  changeContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 8,
  },
  changeText: {
    fontSize: 16,
    fontWeight: 'bold',
    marginLeft: 8,
  },
  confidenceCard: {
    backgroundColor: 'white',
    padding: 20,
    borderRadius: 12,
    marginBottom: 20,
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.1,
    shadowRadius: 3.84,
    elevation: 5,
  },
  confidenceTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 15,
    textAlign: 'center',
  },
  confidenceBar: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 10,
  },
  confidenceFill: {
    flex: 1,
    height: 20,
    backgroundColor: '#f0f0f0',
    borderRadius: 10,
    marginRight: 15,
    overflow: 'hidden',
  },
  confidenceProgress: {
    height: '100%',
    backgroundColor: '#667eea',
    borderRadius: 10,
  },
  confidenceText: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#333',
    minWidth: 50,
  },
  accuracyText: {
    fontSize: 14,
    color: '#666',
    textAlign: 'center',
  },
  predictionsCard: {
    backgroundColor: 'white',
    padding: 20,
    borderRadius: 12,
    marginBottom: 20,
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.1,
    shadowRadius: 3.84,
    elevation: 5,
  },
  predictionsTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 15,
    textAlign: 'center',
  },
  predictionsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
  },
  predictionItem: {
    backgroundColor: '#f8f9fa',
    padding: 12,
    borderRadius: 8,
    width: '48%',
    marginBottom: 10,
    alignItems: 'center',
  },
  predictionHeader: {
    alignItems: 'center',
    marginBottom: 8,
  },
  predictionDay: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#333',
  },
  predictionDate: {
    fontSize: 12,
    color: '#666',
  },
  disclaimerCard: {
    backgroundColor: '#fff3cd',
    padding: 15,
    borderRadius: 8,
    marginBottom: 20,
    flexDirection: 'row',
    alignItems: 'flex-start',
  },
  disclaimerText: {
    fontSize: 12,
    color: '#856404',
    marginLeft: 10,
    flex: 1,
    lineHeight: 18,
  },
  featuresCard: {
    backgroundColor: 'white',
    padding: 20,
    margin: 20,
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
  featuresTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 15,
    textAlign: 'center',
  },
  featureItem: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    marginBottom: 12,
  },
  featureText: {
    fontSize: 14,
    color: '#666',
    marginLeft: 10,
    flex: 1,
    lineHeight: 20,
  },
});

export default MLPredictionsScreen;
