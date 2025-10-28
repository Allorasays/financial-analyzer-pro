import requests
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import time
from fred_service import fred_service

logger = logging.getLogger(__name__)

class EconomicIndicatorsService:
    """Service for fetching additional economic indicators to enhance market confidence"""
    
    def __init__(self):
        self.fred_service = fred_service
        
    def get_housing_market_indicators(self) -> Dict[str, Any]:
        """Get housing market confidence and related indicators"""
        try:
            indicators = {}
            
            # Use existing FRED service methods and add custom series calls
            # Housing Market Index (Builder Confidence) - Custom series call
            hmi_data = self._get_fred_series('HMI')
            if hmi_data and hmi_data.get('observations'):
                latest_hmi = hmi_data['observations'][0]
                indicators['housing_market_index'] = {
                    'value': float(latest_hmi.get('value', 0)),
                    'date': latest_hmi.get('date', ''),
                    'description': 'Housing Market Index (Builder Confidence)',
                    'interpretation': self._interpret_hmi(float(latest_hmi.get('value', 0)))
                }
            
            # New Home Sales - Custom series call
            nhs_data = self._get_fred_series('HSN1F')
            if nhs_data and nhs_data.get('observations'):
                latest_nhs = nhs_data['observations'][0]
                indicators['new_home_sales'] = {
                    'value': float(latest_nhs.get('value', 0)),
                    'date': latest_nhs.get('date', ''),
                    'description': 'New Home Sales (Thousands)',
                    'interpretation': self._interpret_home_sales(float(latest_nhs.get('value', 0)))
                }
            
            # Existing Home Sales - Custom series call
            ehs_data = self._get_fred_series('EXHOSLUSM495S')
            if ehs_data and ehs_data.get('observations'):
                latest_ehs = ehs_data['observations'][0]
                indicators['existing_home_sales'] = {
                    'value': float(latest_ehs.get('value', 0)),
                    'date': latest_ehs.get('date', ''),
                    'description': 'Existing Home Sales (Seasonally Adjusted)',
                    'interpretation': self._interpret_home_sales(float(latest_ehs.get('value', 0)))
                }
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error fetching housing market indicators: {e}")
            return {}
    
    def get_foreclosure_indicators(self) -> Dict[str, Any]:
        """Get foreclosure and delinquency indicators"""
        try:
            indicators = {}
            
            # Mortgage Delinquency Rate - Custom series call
            mdr_data = self._get_fred_series('DRSFRMACBS')
            if mdr_data and mdr_data.get('observations'):
                latest_mdr = mdr_data['observations'][0]
                indicators['mortgage_delinquency_rate'] = {
                    'value': float(latest_mdr.get('value', 0)),
                    'date': latest_mdr.get('date', ''),
                    'description': 'Mortgage Delinquency Rate (%)',
                    'interpretation': self._interpret_delinquency_rate(float(latest_mdr.get('value', 0)))
                }
            
            # Foreclosure Rate - Custom series call
            fr_data = self._get_fred_series('FRRATE')
            if fr_data and fr_data.get('observations'):
                latest_fr = fr_data['observations'][0]
                indicators['foreclosure_rate'] = {
                    'value': float(latest_fr.get('value', 0)),
                    'date': latest_fr.get('date', ''),
                    'description': 'Foreclosure Rate (%)',
                    'interpretation': self._interpret_foreclosure_rate(float(latest_fr.get('value', 0)))
                }
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error fetching foreclosure indicators: {e}")
            return {}
    
    def get_consumer_spending_indicators(self) -> Dict[str, Any]:
        """Get consumer spending and confidence indicators"""
        try:
            indicators = {}
            
            # Personal Consumption Expenditures - Custom series call
            pce_data = self._get_fred_series('PCE')
            if pce_data and pce_data.get('observations'):
                latest_pce = pce_data['observations'][0]
                indicators['personal_consumption_expenditures'] = {
                    'value': float(latest_pce.get('value', 0)),
                    'date': latest_pce.get('date', ''),
                    'description': 'Personal Consumption Expenditures (Billions)',
                    'interpretation': self._interpret_pce(float(latest_pce.get('value', 0)))
                }
            
            # Consumer Confidence Index - Custom series call
            cci_data = self._get_fred_series('UMCSENT')
            if cci_data and cci_data.get('observations'):
                latest_cci = cci_data['observations'][0]
                indicators['consumer_confidence_index'] = {
                    'value': float(latest_cci.get('value', 0)),
                    'date': latest_cci.get('date', ''),
                    'description': 'Consumer Sentiment Index',
                    'interpretation': self._interpret_consumer_confidence(float(latest_cci.get('value', 0)))
                }
            
            # Retail Sales - Custom series call
            rs_data = self._get_fred_series('RSAFS')
            if rs_data and rs_data.get('observations'):
                latest_rs = rs_data['observations'][0]
                indicators['retail_sales'] = {
                    'value': float(latest_rs.get('value', 0)),
                    'date': latest_rs.get('date', ''),
                    'description': 'Retail Sales (Millions)',
                    'interpretation': self._interpret_retail_sales(float(latest_rs.get('value', 0)))
                }
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error fetching consumer spending indicators: {e}")
            return {}
    
    def _get_fred_series(self, series_id: str) -> Optional[Dict]:
        """Get FRED series data using the existing service"""
        try:
            params = {
                'series_id': series_id,
                'limit': 1,
                'sort_order': 'desc'
            }
            return self.fred_service._make_request('/series/observations', params)
        except Exception as e:
            logger.error(f"Error fetching FRED series {series_id}: {e}")
            return None
    
    def get_comprehensive_market_confidence(self) -> Dict[str, Any]:
        """Get comprehensive market confidence score based on all indicators"""
        try:
            housing_indicators = self.get_housing_market_indicators()
            foreclosure_indicators = self.get_foreclosure_indicators()
            consumer_indicators = self.get_consumer_spending_indicators()
            
            # Calculate confidence scores for each category
            housing_score = self._calculate_housing_confidence(housing_indicators)
            foreclosure_score = self._calculate_foreclosure_confidence(foreclosure_indicators)
            consumer_score = self._calculate_consumer_confidence(consumer_indicators)
            
            # Weighted overall confidence (housing 30%, foreclosure 20%, consumer 50%)
            overall_confidence = (housing_score * 0.3 + foreclosure_score * 0.2 + consumer_score * 0.5)
            
            return {
                'overall_market_confidence': round(overall_confidence, 2),
                'housing_confidence': round(housing_score, 2),
                'foreclosure_confidence': round(foreclosure_score, 2),
                'consumer_confidence': round(consumer_score, 2),
                'indicators': {
                    'housing': housing_indicators,
                    'foreclosure': foreclosure_indicators,
                    'consumer': consumer_indicators
                },
                'timestamp': datetime.now().isoformat(),
                'interpretation': self._interpret_overall_confidence(overall_confidence)
            }
            
        except Exception as e:
            logger.error(f"Error calculating comprehensive market confidence: {e}")
            return {'overall_market_confidence': 0.5, 'error': str(e)}
    
    def _interpret_hmi(self, value: float) -> str:
        """Interpret Housing Market Index"""
        if value >= 70:
            return "Very Strong - High builder confidence"
        elif value >= 60:
            return "Strong - Good builder confidence"
        elif value >= 50:
            return "Moderate - Neutral builder confidence"
        elif value >= 40:
            return "Weak - Low builder confidence"
        else:
            return "Very Weak - Very low builder confidence"
    
    def _interpret_home_sales(self, value: float) -> str:
        """Interpret home sales data"""
        if value > 1000:  # For new home sales
            return "Strong - High sales volume"
        elif value > 500:
            return "Moderate - Normal sales volume"
        else:
            return "Weak - Low sales volume"
    
    def _interpret_delinquency_rate(self, value: float) -> str:
        """Interpret mortgage delinquency rate"""
        if value <= 2.0:
            return "Excellent - Very low delinquency"
        elif value <= 4.0:
            return "Good - Low delinquency"
        elif value <= 6.0:
            return "Moderate - Normal delinquency"
        elif value <= 8.0:
            return "Concerning - High delinquency"
        else:
            return "Critical - Very high delinquency"
    
    def _interpret_foreclosure_rate(self, value: float) -> str:
        """Interpret foreclosure rate"""
        if value <= 0.5:
            return "Excellent - Very low foreclosure"
        elif value <= 1.0:
            return "Good - Low foreclosure"
        elif value <= 2.0:
            return "Moderate - Normal foreclosure"
        elif value <= 3.0:
            return "Concerning - High foreclosure"
        else:
            return "Critical - Very high foreclosure"
    
    def _interpret_pce(self, value: float) -> str:
        """Interpret Personal Consumption Expenditures"""
        if value > 15000:  # Billions
            return "Strong - High consumer spending"
        elif value > 12000:
            return "Moderate - Normal consumer spending"
        else:
            return "Weak - Low consumer spending"
    
    def _interpret_consumer_confidence(self, value: float) -> str:
        """Interpret Consumer Confidence Index"""
        if value >= 100:
            return "Excellent - Very high consumer confidence"
        elif value >= 80:
            return "Good - High consumer confidence"
        elif value >= 60:
            return "Moderate - Normal consumer confidence"
        elif value >= 40:
            return "Weak - Low consumer confidence"
        else:
            return "Critical - Very low consumer confidence"
    
    def _interpret_retail_sales(self, value: float) -> str:
        """Interpret Retail Sales"""
        if value > 600000:  # Millions
            return "Strong - High retail sales"
        elif value > 400000:
            return "Moderate - Normal retail sales"
        else:
            return "Weak - Low retail sales"
    
    def _calculate_housing_confidence(self, indicators: Dict) -> float:
        """Calculate housing market confidence score (0-1)"""
        if not indicators:
            return 0.5
        
        score = 0.5  # Base score
        
        # Housing Market Index (0-100 scale, normalize to 0-1)
        if 'housing_market_index' in indicators:
            hmi_value = indicators['housing_market_index']['value']
            score += (hmi_value - 50) / 100 * 0.3  # ±30% impact
        
        # Home Sales (normalize based on typical ranges)
        if 'new_home_sales' in indicators:
            nhs_value = indicators['new_home_sales']['value']
            score += (nhs_value - 600) / 1000 * 0.2  # ±20% impact
        
        return max(0.0, min(1.0, score))
    
    def _calculate_foreclosure_confidence(self, indicators: Dict) -> float:
        """Calculate foreclosure confidence score (0-1)"""
        if not indicators:
            return 0.5
        
        score = 0.5  # Base score
        
        # Lower delinquency/foreclosure rates = higher confidence
        if 'mortgage_delinquency_rate' in indicators:
            del_rate = indicators['mortgage_delinquency_rate']['value']
            score -= (del_rate - 3.0) / 10 * 0.4  # ±40% impact
        
        if 'foreclosure_rate' in indicators:
            fore_rate = indicators['foreclosure_rate']['value']
            score -= (fore_rate - 1.0) / 5 * 0.3  # ±30% impact
        
        return max(0.0, min(1.0, score))
    
    def _calculate_consumer_confidence(self, indicators: Dict) -> float:
        """Calculate consumer confidence score (0-1)"""
        if not indicators:
            return 0.5
        
        score = 0.5  # Base score
        
        # Consumer Confidence Index (normalize from typical range)
        if 'consumer_confidence_index' in indicators:
            cci_value = indicators['consumer_confidence_index']['value']
            score += (cci_value - 80) / 100 * 0.4  # ±40% impact
        
        # PCE and Retail Sales (normalize based on typical ranges)
        if 'personal_consumption_expenditures' in indicators:
            pce_value = indicators['personal_consumption_expenditures']['value']
            score += (pce_value - 13000) / 5000 * 0.2  # ±20% impact
        
        return max(0.0, min(1.0, score))
    
    def _interpret_overall_confidence(self, confidence: float) -> str:
        """Interpret overall market confidence score"""
        if confidence >= 0.8:
            return "Very Strong - Excellent market conditions"
        elif confidence >= 0.6:
            return "Strong - Good market conditions"
        elif confidence >= 0.4:
            return "Moderate - Mixed market conditions"
        elif confidence >= 0.2:
            return "Weak - Challenging market conditions"
        else:
            return "Critical - Very challenging market conditions"

# Create global instance
economic_indicators_service = EconomicIndicatorsService()

def test_economic_indicators():
    """Test economic indicators service"""
    print("Testing Economic Indicators Service...")
    
    # Test comprehensive market confidence
    print("Testing comprehensive market confidence...")
    confidence_data = economic_indicators_service.get_comprehensive_market_confidence()
    
    if 'error' not in confidence_data:
        print(f"[SUCCESS] Overall Market Confidence: {confidence_data['overall_market_confidence']}")
        print(f"[SUCCESS] Housing Confidence: {confidence_data['housing_confidence']}")
        print(f"[SUCCESS] Foreclosure Confidence: {confidence_data['foreclosure_confidence']}")
        print(f"[SUCCESS] Consumer Confidence: {confidence_data['consumer_confidence']}")
        print(f"[SUCCESS] Interpretation: {confidence_data['interpretation']}")
    else:
        print(f"[ERROR] Failed to get market confidence: {confidence_data['error']}")
    
    print("Economic Indicators Service test completed!")

if __name__ == "__main__":
    test_economic_indicators()
