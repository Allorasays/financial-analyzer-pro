/**
 * Repository class for managing API calls
 * Package: com.financialanalyzer.mobile.data.repository
 */

package com.financialanalyzer.mobile.data.repository

import com.financialanalyzer.mobile.data.api.FinancialAnalyzerApiService
import com.financialanalyzer.mobile.data.model.*
import com.financialanalyzer.mobile.data.network.RetrofitClient
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * Repository class for managing API calls
 */
class FinancialAnalyzerRepository(
    private val apiService: FinancialAnalyzerApiService = RetrofitClient.apiService
) {
    
    // ========================================================================
    // MARKET DATA METHODS
    // ========================================================================
    
    suspend fun getMarketData(ticker: String, period: String = "1y"): ApiResponse<MarketDataResponse> {
        return withContext(Dispatchers.IO) {
            try {
                val response = apiService.getMarketData(ticker, period)
                if (response.isSuccessful && response.body() != null) {
                    ApiResponse(data = response.body())
                } else {
                    ApiResponse(error = "Failed to fetch market data: ${response.message()}")
                }
            } catch (e: Exception) {
                ApiResponse(error = "Network error: ${e.message}")
            }
        }
    }
    
    suspend fun getMarketOverview(): ApiResponse<MarketOverviewResponse> {
        return withContext(Dispatchers.IO) {
            try {
                val response = apiService.getMarketOverview()
                if (response.isSuccessful && response.body() != null) {
                    ApiResponse(data = response.body())
                } else {
                    ApiResponse(error = "Failed to fetch market overview: ${response.message()}")
                }
            } catch (e: Exception) {
                ApiResponse(error = "Network error: ${e.message}")
            }
        }
    }
    
    suspend fun getGlobalMarkets(): ApiResponse<GlobalMarketsResponse> {
        return withContext(Dispatchers.IO) {
            try {
                val response = apiService.getGlobalMarkets()
                if (response.isSuccessful && response.body() != null) {
                    ApiResponse(data = response.body())
                } else {
                    ApiResponse(error = "Failed to fetch global markets: ${response.message()}")
                }
            } catch (e: Exception) {
                ApiResponse(error = "Network error: ${e.message}")
            }
        }
    }
    
    // ========================================================================
    // PORTFOLIO METHODS
    // ========================================================================
    
    suspend fun getPortfolioData(): ApiResponse<PortfolioResponse> {
        return withContext(Dispatchers.IO) {
            try {
                val response = apiService.getPortfolioData()
                if (response.isSuccessful && response.body() != null) {
                    ApiResponse(data = response.body())
                } else {
                    ApiResponse(error = "Failed to fetch portfolio data: ${response.message()}")
                }
            } catch (e: Exception) {
                ApiResponse(error = "Network error: ${e.message}")
            }
        }
    }
    
    // ========================================================================
    // TECHNICAL ANALYSIS METHODS
    // ========================================================================
    
    suspend fun getTechnicalAnalysis(ticker: String, period: String = "1y"): ApiResponse<TechnicalAnalysisResponse> {
        return withContext(Dispatchers.IO) {
            try {
                val response = apiService.getTechnicalAnalysis(ticker, period)
                if (response.isSuccessful && response.body() != null) {
                    ApiResponse(data = response.body())
                } else {
                    ApiResponse(error = "Failed to fetch technical analysis: ${response.message()}")
                }
            } catch (e: Exception) {
                ApiResponse(error = "Network error: ${e.message}")
            }
        }
    }
    
    // ========================================================================
    // RISK ANALYSIS METHODS
    // ========================================================================
    
    suspend fun getRiskAnalysis(ticker: String, period: String = "1y"): ApiResponse<RiskAnalysisResponse> {
        return withContext(Dispatchers.IO) {
            try {
                val response = apiService.getRiskAnalysis(ticker, period)
                if (response.isSuccessful && response.body() != null) {
                    ApiResponse(data = response.body())
                } else {
                    ApiResponse(error = "Failed to fetch risk analysis: ${response.message()}")
                }
            } catch (e: Exception) {
                ApiResponse(error = "Network error: ${e.message}")
            }
        }
    }
    
    // ========================================================================
    // PREDICTIONS METHODS
    // ========================================================================
    
    suspend fun getPredictions(ticker: String, predictionDays: Int = 5): ApiResponse<PredictionsResponse> {
        return withContext(Dispatchers.IO) {
            try {
                val response = apiService.getPredictions(ticker, predictionDays)
                if (response.isSuccessful && response.body() != null) {
                    ApiResponse(data = response.body())
                } else {
                    ApiResponse(error = "Failed to fetch predictions: ${response.message()}")
                }
            } catch (e: Exception) {
                ApiResponse(error = "Network error: ${e.message}")
            }
        }
    }
    
    // ========================================================================
    // BATCH DATA METHODS
    // ========================================================================
    
    suspend fun getBatchMarketData(tickers: List<String>): ApiResponse<BatchMarketDataResponse> {
        return withContext(Dispatchers.IO) {
            try {
                val response = apiService.getBatchMarketData(tickers)
                if (response.isSuccessful && response.body() != null) {
                    ApiResponse(data = response.body())
                } else {
                    ApiResponse(error = "Failed to fetch batch market data: ${response.message()}")
                }
            } catch (e: Exception) {
                ApiResponse(error = "Network error: ${e.message}")
            }
        }
    }
    
    // ========================================================================
    // FINANCIAL DATA METHODS (backend-first)
    // ========================================================================

    suspend fun getFinancialData(ticker: String): ApiResponse<FinancialDataResponse> {
        return withContext(Dispatchers.IO) {
            try {
                val response = apiService.getFinancialData(ticker.uppercase())
                if (response.isSuccessful && response.body() != null) {
                    ApiResponse(data = response.body())
                } else {
                    ApiResponse(error = "Failed to fetch financial data: ${response.code()} ${response.message()}")
                }
            } catch (e: Exception) {
                ApiResponse(error = "Network error: ${e.message}")
            }
        }
    }

    // ========================================================================
    // STATUS METHODS
    // ========================================================================
    
    suspend fun getStatus(): ApiResponse<StatusResponse> {
        return withContext(Dispatchers.IO) {
            try {
                val response = apiService.getStatus()
                if (response.isSuccessful && response.body() != null) {
                    ApiResponse(data = response.body())
                } else {
                    ApiResponse(error = "Failed to fetch status: ${response.message()}")
                }
            } catch (e: Exception) {
                ApiResponse(error = "Network error: ${e.message}")
            }
        }
    }
    
    suspend fun getHealth(): ApiResponse<HealthResponse> {
        return withContext(Dispatchers.IO) {
            try {
                val response = apiService.getHealthCheck()
                if (response.isSuccessful && response.body() != null) {
                    ApiResponse(data = response.body())
                } else {
                    ApiResponse(error = "Failed to fetch health: ${response.message()}")
                }
            } catch (e: Exception) {
                ApiResponse(error = "Network error: ${e.message}")
            }
        }
    }
}
