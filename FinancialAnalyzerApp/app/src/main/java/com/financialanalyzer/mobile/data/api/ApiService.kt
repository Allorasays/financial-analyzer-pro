/**
 * API Service Interface for Financial Analyzer Android App
 * Package: com.financialanalyzer.mobile.data.api
 */

package com.financialanalyzer.mobile.data.api

import com.financialanalyzer.mobile.data.model.*
import retrofit2.Response
import retrofit2.http.*

/**
 * Retrofit API Service Interface
 * Defines all API endpoints for Financial Analyzer Pro
 * Connects to: https://financial-analyzer-pro-simple-z6jp.onrender.com
 */
interface FinancialAnalyzerApiService {
    
    // ========================================================================
    // FINANCIAL DATA ENDPOINTS (Comprehensive - uses FMP + yfinance)
    // ========================================================================
    
    @GET("api/financials/{ticker}")
    suspend fun getFinancialData(
        @Path("ticker") ticker: String
    ): Response<FinancialDataResponse>
    
    @GET("api/stock/{ticker}")
    suspend fun getStockData(
        @Path("ticker") ticker: String
    ): Response<StockDataResponse>
    
    @GET("api/peers/{ticker}")
    suspend fun getPeerComparison(
        @Path("ticker") ticker: String
    ): Response<PeerComparisonResponse>
    
    // ========================================================================
    // MARKET DATA ENDPOINTS
    // ========================================================================
    
    @GET("api/ai/market-data/{ticker}")
    suspend fun getMarketData(
        @Path("ticker") ticker: String,
        @Query("period") period: String = "1y"
    ): Response<MarketDataResponse>
    
    @GET("api/ai/market-overview")
    suspend fun getMarketOverview(): Response<MarketOverviewResponse>
    
    @GET("api/ai/global-markets")
    suspend fun getGlobalMarkets(): Response<GlobalMarketsResponse>
    
    @GET("api/ai/batch-market-data")
    suspend fun getBatchMarketData(
        @Query("tickers") tickers: List<String>
    ): Response<BatchMarketDataResponse>
    
    // ========================================================================
    // PORTFOLIO ENDPOINTS
    // ========================================================================
    
    @GET("api/ai/portfolio")
    suspend fun getPortfolioData(): Response<PortfolioResponse>
    
    // ========================================================================
    // TECHNICAL ANALYSIS ENDPOINTS
    // ========================================================================
    
    @GET("api/ai/technical-analysis/{ticker}")
    suspend fun getTechnicalAnalysis(
        @Path("ticker") ticker: String,
        @Query("period") period: String = "1y"
    ): Response<TechnicalAnalysisResponse>
    
    // ========================================================================
    // RISK ANALYSIS ENDPOINTS
    // ========================================================================
    
    @GET("api/ai/risk-analysis/{ticker}")
    suspend fun getRiskAnalysis(
        @Path("ticker") ticker: String,
        @Query("period") period: String = "1y"
    ): Response<RiskAnalysisResponse>
    
    // ========================================================================
    // ML PREDICTIONS ENDPOINTS
    // ========================================================================
    
    @GET("api/ml/predictions/{ticker}")
    suspend fun getPredictions(
        @Path("ticker") ticker: String,
        @Query("prediction_days") predictionDays: Int = 5
    ): Response<PredictionsResponse>
    
    // ========================================================================
    // SENTIMENT ANALYSIS ENDPOINTS
    // ========================================================================
    
    @GET("api/ai/sentiment/{ticker}")
    suspend fun getSentimentAnalysis(
        @Path("ticker") ticker: String
    ): Response<SentimentAnalysisResponse>
    
    @GET("api/ai/comprehensive-analysis/{ticker}")
    suspend fun getComprehensiveAnalysis(
        @Path("ticker") ticker: String,
        @Query("prediction_days") predictionDays: Int = 30
    ): Response<ComprehensiveAnalysisResponse>
    
    // ========================================================================
    // STATUS & HEALTH ENDPOINTS
    // ========================================================================
    
    @GET("api/ai/status")
    suspend fun getStatus(): Response<StatusResponse>
    
    @GET("api/ai/health")
    suspend fun getHealthCheck(): Response<HealthResponse>
    
    // ========================================================================
    // AUTHENTICATION ENDPOINTS
    // ========================================================================
    
    @POST("api/auth/forgot-password")
    suspend fun forgotPassword(
        @Body request: ForgotPasswordRequest
    ): Response<ForgotPasswordResponse>
    
    @POST("api/auth/forgot-username")
    suspend fun forgotUsername(
        @Body request: ForgotUsernameRequest
    ): Response<ForgotUsernameResponse>
    
    @POST("api/auth/reset-password")
    suspend fun resetPassword(
        @Body request: ResetPasswordRequest
    ): Response<ResetPasswordResponse>
}