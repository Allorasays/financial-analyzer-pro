/**
 * Complete Data Models for Financial Analyzer Pro Android App
 * Includes ALL advanced features from the working web platform
 * Package: com.financialanalyzer.mobile.data.model
 */

package com.financialanalyzer.mobile.data.model

import com.google.gson.annotations.SerializedName

// ========================================================================
// CORE FINANCIAL DATA MODELS
// Updated to match backend /api/financials/{ticker} response structure
// ========================================================================

// Backend returns flat structure - matches proxy.py get_financial_metrics response
data class FinancialDataResponse(
    val ticker: String,
    val company_name: String? = null,
    val industry: String? = null,
    val sector: String? = null,
    val website: String? = null,
    val description: String? = null,
    
    // Market Data
    val current_price: Double? = null,
    val previous_close: Double? = null,
    val market_cap: Long? = null,
    val enterprise_value: Long? = null,
    val shares_outstanding: Long? = null,
    val float_shares: Long? = null,
    val shares_short: Long? = null,
    val short_ratio: Double? = null,
    @SerializedName("52_week_high") val week52_high: Double? = null,
    @SerializedName("52_week_low") val week52_low: Double? = null,
    
    // Valuation Ratios
    val pe_ratio: Double? = null,
    val forward_pe: Double? = null,
    val peg_ratio: Double? = null,
    val price_to_book: Double? = null,
    val price_to_sales: Double? = null,
    val enterprise_value_to_revenue: Double? = null,
    val enterprise_value_to_ebitda: Double? = null,
    val ev_to_revenue: Double? = null,
    val ev_to_ebitda: Double? = null,
    
    // Profitability Metrics
    val revenue: Long? = null,
    val revenue_per_share: Double? = null,
    val revenue_growth: Double? = null,
    val net_income: Long? = null,
    val net_income_common: Long? = null,
    val earnings_per_share: Double? = null,
    val forward_eps: Double? = null,
    val earnings_growth: Double? = null,
    val earnings_quarterly_growth: Double? = null,
    
    // Margins
    val gross_margin: Double? = null,
    val operating_margin: Double? = null,
    val profit_margin: Double? = null,
    val ebitda_margin: Double? = null,
    
    // Cash Flow
    val ebitda: Long? = null,
    val free_cash_flow: Long? = null,
    val operating_cash_flow: Long? = null,
    val cash_per_share: Double? = null,
    
    // Returns
    val return_on_equity: Double? = null,
    val return_on_assets: Double? = null,
    val return_on_invested_capital: Double? = null,
    
    // Debt & Liquidity
    val debt_to_equity: Double? = null,
    val debt_to_assets: Double? = null,
    val current_ratio: Double? = null,
    val quick_ratio: Double? = null,
    val cash_ratio: Double? = null,
    val total_debt: Long? = null,
    val total_cash: Long? = null,
    val total_cash_per_share: Double? = null,
    
    // Dividends
    val dividend_yield: Double? = null,
    val dividend_rate: Double? = null,
    val dividend_per_share: Double? = null,
    val payout_ratio: Double? = null,
    val ex_dividend_date: Long? = null,
    val dividend_date: Long? = null,
    
    // Trading Metrics
    val beta: Double? = null,
    val volume: Long? = null,
    val average_volume: Long? = null,
    @SerializedName("average_volume_10days") val average_volume_10days: Long? = null,
    val bid: Double? = null,
    val ask: Double? = null,
    val bid_size: Int? = null,
    val ask_size: Int? = null,
    val day_low: Double? = null,
    val day_high: Double? = null,
    val open: Double? = null,
    
    // Analyst Data
    val target_high_price: Double? = null,
    val target_low_price: Double? = null,
    val target_mean_price: Double? = null,
    val target_median_price: Double? = null,
    val recommendation_mean: Double? = null,
    val recommendation_key: String? = null,
    val number_of_analyst_opinions: Int? = null,
    
    // Additional Metrics
    val book_value: Double? = null,
    val price_to_sales_trailing_12months: Double? = null,
    val held_percent_insiders: Double? = null,
    val held_percent_institutions: Double? = null,
    
    // Metadata
    val timestamp: String? = null,
    val data_source: String? = null
)

// Legacy models kept for backward compatibility (if needed elsewhere)
data class FinancialData(
    val ticker: String,
    val companyName: String,
    val industry: String,
    val sector: String,
    val marketCap: Long,
    val revenue: List<RevenueData>,
    val netIncome: List<IncomeData>,
    val ebitda: List<EBITDAData>,
    val freeCashFlow: List<CashFlowData>,
    val growthRates: GrowthRates,
    val keyMetrics: KeyMetrics
)

data class RevenueData(
    val year: Int,
    val quarter: String?,
    val value: Double,
    val growth: Double?
)

data class IncomeData(
    val year: Int,
    val quarter: String?,
    val value: Double,
    val growth: Double?
)

data class EBITDAData(
    val year: Int,
    val quarter: String?,
    val value: Double,
    val margin: Double
)

data class CashFlowData(
    val year: Int,
    val quarter: String?,
    val value: Double,
    val growth: Double?
)

data class GrowthRates(
    val revenue: Double,
    val netIncome: Double,
    val ebitda: Double,
    val freeCashFlow: Double
)

data class KeyMetrics(
    val pe: Double?,
    val ps: Double?,
    val pb: Double?,
    val peg: Double?,
    val evEbitda: Double?,
    val roe: Double?,
    val roa: Double?,
    val debtToEquity: Double?
)

// ========================================================================
// PEER COMPARISON MODELS
// ========================================================================

data class PeerComparisonResponse(
    val success: Boolean,
    val data: PeerComparison,
    val message: String? = null
)

data class PeerComparison(
    val ticker: String,
    val peers: List<PeerCompany>,
    val industryBenchmarks: IndustryBenchmarks,
    val relativePerformance: RelativePerformance
)

data class PeerCompany(
    val ticker: String,
    val name: String,
    val marketCap: Long,
    val pe: Double?,
    val ps: Double?,
    val pb: Double?,
    val revenueGrowth: Double,
    val netIncomeGrowth: Double,
    val roe: Double?
)

data class IndustryBenchmarks(
    val avgPe: Double,
    val avgPs: Double,
    val avgPb: Double,
    val avgGrowth: Double,
    val avgRoe: Double
)

data class RelativePerformance(
    val peRank: Int,
    val psRank: Int,
    val pbRank: Int,
    val growthRank: Int,
    val overallRank: Int
)

// ========================================================================
// TECHNICAL ANALYSIS MODELS
// ========================================================================

data class TechnicalAnalysisResponse(
    val success: Boolean,
    val data: TechnicalAnalysis,
    val message: String? = null
)

data class TechnicalAnalysis(
    val ticker: String,
    val indicators: TechnicalIndicators,
    val signals: TradingSignals,
    val supportResistance: SupportResistance,
    val trendAnalysis: TrendAnalysis
)

data class TechnicalIndicators(
    val sma: Map<String, Double>, // period -> value
    val ema: Map<String, Double>,
    val rsi: Double,
    val macd: MACDData,
    val bollingerBands: BollingerBands,
    val stochastic: StochasticData,
    val williamsR: Double,
    val cci: Double,
    val adx: Double
)

data class MACDData(
    val macd: Double,
    val signal: Double,
    val histogram: Double
)

data class BollingerBands(
    val upper: Double,
    val middle: Double,
    val lower: Double,
    val bandwidth: Double
)

data class StochasticData(
    val k: Double,
    val d: Double,
    val j: Double
)

data class TradingSignals(
    val overall: String, // "BUY", "SELL", "HOLD"
    val rsi: String,
    val macd: String,
    val bollinger: String,
    val stochastic: String,
    val adx: String,
    val confidence: Double
)

data class SupportResistance(
    val support: List<Double>,
    val resistance: List<Double>,
    val currentLevel: String
)

data class TrendAnalysis(
    val shortTerm: String, // "BULLISH", "BEARISH", "SIDEWAYS"
    val mediumTerm: String,
    val longTerm: String,
    val strength: Double
)

// ========================================================================
// MACHINE LEARNING & PREDICTIONS MODELS
// ========================================================================

data class MLPredictionsResponse(
    val success: Boolean,
    val data: MLPredictions,
    val message: String? = null
)

data class MLPredictions(
    val ticker: String,
    val predictions: List<PricePrediction>,
    val modelInfo: ModelInfo,
    val accuracy: Double,
    val confidence: Double
)

data class PricePrediction(
    val date: String,
    val predictedPrice: Double,
    val confidence: Double,
    val factors: List<String>
)

data class ModelInfo(
    val modelType: String,
    val features: List<String>,
    val trainingDate: String,
    val performance: ModelPerformance
)

data class ModelPerformance(
    val rmse: Double,
    val mae: Double,
    val r2: Double,
    val accuracy: Double
)

data class MLTrainingRequest(
    val ticker: String,
    val modelType: String,
    val features: List<String>,
    val parameters: Map<String, Any>
)

data class MLTrainingResponse(
    val success: Boolean,
    val modelId: String,
    val performance: ModelPerformance,
    val message: String? = null
)

// ========================================================================
// REAL-TIME DATA MODELS
// ========================================================================

data class RealtimeMarketOverviewResponse(
    val success: Boolean,
    val data: RealtimeMarketOverview,
    val message: String? = null
)

data class RealtimeMarketOverview(
    val timestamp: Long,
    val indices: Map<String, MarketIndex>,
    val trending: List<TrendingStock>,
    val volume: MarketVolume,
    val sentiment: MarketSentiment
)

data class MarketIndex(
    val symbol: String,
    val name: String,
    val price: Double,
    val change: Double,
    val changePercent: Double,
    val volume: Long
)

data class TrendingStock(
    val symbol: String,
    val name: String,
    val price: Double,
    val change: Double,
    val changePercent: Double,
    val volume: Long,
    val trend: String
)

data class MarketVolume(
    val total: Long,
    val average: Long,
    val ratio: Double
)

data class MarketSentiment(
    val overall: String, // "BULLISH", "BEARISH", "NEUTRAL"
    val score: Double,
    val fearGreed: Double,
    val vix: Double?
)

data class RealtimeStockResponse(
    val success: Boolean,
    val data: RealtimeStock,
    val message: String? = null
)

data class RealtimeStock(
    val symbol: String,
    val price: Double,
    val change: Double,
    val changePercent: Double,
    val volume: Long,
    val timestamp: Long,
    val bid: Double?,
    val ask: Double?,
    val high: Double,
    val low: Double,
    val open: Double
)

// ========================================================================
// GLOBAL MARKETS MODELS
// ========================================================================

data class GlobalMarketsResponse(
    val success: Boolean,
    val data: GlobalMarkets,
    val message: String? = null
)

data class GlobalMarkets(
    val indices: List<GlobalIndex>,
    val currencies: List<Currency>,
    val commodities: List<Commodity>,
    val bonds: List<Bond>
)

data class GlobalIndex(
    val symbol: String,
    val name: String,
    val country: String,
    val price: Double,
    val change: Double,
    val changePercent: Double
)

data class Currency(
    val pair: String,
    val rate: Double,
    val change: Double,
    val changePercent: Double
)

data class Commodity(
    val symbol: String,
    val name: String,
    val price: Double,
    val change: Double,
    val changePercent: Double,
    val unit: String
)

data class Bond(
    val symbol: String,
    val name: String,
    val yield: Double,
    val change: Double,
    val maturity: String
)

data class ForexDataResponse(
    val success: Boolean,
    val data: ForexData,
    val message: String? = null
)

data class ForexData(
    val majorPairs: List<ForexPair>,
    val minorPairs: List<ForexPair>,
    val exoticPairs: List<ForexPair>,
    val indices: List<CurrencyIndex>
)

data class ForexPair(
    val pair: String,
    val rate: Double,
    val change: Double,
    val changePercent: Double,
    val bid: Double,
    val ask: Double,
    val spread: Double
)

data class CurrencyIndex(
    val currency: String,
    val index: Double,
    val change: Double,
    val changePercent: Double
)

data class CryptoMarketsResponse(
    val success: Boolean,
    val data: CryptoMarkets,
    val message: String? = null
)

data class CryptoMarkets(
    val marketCap: Double,
    val volume24h: Double,
    val dominance: Map<String, Double>,
    val topCoins: List<CryptoCoin>
)

data class CryptoCoin(
    val symbol: String,
    val name: String,
    val price: Double,
    val change24h: Double,
    val changePercent24h: Double,
    val marketCap: Double,
    val volume24h: Double
)

// ========================================================================
// PORTFOLIO MANAGEMENT MODELS
// ========================================================================

data class PortfolioSummaryResponse(
    val success: Boolean,
    val data: PortfolioSummary,
    val message: String? = null
)

data class PortfolioSummary(
    val totalValue: Double,
    val totalCost: Double,
    val totalPnL: Double,
    val totalPnLPercent: Double,
    val positions: Int,
    val dayChange: Double,
    val dayChangePercent: Double,
    val allocation: Map<String, Double>, // sector -> percentage
    val performance: PortfolioPerformance
)

data class PortfolioPerformance(
    val daily: Double,
    val weekly: Double,
    val monthly: Double,
    val quarterly: Double,
    val yearly: Double,
    val ytd: Double
)

data class PortfolioPositionsResponse(
    val success: Boolean,
    val data: List<PortfolioPosition>,
    val message: String? = null
)

data class PortfolioPosition(
    val id: String,
    val symbol: String,
    val name: String,
    val shares: Int,
    val currentPrice: Double,
    val costBasis: Double,
    val marketValue: Double,
    val pnl: Double,
    val pnlPercent: Double,
    val allocation: Double,
    val sector: String,
    val lastUpdated: Long
)

data class AddPositionRequest(
    val symbol: String,
    val shares: Int,
    val costBasis: Double,
    val date: String
)

data class UpdatePositionRequest(
    val shares: Int?,
    val costBasis: Double?
)

data class PortfolioRiskMetrics(
    val volatility: Double,
    val sharpeRatio: Double,
    val maxDrawdown: Double,
    val var95: Double
)

// Portfolio Position from API response
data class PortfolioPositionApi(
    val ticker: String,
    val shares: Double,
    val avg_price: Double,
    val current_price: Double,
    val total_value: Double,
    val total_cost: Double,
    val gain_loss: Double,
    val gain_loss_pct: Double,
    val added_at: String
)

// Portfolio Summary from API response
data class PortfolioSummaryApi(
    val total_value: Double,
    val total_cost: Double,
    val total_gain_loss: Double,
    val total_gain_loss_pct: Double,
    val num_positions: Int
)

// Actual API response structure
data class PortfolioResponse(
    val portfolio: List<PortfolioPositionApi>,
    val summary: PortfolioSummaryApi
)

// Android model for internal use
data class Portfolio(
    val totalValue: Double,
    val totalCost: Double,
    val totalPnl: Double,
    val totalPnlPercent: Double,
    val positions: List<Position>,
    val riskMetrics: PortfolioRiskMetrics
)

// ========================================================================
// RISK ASSESSMENT MODELS
// ========================================================================

data class RiskAssessmentResponse(
    val success: Boolean,
    val data: RiskAssessment,
    val message: String? = null
)

data class RiskAssessment(
    val ticker: String,
    val overallRisk: String, // "LOW", "MEDIUM", "HIGH"
    val riskScore: Double,
    val factors: RiskFactors,
    val metrics: RiskMetrics,
    val recommendations: List<String>
)

data class RiskFactors(
    val volatility: Double,
    val beta: Double,
    val debtToEquity: Double,
    val currentRatio: Double,
    val interestCoverage: Double
)

data class RiskMetrics(
    val var95: Double, // Value at Risk 95%
    val cvar95: Double, // Conditional Value at Risk 95%
    val sharpeRatio: Double,
    val sortinoRatio: Double,
    val maxDrawdown: Double
)

// ========================================================================
// SENTIMENT ANALYSIS MODELS - REMOVED DUPLICATES
// ========================================================================

data class NewsItem(
    val title: String,
    val content: String,
    val sentiment: String,
    val score: Double,
    val date: String,
    val source: String,
    val url: String
)

// ========================================================================
// CHART & VISUALIZATION MODELS
// ========================================================================

data class PriceChartResponse(
    val success: Boolean,
    val data: PriceChart,
    val message: String? = null
)

data class PriceChart(
    val symbol: String,
    val period: String,
    val interval: String,
    val prices: List<ChartPoint>,
    val indicators: ChartIndicators
)

data class ChartPoint(
    val timestamp: Long,
    val date: String,
    val open: Double,
    val high: Double,
    val low: Double,
    val close: Double,
    val volume: Long
)

data class ChartIndicators(
    val sma: List<IndicatorPoint>,
    val ema: List<IndicatorPoint>,
    val rsi: List<IndicatorPoint>,
    val macd: List<MACDPoint>,
    val bollinger: List<BollingerPoint>
)

data class IndicatorPoint(
    val date: String,
    val value: Double
)

data class MACDPoint(
    val date: String,
    val macd: Double,
    val signal: Double,
    val histogram: Double
)

data class BollingerPoint(
    val date: String,
    val upper: Double,
    val middle: Double,
    val lower: Double
)

// ========================================================================
// NOTIFICATION & ALERTS MODELS
// ========================================================================

data class AlertsResponse(
    val success: Boolean,
    val data: List<Alert>,
    val message: String? = null
)

data class Alert(
    val id: String,
    val type: String, // "PRICE", "VOLUME", "VOLATILITY", "NEWS"
    val symbol: String?,
    val condition: String,
    val value: Double,
    val currentValue: Double,
    val status: String, // "ACTIVE", "TRIGGERED", "CANCELLED"
    val createdAt: String,
    val triggeredAt: String?
)

data class CreateAlertRequest(
    val type: String,
    val symbol: String?,
    val condition: String,
    val value: Double
)

data class UpdateAlertRequest(
    val condition: String?,
    val value: Double?,
    val status: String?
)

data class AlertResponse(
    val success: Boolean,
    val message: String? = null
)

// ========================================================================
// SYSTEM & HEALTH MODELS
// ========================================================================

data class HealthResponse(
    val success: Boolean,
    val status: String,
    val timestamp: Long,
    val version: String
)

data class SystemStatusResponse(
    val success: Boolean,
    val data: SystemStatus,
    val message: String? = null
)

data class SystemStatus(
    val api: String,
    val database: String,
    val realtime: String,
    val ml: String,
    val overall: String
)

data class VersionResponse(
    val success: Boolean,
    val data: VersionInfo,
    val message: String? = null
)

data class VersionInfo(
    val version: String,
    val build: String,
    val lastUpdated: String,
    val features: List<String>
)

// ========================================================================
// ERROR RESPONSE MODEL
// ========================================================================

data class ErrorResponse(
    val success: Boolean = false,
    val message: String,
    val code: Int? = null,
    val details: Map<String, Any>? = null
)

// ============================================================================
// ADDITIONAL RESPONSE MODELS FOR API SERVICE
// ============================================================================

data class StockDataResponse(
    @SerializedName("ticker") val ticker: String,
    @SerializedName("data") val data: MarketDataResponse
)

data class StockQuoteResponse(
    @SerializedName("ticker") val ticker: String,
    @SerializedName("price") val price: Double,
    @SerializedName("change") val change: Double,
    @SerializedName("change_percent") val changePercent: Double,
    @SerializedName("volume") val volume: Long,
    @SerializedName("timestamp") val timestamp: String
)

data class RealtimePortfolioResponse(
    @SerializedName("portfolio_value") val portfolioValue: Double,
    @SerializedName("total_change") val totalChange: Double,
    @SerializedName("total_change_percent") val totalChangePercent: Double,
    @SerializedName("positions") val positions: List<PortfolioPosition>,
    @SerializedName("timestamp") val timestamp: String
)

data class PriceAlertsResponse(
    @SerializedName("alerts") val alerts: List<PriceAlert>,
    @SerializedName("timestamp") val timestamp: String
)

data class PriceAlert(
    @SerializedName("ticker") val ticker: String,
    @SerializedName("alert_type") val alertType: String,
    @SerializedName("target_price") val targetPrice: Double,
    @SerializedName("current_price") val currentPrice: Double,
    @SerializedName("is_active") val isActive: Boolean
)

data class PerformanceAnalyticsResponse(
    @SerializedName("analytics") val analytics: PerformanceAnalytics,
    @SerializedName("timestamp") val timestamp: String
)

data class PerformanceAnalytics(
    @SerializedName("total_return") val totalReturn: Double,
    @SerializedName("annualized_return") val annualizedReturn: Double,
    @SerializedName("volatility") val volatility: Double,
    @SerializedName("sharpe_ratio") val sharpeRatio: Double,
    @SerializedName("max_drawdown") val maxDrawdown: Double
)

data class CorrelationAnalysisResponse(
    @SerializedName("correlations") val correlations: Map<String, Double>,
    @SerializedName("timestamp") val timestamp: String
)

data class IndustryAnalysisResponse(
    @SerializedName("industry") val industry: String,
    @SerializedName("analysis") val analysis: IndustryAnalysis,
    @SerializedName("timestamp") val timestamp: String
)

data class IndustryAnalysis(
    @SerializedName("sector_performance") val sectorPerformance: Double,
    @SerializedName("top_performers") val topPerformers: List<String>,
    @SerializedName("trends") val trends: List<String>
)

data class IndustryBenchmarksResponse(
    @SerializedName("benchmarks") val benchmarks: Map<String, Double>,
    @SerializedName("timestamp") val timestamp: String
)

data class IndustryPeersResponse(
    @SerializedName("peers") val peers: List<PeerStock>,
    @SerializedName("timestamp") val timestamp: String
)

data class PeerStock(
    @SerializedName("ticker") val ticker: String,
    @SerializedName("company_name") val companyName: String,
    @SerializedName("price") val price: Double,
    @SerializedName("change_percent") val changePercent: Double
)

data class VolumeChartResponse(
    @SerializedName("ticker") val ticker: String,
    @SerializedName("volume_data") val volumeData: List<VolumeData>,
    @SerializedName("timestamp") val timestamp: String
)

data class VolumeData(
    @SerializedName("date") val date: String,
    @SerializedName("volume") val volume: Long,
    @SerializedName("price") val price: Double
)

data class TechnicalChartResponse(
    @SerializedName("ticker") val ticker: String,
    @SerializedName("chart_data") val chartData: ChartData,
    @SerializedName("timestamp") val timestamp: String
)

data class ChartData(
    @SerializedName("dates") val dates: List<String>,
    @SerializedName("prices") val prices: List<Double>,
    @SerializedName("indicators") val indicators: Map<String, List<Double>>
)

data class ExportResponse(
    @SerializedName("export_id") val exportId: String,
    @SerializedName("file_url") val fileUrl: String,
    @SerializedName("format") val format: String,
    @SerializedName("status") val status: String,
    @SerializedName("timestamp") val timestamp: String
)

data class PerformanceReportResponse(
    @SerializedName("report_data") val reportData: Map<String, Any>,
    @SerializedName("format") val format: String,
    @SerializedName("timestamp") val timestamp: String
)

// ============================================================================
// ADDITIONAL API RESPONSE CLASSES FOR API SERVICE
// ============================================================================

data class MarketDataResponse(
    @SerializedName("ticker") val ticker: String,
    @SerializedName("period") val period: String,
    @SerializedName("timestamp") val timestamp: String,
    @SerializedName("data_points") val dataPoints: Int,
    @SerializedName("price_data") val priceData: PriceData,
    @SerializedName("technical_indicators") val technicalIndicators: TechnicalIndicators?,
    @SerializedName("risk_metrics") val riskMetrics: RiskMetrics?
)

data class PriceData(
    @SerializedName("dates") val dates: List<String>,
    @SerializedName("open") val open: List<Double>,
    @SerializedName("high") val high: List<Double>,
    @SerializedName("low") val low: List<Double>,
    @SerializedName("close") val close: List<Double>,
    @SerializedName("volume") val volume: List<Long>
)

// Note: TechnicalIndicators is already defined above (line 144) - removed duplicate

// Note: RiskMetrics is already defined above (line 520) - removed duplicate

data class MarketOverviewResponse(
    @SerializedName("timestamp") val timestamp: String,
    @SerializedName("indices") val indices: Map<String, IndexData>
)

data class IndexData(
    @SerializedName("name") val name: String,
    @SerializedName("symbol") val symbol: String,
    @SerializedName("price") val price: Double,
    @SerializedName("change") val change: Double,
    @SerializedName("change_percent") val changePercent: Double,
    @SerializedName("timestamp") val timestamp: String
)

data class BatchMarketDataResponse(
    @SerializedName("timestamp") val timestamp: String,
    @SerializedName("period") val period: String,
    @SerializedName("tickers") val tickers: Map<String, BatchTickerData>,
    @SerializedName("summary") val summary: BatchSummary
)

data class BatchTickerData(
    @SerializedName("price_data") val priceData: SimplePriceData,
    @SerializedName("technical_indicators") val technicalIndicators: SimpleTechnicalIndicators?
)

data class SimplePriceData(
    @SerializedName("dates") val dates: List<String>,
    @SerializedName("close") val close: List<Double>,
    @SerializedName("volume") val volume: List<Long>
)

data class SimpleTechnicalIndicators(
    @SerializedName("rsi") val rsi: List<Double>,
    @SerializedName("sma_20") val sma20: List<Double>,
    @SerializedName("macd") val macd: List<Double>
)

data class BatchSummary(
    @SerializedName("total_tickers") val totalTickers: Int,
    @SerializedName("successful_requests") val successfulRequests: Int,
    @SerializedName("failed_requests") val failedRequests: Int
)

data class PredictionsResponse(
    @SerializedName("ticker") val ticker: String,
    @SerializedName("prediction_days") val predictionDays: Int,
    @SerializedName("model_type") val modelType: String,
    @SerializedName("timestamp") val timestamp: String,
    @SerializedName("predictions") val predictions: Predictions,
    @SerializedName("model_metadata") val modelMetadata: ModelMetadata,
    @SerializedName("current_price") val currentPrice: Double? = null,
    @SerializedName("next_day") val nextDay: Double? = null,
    @SerializedName("next_week") val nextWeek: Double? = null,
    @SerializedName("next_month") val nextMonth: Double? = null,
    @SerializedName("next_quarter") val nextQuarter: Double? = null,
    @SerializedName("confidence_score") val confidenceScore: Double? = null,
    @SerializedName("model_metrics") val modelMetrics: ModelMetrics? = null,
    @SerializedName("data_points") val dataPoints: Int? = null,
    @SerializedName("features_used") val featuresUsed: Int? = null,
    @SerializedName("future_predictions") val futurePredictions: List<FuturePrediction>? = null,
    @SerializedName("status") val status: String? = null,
    @SerializedName("error") val error: String? = null
)

data class Predictions(
    @SerializedName("price_forecast") val priceForecast: List<Double>,
    @SerializedName("confidence_scores") val confidenceScores: List<Double>,
    @SerializedName("model_accuracy") val modelAccuracy: Double,
    @SerializedName("risk_assessment") val riskAssessment: String
)

data class ModelMetadata(
    @SerializedName("training_data_points") val trainingDataPoints: Int,
    @SerializedName("last_training_date") val lastTrainingDate: String?,
    @SerializedName("model_version") val modelVersion: String
)

data class ModelMetrics(
    @SerializedName("mse") val mse: Double,
    @SerializedName("rmse") val rmse: Double,
    @SerializedName("mae") val mae: Double,
    @SerializedName("r2_score") val r2Score: Double
)

data class FuturePrediction(
    @SerializedName("day") val day: Int,
    @SerializedName("predicted_price") val predictedPrice: Double,
    @SerializedName("date") val date: String,
    @SerializedName("confidence") val confidence: Double = 0.85 // Default confidence of 85%
)

data class StatusResponse(
    @SerializedName("status") val status: String,
    @SerializedName("timestamp") val timestamp: String,
    @SerializedName("available_endpoints") val availableEndpoints: List<String>,
    @SerializedName("data_sources") val dataSources: Map<String, String>
)

data class RiskAnalysisResponse(
    @SerializedName("ticker") val ticker: String,
    @SerializedName("period") val period: String,
    @SerializedName("timestamp") val timestamp: String,
    @SerializedName("risk_metrics") val riskMetrics: RiskAnalysisMetrics,
    @SerializedName("additional_metrics") val additionalMetrics: AdditionalRiskMetrics
)

data class RiskAnalysisMetrics(
    @SerializedName("volatility") val volatility: String,
    @SerializedName("sharpe_ratio") val sharpeRatio: String,
    @SerializedName("max_drawdown") val maxDrawdown: String,
    @SerializedName("var_95") val var95: String,
    @SerializedName("var_99") val var99: String
)

data class AdditionalRiskMetrics(
    @SerializedName("daily_returns") val dailyReturns: List<Double>,
    @SerializedName("volatility_daily") val volatilityDaily: Double,
    @SerializedName("skewness") val skewness: Double,
    @SerializedName("kurtosis") val kurtosis: Double,
    @SerializedName("sharpe_ratio_annual") val sharpeRatioAnnual: Double
)

// ============================================================================
// API RESPONSE WRAPPER
// ============================================================================

data class ApiResponse<T>(
    val data: T? = null,
    val error: String? = null,
    val isSuccess: Boolean = error == null
)

// ============================================================================
// PORTFOLIO POSITION CLASSES FOR ADAPTERS
// ============================================================================

data class Position(
    @SerializedName("ticker") val ticker: String,
    @SerializedName("shares") val shares: Double,
    @SerializedName("cost_basis") val costBasis: Double,
    @SerializedName("current_price") val currentPrice: Double,
    @SerializedName("value") val value: Double,
    @SerializedName("pnl") val pnl: Double,
    @SerializedName("pnl_percent") val pnlPercent: Double
)

data class StockSummary(
    @SerializedName("ticker") val ticker: String,
    @SerializedName("current_price") val currentPrice: Double,
    @SerializedName("change") val change: Double,
    @SerializedName("change_percent") val changePercent: Double,
    @SerializedName("volume") val volume: Long,
    @SerializedName("is_positive") val isPositive: Boolean = change >= 0
)

// ============================================================================
// SENTIMENT ANALYSIS RESPONSE MODELS - SIMPLIFIED VERSION
// Updated for Android app integration - Force recompilation
// ============================================================================

data class PlatformSentiment(
    @SerializedName("platform") val platform: String,
    @SerializedName("sentiment_score") val sentimentScore: Double,
    @SerializedName("sentiment_label") val sentimentLabel: String,
    @SerializedName("volume") val volume: Int,
    @SerializedName("confidence") val confidence: Double,
    @SerializedName("timestamp") val timestamp: String
)

data class SentimentSources(
    @SerializedName("twitter") val twitter: PlatformSentiment,
    @SerializedName("reddit") val reddit: PlatformSentiment,
    @SerializedName("news") val news: PlatformSentiment
)

data class SentimentSummary(
    @SerializedName("bullish_sources") val bullishSources: Int,
    @SerializedName("bearish_sources") val bearishSources: Int,
    @SerializedName("neutral_sources") val neutralSources: Int,
    @SerializedName("total_sources") val totalSources: Int
)

data class SentimentData(
    @SerializedName("overall_sentiment") val overallSentiment: String,
    @SerializedName("sentiment_score") val sentimentScore: Double,
    @SerializedName("confidence") val confidence: Double,
    @SerializedName("trend") val trend: String,
    @SerializedName("volume") val volume: Int,
    @SerializedName("sources") val sources: SentimentSources,
    @SerializedName("summary") val summary: SentimentSummary,
    @SerializedName("timestamp") val timestamp: String
)

data class SentimentAnalysisResponse(
    @SerializedName("success") val success: Boolean,
    @SerializedName("ticker") val ticker: String,
    @SerializedName("data") val data: SentimentData,
    @SerializedName("timestamp") val timestamp: String
)

data class ComprehensiveAnalysisResponse(
    @SerializedName("success") val success: Boolean,
    @SerializedName("data") val data: ComprehensiveAnalysisData
)

data class ComprehensiveAnalysisData(
    @SerializedName("ticker") val ticker: String,
    @SerializedName("current_price") val currentPrice: Double,
    @SerializedName("price_change") val priceChange: Double,
    @SerializedName("price_change_percent") val priceChangePercent: Double,
    @SerializedName("ml_predictions") val mlPredictions: PredictionsResponse,
    @SerializedName("sentiment_analysis") val sentimentAnalysis: SentimentData,
    @SerializedName("analysis_summary") val analysisSummary: AnalysisSummary,
    @SerializedName("timestamp") val timestamp: String
)

data class AnalysisSummary(
    @SerializedName("ml_signal") val mlSignal: Any, // Can be Double or String
    @SerializedName("sentiment_signal") val sentimentSignal: String,
    @SerializedName("combined_signal") val combinedSignal: String,
    @SerializedName("confidence") val confidence: Double
)

// ========================================================================
// AUTHENTICATION REQUEST/RESPONSE MODELS
// ========================================================================

data class ForgotPasswordRequest(
    val email: String
)

data class ForgotPasswordResponse(
    val message: String,
    val success: Boolean,
    @SerializedName("reset_token") val resetToken: String? = null,
    @SerializedName("reset_link") val resetLink: String? = null
)

data class ForgotUsernameRequest(
    val email: String
)

data class ForgotUsernameResponse(
    val message: String,
    val success: Boolean,
    val username: String? = null
)

data class ResetPasswordRequest(
    val token: String,
    @SerializedName("new_password") val newPassword: String
)

data class ResetPasswordResponse(
    val message: String,
    val success: Boolean
)