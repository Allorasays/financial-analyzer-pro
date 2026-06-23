/**
 * Retrofit Network Module
 * Package: com.financialanalyzer.mobile.data.network
 */

package com.financialanalyzer.mobile.data.network

import com.financialanalyzer.mobile.BuildConfig
import com.financialanalyzer.mobile.data.api.FinancialAnalyzerApiService
import com.google.gson.GsonBuilder
import okhttp3.OkHttpClient
import okhttp3.logging.HttpLoggingInterceptor
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import java.util.concurrent.TimeUnit

/**
 * Retrofit client configuration for Financial Analyzer API
 * Connects to the working web platform at https://financial-analyzer-pro-simple-z6jp.onrender.com
 */
object RetrofitClient {
    
    // Base URL for the working web platform
    // For development with ML and Sentiment Analysis, use local server
    // For production, use the Render backend API URL
    // Production URL: https://moneta-backend-api.onrender.com/
    // Local development: http://10.0.2.2:8000/ (Android emulator localhost)
    // If you have a different backend service, update this URL
    private const val BASE_URL = "https://moneta-backend-api.onrender.com/"
    
    // Timeout configurations
    private const val CONNECT_TIMEOUT = 30L
    private const val READ_TIMEOUT = 30L
    private const val WRITE_TIMEOUT = 30L
    
    // Create OkHttpClient with logging
    private val okHttpClient: OkHttpClient by lazy {
        val builder = OkHttpClient.Builder()
            .connectTimeout(CONNECT_TIMEOUT, TimeUnit.SECONDS)
            .readTimeout(READ_TIMEOUT, TimeUnit.SECONDS)
            .writeTimeout(WRITE_TIMEOUT, TimeUnit.SECONDS)
        
        // Log full bodies only in debug builds
        if (BuildConfig.DEBUG) {
            val loggingInterceptor = HttpLoggingInterceptor().apply {
                level = HttpLoggingInterceptor.Level.BODY
            }
            builder.addInterceptor(loggingInterceptor)
        }
        
        builder.build()
    }
    
    // Create Gson converter with custom configuration
    private val gsonConverter: GsonConverterFactory by lazy {
        GsonConverterFactory.create(
            GsonBuilder()
                .setLenient()
                .create()
        )
    }
    
    // Create Retrofit instance
    private val retrofit: Retrofit by lazy {
        Retrofit.Builder()
            .baseUrl(BASE_URL)
            .client(okHttpClient)
            .addConverterFactory(gsonConverter)
            .build()
    }
    
    // API service instance
    val apiService: FinancialAnalyzerApiService by lazy {
        retrofit.create(FinancialAnalyzerApiService::class.java)
    }
}
