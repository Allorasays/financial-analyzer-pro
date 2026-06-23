package com.financialanalyzer.mobile.data.security

import android.content.Context
import android.content.SharedPreferences
import androidx.security.crypto.EncryptedSharedPreferences
import androidx.security.crypto.MasterKey

/**
 * Stores personal API keys in EncryptedSharedPreferences (falls back to plain prefs if needed).
 */
object ApiKeyManager {

    const val KEY_FMP = "fmp_api_key"
    const val KEY_ALPHA_VANTAGE = "alpha_vantage_api_key"
    const val KEY_POLYGON = "polygon_api_key"
    const val KEY_TIINGO = "tiingo_api_key"

    private const val PREFS_NAME = "api_keys"

    private fun prefs(context: Context): SharedPreferences {
        return try {
            val masterKey = MasterKey.Builder(context)
                .setKeyScheme(MasterKey.KeyScheme.AES256_GCM)
                .build()
            EncryptedSharedPreferences.create(
                context,
                PREFS_NAME,
                masterKey,
                EncryptedSharedPreferences.PrefKeyEncryptionScheme.AES256_SIV,
                EncryptedSharedPreferences.PrefValueEncryptionScheme.AES256_GCM
            )
        } catch (e: Exception) {
            context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        }
    }

    fun getKey(context: Context, prefKey: String): String? {
        val key = prefs(context).getString(prefKey, null)?.trim()
        return if (key.isNullOrEmpty()) null else key
    }

    fun saveKey(context: Context, prefKey: String, value: String) {
        prefs(context).edit().putString(prefKey, value.trim()).apply()
    }

    fun clearKey(context: Context, prefKey: String) {
        prefs(context).edit().remove(prefKey).apply()
    }

    fun migrateFromLegacyPrefs(context: Context) {
        val legacy = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        val secure = prefs(context)
        if (legacy === secure) return

        val keys = listOf(KEY_FMP, KEY_ALPHA_VANTAGE, KEY_POLYGON, KEY_TIINGO)
        val editor = secure.edit()
        var migrated = false
        for (key in keys) {
            legacy.getString(key, null)?.trim()?.takeIf { it.isNotEmpty() }?.let { value ->
                editor.putString(key, value)
                migrated = true
            }
        }
        if (migrated) {
            editor.apply()
            legacy.edit().clear().apply()
        }
    }
}
