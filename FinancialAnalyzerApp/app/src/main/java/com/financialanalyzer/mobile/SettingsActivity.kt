package com.financialanalyzer.mobile

import android.content.Intent
import android.os.Bundle
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.financialanalyzer.mobile.data.security.ApiKeyManager

class SettingsActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_settings)

        ApiKeyManager.migrateFromLegacyPrefs(this)

        val editFmp = findViewById<EditText>(R.id.editFmpKey)
        val editAv = findViewById<EditText>(R.id.editAlphaVantageKey)
        val editPolygon = findViewById<EditText>(R.id.editPolygonKey)
        val editTiingo = findViewById<EditText>(R.id.editTiingoKey)

        editFmp.setText(ApiKeyManager.getKey(this, ApiKeyManager.KEY_FMP).orEmpty())
        editAv.setText(ApiKeyManager.getKey(this, ApiKeyManager.KEY_ALPHA_VANTAGE).orEmpty())
        editPolygon.setText(ApiKeyManager.getKey(this, ApiKeyManager.KEY_POLYGON).orEmpty())
        editTiingo.setText(ApiKeyManager.getKey(this, ApiKeyManager.KEY_TIINGO).orEmpty())

        findViewById<Button>(R.id.btnSaveKeys).setOnClickListener {
            ApiKeyManager.saveKey(this, ApiKeyManager.KEY_FMP, editFmp.text.toString())
            ApiKeyManager.saveKey(this, ApiKeyManager.KEY_ALPHA_VANTAGE, editAv.text.toString())
            ApiKeyManager.saveKey(this, ApiKeyManager.KEY_POLYGON, editPolygon.text.toString())
            ApiKeyManager.saveKey(this, ApiKeyManager.KEY_TIINGO, editTiingo.text.toString())
            Toast.makeText(this, "API keys saved securely on this device", Toast.LENGTH_SHORT).show()
        }

        findViewById<Button>(R.id.btnClearKeys).setOnClickListener {
            listOf(
                ApiKeyManager.KEY_FMP,
                ApiKeyManager.KEY_ALPHA_VANTAGE,
                ApiKeyManager.KEY_POLYGON,
                ApiKeyManager.KEY_TIINGO
            ).forEach { ApiKeyManager.clearKey(this, it) }
            editFmp.text.clear()
            editAv.text.clear()
            editPolygon.text.clear()
            editTiingo.text.clear()
            Toast.makeText(this, "API keys cleared", Toast.LENGTH_SHORT).show()
        }

        findViewById<TextView>(R.id.txtPrivacy).setOnClickListener {
            openLegal("privacy")
        }
        findViewById<TextView>(R.id.txtTerms).setOnClickListener {
            openLegal("terms")
        }
    }

    private fun openLegal(page: String) {
        val intent = Intent(this, LegalWebViewActivity::class.java)
        intent.putExtra("page", page)
        startActivity(intent)
    }
}
