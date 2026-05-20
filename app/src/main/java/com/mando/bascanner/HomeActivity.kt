package com.mando.bascanner

import android.content.Intent
import android.os.Bundle
import android.widget.Button
import androidx.appcompat.app.AppCompatActivity
import com.mando.bascanner.ui.theme.BAScannerTheme


class HomeActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_home)

        findViewById<Button>(R.id.btn_ultrasound).setOnClickListener {
            launchAnalysisScreen(Modality.ULTRASOUND)
        }
        findViewById<Button>(R.id.btn_mammography).setOnClickListener {
            launchAnalysisScreen(Modality.MAMMOGRAPHY)
        }
        findViewById<Button>(R.id.btn_exterior).setOnClickListener {
            launchAnalysisScreen(Modality.EXTERIOR)
        }
    }

    private fun launchAnalysisScreen(modality: Modality) {
        val intent = Intent(this, AnalysisActivity::class.java)
        intent.putExtra("SELECTED_MODALITY", modality.name)
        startActivity(intent)
    }
}

