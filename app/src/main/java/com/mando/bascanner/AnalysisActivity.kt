package com.mando.bascanner
import android.Manifest
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.view.View
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class AnalysisActivity : AppCompatActivity() {

    private lateinit var diagnosticEngine: DiagnosticEngine
    private lateinit var currentModality: Modality
    private lateinit var cameraExecutor: ExecutorService
    private var cameraProvider: ProcessCameraProvider? = null
    @Volatile private var isClosing = false

    private lateinit var viewFinder: PreviewView
    private lateinit var staticImageView: ImageView
    private lateinit var tvResult: TextView

    // 1. Camera Permission Handler
    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        if (isGranted) startCamera() else {
            Toast.makeText(this, "Camera required for live scan.", Toast.LENGTH_LONG).show()
        }
    }

    // 2. Gallery Picker Handler
    private val pickMediaLauncher = registerForActivityResult(
        ActivityResultContracts.PickVisualMedia()
    ) { uri ->
        if (uri != null) {
            val bitmap = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                ImageDecoder.decodeBitmap(ImageDecoder.createSource(contentResolver, uri))
            } else {
                @Suppress("DEPRECATION")
                MediaStore.Images.Media.getBitmap(contentResolver, uri)
            }

            val softwareBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)

            // Switch UI to show static image instead of live feed
            viewFinder.visibility = View.GONE
            staticImageView.visibility = View.VISIBLE
            staticImageView.setImageBitmap(softwareBitmap)

            // Run Analysis
            val probability = diagnosticEngine.analyzeImage(softwareBitmap)
            tvResult.text = "Gallery Scan Probability: ${"%.2f".format(probability * 100)}%"
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_analysis)

        viewFinder = findViewById(R.id.viewFinder)
        staticImageView = findViewById(R.id.staticImageView)
        tvResult = findViewById(R.id.tv_result)

        val btnImport = findViewById<Button>(R.id.btn_import_gallery)

        // Parse Routing Intent
        val modalityName = intent.getStringExtra("SELECTED_MODALITY") ?: return
        currentModality = Modality.valueOf(modalityName)
        title = currentModality.title

        // Init Engine & Camera Executor
        try {
            diagnosticEngine = DiagnosticEngine(this, currentModality.modelFile)
        } catch (e: Exception) {
            Toast.makeText(this, "Failed to initialize ML Engine", Toast.LENGTH_LONG).show()
            finish()
            return
        }
        cameraExecutor = Executors.newSingleThreadExecutor()

        // Setup Button Listeners
        btnImport.setOnClickListener {
            pickMediaLauncher.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
        }

        // Start Camera Request Flow
        requestPermissionLauncher.launch(Manifest.permission.CAMERA)
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            val provider = cameraProviderFuture.get()
            cameraProvider = provider

            // Viewfinder Preview
            val preview = Preview.Builder().build().also {
                it.surfaceProvider = viewFinder.surfaceProvider
            }

            // Image Analyzer
            val imageAnalyzer = ImageAnalysis.Builder()
                .setTargetResolution(android.util.Size(480, 640))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor) { imageProxy ->
                        if (isClosing) {
                            imageProxy.close()
                            return@setAnalyzer
                        }
                        val bitmap = imageProxy.toBitmap()
                        val probability = diagnosticEngine.analyzeImage(bitmap)

                        // Update UI (Only if Gallery image isn't currently active)
                        if (staticImageView.visibility == View.GONE) {
                            runOnUiThread {
                                if (!isClosing) {
                                    tvResult.text = "Live Probability: ${"%.2f".format(probability * 100)}%"
                                }
                            }
                        }
                        imageProxy.close()
                    }
                }

            try {
                provider.unbindAll()
                provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, imageAnalyzer)
            } catch(exc: Exception) {
                // Log failure
            }
        }, ContextCompat.getMainExecutor(this))
    }

    override fun onDestroy() {
        isClosing = true
        cameraProvider?.unbindAll()
        super.onDestroy()
        cameraExecutor.shutdown()
        diagnosticEngine.close()
    }
}