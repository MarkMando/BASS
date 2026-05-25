package com.mando.bascanner
import android.Manifest
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageDecoder
import android.graphics.Matrix
import android.os.Build
import android.os.Bundle
import android.util.Size
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
import androidx.camera.core.ImageProxy
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
    private lateinit var tvModalityIndicator: TextView

    // Smoothing state for live analysis
    private var smoothedProbability = -1f
    private val smoothingFactor = 0.15f // Adjust between 0.0 and 1.0 for more/less smoothing

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
            val bitmap = try {
                val maxDimension = 1600 // Safer limit for various hardware canvases
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                    ImageDecoder.decodeBitmap(ImageDecoder.createSource(contentResolver, uri)) { decoder, info, _ ->
                        // Ensure we don't decode a massive bitmap into memory
                        if (info.size.width > maxDimension || info.size.height > maxDimension) {
                            val ratio = info.size.width.toFloat() / info.size.height.toFloat()
                            if (ratio > 1) {
                                decoder.setTargetSize(maxDimension, (maxDimension / ratio).toInt())
                            } else {
                                decoder.setTargetSize((maxDimension * ratio).toInt(), maxDimension)
                            }
                        }
                        // Force software allocator to avoid hardware canvas limits on some devices
                        decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
                    }
                } else {
                    val options = BitmapFactory.Options().apply { inJustDecodeBounds = true }
                    contentResolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it, null, options) }
                    
                    var inSampleSize = 1
                    while (options.outHeight / inSampleSize > maxDimension || options.outWidth / inSampleSize > maxDimension) {
                        inSampleSize *= 2
                    }

                    val finalOptions = BitmapFactory.Options().apply { 
                        this.inSampleSize = inSampleSize 
                        this.inPreferredConfig = Bitmap.Config.ARGB_8888
                    }
                    contentResolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it, null, finalOptions) }
                }
            } catch (_: Exception) {
                null
            }

            if (bitmap != null) {
                // Secondary check: If for some reason the decoder didn't scale enough
                val finalBitmap = if (bitmap.width > 2000 || bitmap.height > 2000) {
                    val scale = 1600f / Math.max(bitmap.width, bitmap.height)
                    val scaled = Bitmap.createScaledBitmap(bitmap, (bitmap.width * scale).toInt(), (bitmap.height * scale).toInt(), true)
                    if (scaled != bitmap) bitmap.recycle()
                    scaled
                } else {
                    bitmap
                }

                // Switch UI to show static image instead of live feed
                viewFinder.visibility = View.GONE
                staticImageView.visibility = View.VISIBLE
                
                // Clear old bitmap if necessary to free memory
                staticImageView.drawable?.let { 
                    staticImageView.setImageDrawable(null)
                    // If it was a bitmap drawable, we could recycle the bitmap, 
                    // but it's tricky if it's shared. For now, just setting null.
                }
                
                staticImageView.setImageBitmap(finalBitmap)

                // Run Analysis on background thread to avoid UI hangs and potential Canvas issues
                cameraExecutor.execute {
                    val probability = diagnosticEngine.analyzeImage(finalBitmap)
                    runOnUiThread {
                        if (!isClosing) {
                            tvResult.text = "Gallery Scan Malignancy Probability: ${"%.2f".format(probability * 100)}%"
                        }
                    }
                }
            } else {
                Toast.makeText(this, "Failed to load image", Toast.LENGTH_SHORT).show()
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_analysis)

        viewFinder = findViewById(R.id.viewFinder)
        staticImageView = findViewById(R.id.staticImageView)
        tvResult = findViewById(R.id.tv_result)
        tvModalityIndicator = findViewById(R.id.tv_modality_indicator)

        val btnImport = findViewById<Button>(R.id.btn_import_gallery)

        // Parse Routing Intent
        val modalityName = intent.getStringExtra("SELECTED_MODALITY") ?: return
        currentModality = Modality.valueOf(modalityName)
        title = currentModality.title
        tvModalityIndicator.text = currentModality.title

        // Init Engine & Camera Executor
        try {
            diagnosticEngine = DiagnosticEngine(this, currentModality.modelFile)
        } catch (_: Exception) {
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
            val targetSize = Size(480, 640)
            val imageAnalyzer = ImageAnalysis.Builder()
                .setTargetResolution(targetSize)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor) { imageProxy ->
                        if (isClosing || staticImageView.visibility == View.VISIBLE) {
                            imageProxy.close()
                            return@setAnalyzer
                        }
                        
                        // Scale directly to a manageable size to avoid Canvas draw errors
                        val bitmap = try {
                            val original = imageProxy.toBitmap()
                            val maxDim = 1024
                            if (original.width > maxDim || original.height > maxDim) {
                                val scale = maxDim.toFloat() / Math.max(original.width, original.height)
                                val matrix = Matrix().apply { postScale(scale, scale) }
                                val scaled = Bitmap.createBitmap(original, 0, 0, original.width, original.height, matrix, true)
                                if (scaled != original) original.recycle()
                                scaled
                            } else {
                                original
                            }
                        } catch (e: Exception) {
                            null
                        }

                        if (bitmap != null) {
                            val probability = diagnosticEngine.analyzeImage(bitmap)

                            // Apply Exponential Moving Average (EMA) for smoothing
                            if (smoothedProbability < 0) {
                                smoothedProbability = probability
                            } else {
                                smoothedProbability = (probability * smoothingFactor) + (smoothedProbability * (1 - smoothingFactor))
                            }

                            // Update UI (Only if Gallery image isn't currently active)
                            if (staticImageView.visibility == View.GONE) {
                                runOnUiThread {
                                    if (!isClosing) {
                                        tvResult.text = "Live Malignancy Probability: ${"%.2f".format(smoothedProbability * 100)}%"
                                    }
                                }
                            }
                            bitmap.recycle()
                        }

                        imageProxy.close()
                    }
                }

            try {
                provider.unbindAll()
                provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, imageAnalyzer)
            } catch(_: Exception) {
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