package com.mando.bascanner
import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class DiagnosticEngine(context: Context, modelFileName: String, private val imageSize: Int) {

    private var interpreter: Interpreter? = null
    private var gpuDelegate: GpuDelegate? = null

    init {
        val modelBuffer = loadModelFile(context, modelFileName)
        
        try {
            // Attempt 1: Try with GPU if supported
            val options = Interpreter.Options().apply {
                setUseXNNPACK(false)
            }
            val compatList = CompatibilityList()
            if (compatList.isDelegateSupportedOnThisDevice) {
                gpuDelegate = GpuDelegate(compatList.bestOptionsForThisDevice)
                options.addDelegate(gpuDelegate)
            } else {
                options.setNumThreads(4)
            }
            interpreter = Interpreter(modelBuffer, options)
            // Force allocation to catch preparation errors early
            interpreter?.allocateTensors()
        } catch (t: Throwable) {
            gpuDelegate?.close()
            gpuDelegate = null
            
            // Fallback: Try with pure CPU (no GPU, no XNNPACK)
            try {
                val fallbackOptions = Interpreter.Options().apply {
                    setUseXNNPACK(false)
                    setNumThreads(4)
                }
                interpreter = Interpreter(modelBuffer, fallbackOptions)
                interpreter?.allocateTensors()
            } catch (t2: Throwable) {
                // Total failure
                interpreter = null
            }
        }
    }

    private fun loadModelFile(context: Context, modelName: String): MappedByteBuffer {
        val fileDescriptor = context.assets.openFd(modelName)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, fileDescriptor.startOffset, fileDescriptor.declaredLength)
    }

    fun analyzeImage(bitmap: Bitmap): Float {
        val tflite = interpreter ?: return -1f

        // 1. Dynamic Input Configuration
        val inputTensor = tflite.getInputTensor(0)
        val shape = inputTensor.shape() // Usually [1, H, W, 3]
        val h = if (shape[1] > 1) shape[1] else shape[2]
        val w = if (shape[2] > 1) shape[2] else shape[1]
        val dataType = inputTensor.dataType()

        val builder = ImageProcessor.Builder()
            .add(ResizeOp(h, w, ResizeOp.ResizeMethod.BILINEAR))
        
        if (dataType == DataType.FLOAT32) {
            builder.add(NormalizeOp(0f, 255f))
        }
        
        val imageProcessor = builder.build()

        var tensorImage = TensorImage(dataType)
        tensorImage.load(bitmap)
        tensorImage = imageProcessor.process(tensorImage)

        // 2. Flexible Output Handling
        val outputTensor = tflite.getOutputTensor(0)
        val outputShape = outputTensor.shape()
        val numClasses = outputShape[outputShape.size - 1]
        
        return if (outputTensor.dataType() == DataType.FLOAT32) {
            val probabilityBuffer = Array(1) { FloatArray(numClasses) }
            tflite.run(tensorImage.buffer, probabilityBuffer)
            // If binary classification with 1 output, return it. 
            // If multi-class or 2-class softmax, return the last class probability (assuming it's the 'positive' class)
            if (numClasses == 1) probabilityBuffer[0][0] else probabilityBuffer[0][numClasses - 1]
        } else {
            val probabilityBuffer = Array(1) { ByteArray(numClasses) }
            tflite.run(tensorImage.buffer, probabilityBuffer)
            val rawValue = probabilityBuffer[0][numClasses - 1].toInt() and 0xFF
            rawValue / 255.0f
        }
    }

    fun close() {
        interpreter?.close()
        gpuDelegate?.close()
        interpreter = null
        gpuDelegate = null
    }
}