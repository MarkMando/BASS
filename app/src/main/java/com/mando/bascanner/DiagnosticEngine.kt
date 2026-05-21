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
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.exp

class DiagnosticEngine(context: Context, private val modelFileName: String) {

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
        } catch (_: Throwable) {
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
            } catch (_: Throwable) {
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

    fun analyzeImage(bitmap: Bitmap): Float = synchronized(this) {
        val tflite = interpreter ?: return 0f

        // 1. Prepare Input(s)
        val inputTensor = tflite.getInputTensor(0)
        val shape = inputTensor.shape() // Usually [1, H, W, 3]
        val h = if (shape[1] > 1) shape[1] else shape[2]
        val w = if (shape[2] > 1) shape[2] else shape[1]
        val dataType = inputTensor.dataType()

        val builder = ImageProcessor.Builder()
            .add(ResizeOp(h, w, ResizeOp.ResizeMethod.BILINEAR))
        
        if (dataType == DataType.FLOAT32) {
            // Heuristic for normalization: BUSI/Mammo models often use [0, 1], 
            // while MobileNet-based exterior models often use [-1, 1]
            if (modelFileName.contains("busi") || modelFileName.contains("mammo")) {
                builder.add(NormalizeOp(0f, 255f))
            } else {
                builder.add(NormalizeOp(127.5f, 127.5f))
            }
        }
        
        val imageProcessor = builder.build()

        var tensorImage = TensorImage(dataType)
        tensorImage.load(bitmap)
        tensorImage = imageProcessor.process(tensorImage)

        // Multiple Input Handling: Some models have auxiliary inputs (e.g. tensor 42)
        // that must be provided even if unused.
        val inputCount = tflite.inputTensorCount
        val inputs = arrayOfNulls<Any>(inputCount)
        inputs[0] = tensorImage.buffer
        for (i in 1 until inputCount) {
            val extraInput = tflite.getInputTensor(i)
            val dummyBuffer = ByteBuffer.allocateDirect(extraInput.numBytes())
            dummyBuffer.order(ByteOrder.nativeOrder())
            inputs[i] = dummyBuffer
        }

        // 2. Prepare Output(s)
        val outputTensor = tflite.getOutputTensor(0)
        val outputShape = outputTensor.shape()
        val numClasses = outputShape[outputShape.size - 1]
        
        val outputs = mutableMapOf<Int, Any>()
        
        return if (outputTensor.dataType() == DataType.FLOAT32) {
            val probabilityBuffer = Array(1) { FloatArray(numClasses) }
            outputs[0] = probabilityBuffer
            tflite.runForMultipleInputsOutputs(inputs, outputs)
            
            if (numClasses == 1) {
                val logit = probabilityBuffer[0][0]
                val prob = sigmoid(logit)
                // If it's the exterior model and the logit is negative for a positive case,
                // it likely means the classes are inverted (0=Malignant, 1=Benign or similar)
                if (modelFileName.contains("ext") && logit < 0) 1f - prob else prob
            } else {
                val probs = softmax(probabilityBuffer[0])
                // Heuristic: for the 2-class exterior model, index 0 is often Malignant.
                // For others (like BUSI), the last class is Malignant.
                if (numClasses == 2 && modelFileName.contains("ext")) probs[0] else probs[numClasses - 1]
            }
        } else {
            val probabilityBuffer = Array(1) { ByteArray(numClasses) }
            outputs[0] = probabilityBuffer
            tflite.runForMultipleInputsOutputs(inputs, outputs)
            val rawValue = (probabilityBuffer[0][numClasses - 1].toInt() and 0xFF) / 255.0f
            
            // Apply similar heuristic for quantized output if needed, 
            // but usually these are already probabilities.
            rawValue
        }
    }

    private fun sigmoid(x: Float): Float {
        return (1.0f / (1.0f + exp(-x.toDouble()).toFloat()))
    }

    private fun softmax(logits: FloatArray): FloatArray {
        var maxLogit = Float.NEGATIVE_INFINITY
        for (logit in logits) if (logit > maxLogit) maxLogit = logit
        var sum = 0f
        val probabilities = FloatArray(logits.size)
        for (i in logits.indices) {
            probabilities[i] = exp((logits[i] - maxLogit).toDouble()).toFloat()
            sum += probabilities[i]
        }
        for (i in probabilities.indices) probabilities[i] /= sum
        return probabilities
    }

    fun close() = synchronized(this) {
        interpreter?.close()
        gpuDelegate?.close()
        interpreter = null
        gpuDelegate = null
    }
}