package com.mando.bascanner

enum class Modality(val title: String, val modelFile: String, val inputSize: Int) {
    ULTRASOUND("Ultrasound Analysis", "busi_float32.tflite", 224),
    MAMMOGRAPHY("Mammography Analysis", "mammo_native_float32.tflite", 224),
    EXTERIOR("Exterior Examination", "ext_mobilenetv3_int8.tflite", 224)
}