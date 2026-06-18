package com.mando.bascanner

enum class Modality(val title: String, val modelFile: String, val inputSize: Int) {
    ULTRASOUND("Ultrasound Analysis", "ultrasound_mobilenetv3_int8", 224),
    MAMMOGRAPHY("Mammography Analysis", "mammo_mobilenetv3_int8.tflite", 224),
    EXTERIOR("Exterior Examination", "ext_mobilenetv3_int8.tflite", 224)
}