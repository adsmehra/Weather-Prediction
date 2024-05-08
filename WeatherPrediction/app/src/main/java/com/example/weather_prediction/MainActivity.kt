package com.example.weather_prediction

import androidx.appcompat.app.AppCompatActivity
import android.content.Context
import android.content.res.AssetFileDescriptor
import android.os.Bundle
import android.text.Editable
import android.text.TextWatcher
import android.util.Log
import android.view.inputmethod.EditorInfo
import android.view.inputmethod.InputMethodManager
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import android.widget.Toast
import com.example.weather_prediction.ml.WeatherPredictor
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

class MainActivity : AppCompatActivity() {

    lateinit var predictBtn: Button
    lateinit var tempEt: EditText
    lateinit var humidEt: EditText
    lateinit var resultTv: TextView
    private lateinit var tflite: Interpreter
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        predictBtn = findViewById(R.id.predictBtn)
        tempEt = findViewById(R.id.tempET)
        humidEt = findViewById(R.id.humidET)
        resultTv = findViewById(R.id.resultTV)

        //When User presses Enter Button, Move the Cursor to Next EditText
        tempEt.addTextChangedListener(object : TextWatcher {
            override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) {}

            override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) {}

            override fun afterTextChanged(s: Editable?) {}
        })
        tempEt.setOnEditorActionListener { _, actionId, _ ->
            if (actionId == EditorInfo.IME_ACTION_NEXT) {
                humidEt.requestFocus()
                true
            } else {
                false
            }
        }

        //When User presses Enter button, Close the Input Method
        humidEt.addTextChangedListener(object : TextWatcher {
            override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) {}

            override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) {}

            override fun afterTextChanged(s: Editable?) {}

        })
        humidEt.setOnEditorActionListener { _, actionId, _ ->
            if (actionId == EditorInfo.IME_ACTION_DONE) {
                val inputMethodManager =
                    getSystemService(Context.INPUT_METHOD_SERVICE) as InputMethodManager
                inputMethodManager.hideSoftInputFromWindow(humidEt.windowToken, 0)
                humidEt.clearFocus()
                true
            } else {
                false
            }
        }

        try {
            tflite = Interpreter(loadModelFile())       //Initialize ML Model
            predictBtn.setOnClickListener {
                resultTv.text=null

                val weatherData = fetchData()       //Get Data from EditText Views
                if (weatherData[2]=="1"){       //Verify Data
                    val temp = weatherData[0]
                    val humidity = weatherData[1]
                    try {
                        val temperatureC = temp.toFloat()
                        val humidityPer = humidity.toFloat()
                        val model = WeatherPredictor.newInstance(this)      //Initialize Weather Predictor
                        val weather = predictWeather(temperatureC, humidityPer)     //Predict Weather

                        updateUI(weather)       //Update UI

                        model.close()       // Releases model resources if no longer used.
                    }
                    catch (e: NumberFormatException) {
                        // Handles the case where the user entered a non-numeric value
                        Toast.makeText(
                            this,
                            "Please enter valid numeric values for temperature and humidity",
                            Toast.LENGTH_SHORT
                        ).show()
                    }
                }
            }

        }
        //Handle Initialization Error
        catch (ex: Exception) {
            ex.printStackTrace()
            Toast.makeText(this, "Failed to initialize model: ${ex.message}", Toast.LENGTH_SHORT)
                .show()
        }
    }

    private fun updateUI(weatherInfo: String){
        //Get Data
        val weatherIcon = when(weatherInfo) {
            "Sunny" -> R.drawable.sunny
            "Cloudy" -> R.drawable.cloudy
            "Partly Cloudy" -> R.drawable.partly_cloudy
            "Rainy" -> R.drawable.rainy
            "Cold" -> R.drawable.cold
            else -> 0
        }

        // Update UI on Separate Thread
        runOnUiThread {
            resultTv.text = "Predicted Weather: \n$weatherInfo"
            resultTv.setCompoundDrawablesRelativeWithIntrinsicBounds(0, weatherIcon, 0, 0)
        }
    }

    private fun fetchData():Array<String>{
        val temp = tempEt.text.toString().trim()
        val humidity = humidEt.text.toString().trim()
        var flag = "0"      //Set Flag 0 to check if both Temperature and Humidity are Entered by User
        if (temp.isEmpty() || humidity.isEmpty()) {
            // Toast message indicating that the fields are empty
            Toast.makeText(
                this,
                "Please enter both temperature and humidity",
                Toast.LENGTH_SHORT
            ).show()
        }else{flag = "1"}
        return arrayOf(temp,humidity,flag)
    }

    private fun loadModelFile(): ByteBuffer {
        //Load TfLite Model from Assets Directory
        val fileDescriptor: AssetFileDescriptor = assets.openFd("Weather_predictor.tflite")
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declareLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declareLength)
    }

    private fun predictWeather(temperatureC: Float, humidityPer: Float): String {
        val byteBuffer =
            ByteBuffer.allocateDirect(2 * 4) // Assuming 2 input features and 4 bytes per float
        byteBuffer.order(ByteOrder.nativeOrder())
        // Creates inputs for reference.
        val inputFeature0 = byteBuffer.asFloatBuffer()
        inputFeature0.put(floatArrayOf(temperatureC, humidityPer))

        byteBuffer.rewind()     //Rewind Cursor to reading next value

        val outputs = Array(1) { FloatArray(5) }        // Runs model inference and gets result.
        tflite.run(inputFeature0, outputs)

        val maxIndex = outputs[0].indices.maxByOrNull { outputs[0][it] } ?: -1      //Get Index Value for Predicted Weather
        val predictedClassIndex = if (maxIndex != -1) maxIndex else 0


        val weatherConditions =
            arrayOf("Cloudy", "Cold", "Rainy", "Sunny", "Partly Cloudy")        //Default Set of Weather Data
        val predictedWeather = weatherConditions[predictedClassIndex]
        Log.d("Weather", "Predicted: $predictedClassIndex")

        return predictedWeather     //Return Predicted Weather
    }



    override fun onDestroy() {
        super.onDestroy()
        tflite.close()
    }
}