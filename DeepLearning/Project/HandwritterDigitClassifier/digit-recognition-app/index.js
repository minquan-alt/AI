
const express = require("express");
const path = require("path");
const tf = require("@tensorflow/tfjs-node");
const app = express();

const port = 8080;

app.use(express.static('public'));
app.use(express.urlencoded({ limit: '50mb', extended: true}));

let model;
async function loadModel() {
    model = await tf.loadGraphModel('file://./tfjs_model/model.json');
    console.log("Model loaded successfully");
}
loadModel();

app.post('/predict', express.json(), async (req, res) => {
    try {
      const imageData = req.body.imageData; // Dữ liệu ảnh từ canvas (hình dạng [1, 784])
      const tensor = tf.tensor(imageData, [1, 784]); // Chuyển đổi thành tensor với hình dạng [1, 784]
      const prediction = model.predict(tensor); // Dự đoán
      const probabilities = await prediction.data(); // Lấy xác suất
      res.json({ probabilities: Array.from(probabilities) }); // Trả về kết quả
    } catch (error) {
      console.error('Error during prediction:', error);
      res.status(500).json({ error: 'Prediction failed' });
    }
  });

app.listen(port, () => {
    console.log(`App is running on ${port}`);
})