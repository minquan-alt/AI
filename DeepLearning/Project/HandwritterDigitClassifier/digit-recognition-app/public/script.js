const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const clearBtn = document.getElementById('clear-btn');
const predictBtn = document.getElementById('predict-btn');
const resultDiv = document.getElementById('result');

let isDrawing = false;

ctx.fillStyle = '#000';
ctx.fillRect(0, 0, canvas.width, canvas.height);

canvas.addEventListener('mousedown', () => (isDrawing = true));
canvas.addEventListener('mousemove', (e) => {
  if (!isDrawing) return;
  ctx.fillStyle = '#FFF'; 
  ctx.fillRect(e.offsetX, e.offsetY, 20, 20);
});
canvas.addEventListener('mouseup', () => (isDrawing = false));

clearBtn.addEventListener('click', () => {
  ctx.fillStyle = '#000'; 
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  resultDiv.innerHTML = '';
});

predictBtn.addEventListener('click', async () => {
  const tempCanvas = document.createElement('canvas');
  const tempCtx = tempCanvas.getContext('2d');
  tempCanvas.width = 28;
  tempCanvas.height = 28;

  tempCtx.drawImage(canvas, 0, 0, 280, 280, 0, 0, 28, 28);
  const imageData = tempCtx.getImageData(0, 0, 28, 28);

  const data = [];
  for (let i = 0; i < imageData.data.length; i += 4) {
    data.push(imageData.data[i] / 255); 
  }
  const flattenedData = [data]; 

  const response = await fetch('/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ imageData: flattenedData }),
  });
  const { probabilities } = await response.json();
  const maxProbability = Math.max(...probabilities);
  const predictedNumber = probabilities.indexOf(maxProbability);

  resultDiv.innerHTML = `
    <h2>Prediction Probabilities:</h2>
    <p>Predicted Number: <strong>${predictedNumber}</strong> (Probability: ${(maxProbability * 100).toFixed(2)}%)</p>
    <ul>
      ${probabilities.map((prob, index) => `<p>${index}: ${(prob * 100).toFixed(2)}%</p>`).join('')}
    </ul>
  `;
});