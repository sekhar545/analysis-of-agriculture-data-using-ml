
// Import required machine learning libraries
import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';


function predictCrop(){
// Define the model architecture
const model = tf.sequential();
model.add(tf.layers.dense({inputShape: [7], units: 64, activation: 'relu'}));
model.add(tf.layers.dense({units: 64, activation: 'relu'}));
model.add(tf.layers.dense({units: 64, activation: 'relu'}));
model.add(tf.layers.dense({units: 4, activation: 'softmax'}));

// Define the loss function and optimizer
const loss = 'categoricalCrossentropy';
const optimizer = 'adam';

// Compile the model
model.compile({optimizer: optimizer, loss: loss});

// Define the training data
const trainingData = [
  {input: [1, 'Andhra Pradesh', 'Red Soil', 6.0, 7.5, 500, 1000], output: [1, 0, 0, 0]},
  {input: [1, 'Andhra Pradesh', 'Black Soil', 5.5, 7.0, 500, 1000], output: [0, 1, 0, 0]},
  {input: [1, 'Andhra Pradesh', 'Other Soil', 5.0, 6.5, 500, 1000], output: [0, 0, 1, 0]},
  {input: [2, 'Gujarat', 'Alluvial Soil', 6.0, 7.5, 500, 1000], output: [1, 0, 0, 0]},
  {input: [2, 'Gujarat', 'Red and Black Soil', 6.0, 7.0, 500, 1000], output: [0, 1, 0, 0]},
  {input: [2, 'Gujarat', 'Other Soil', 6.0, 7.0, 500, 1000], output: [0, 0, 1, 0]},
  {input: [3, 'Punjab', 'Sandy Soil', 6.0, 7.5, 500, 1000], output: [1, 0, 0, 0]},
  {input: [3, 'Punjab', 'Clayey Soil', 5.0, 6.5, 500, 1000], output: [0, 1, 0, 0]},
  {input: [3, 'Punjab', 'Other Soil', 5.5, 7.0, 500, 1000], output: [0, 0, 1, 0]},
];

// Convert training data to tensors
const inputTensor = tf.tensor2d(trainingData.map(item => item.input));
const outputTensor = tf.tensor2d(trainingData.map(item => item.output));

// Train the model
async function train() {
  const history = await model.fit(inputTensor, outputTensor, {
    epochs: 100,
    callbacks: tfvis.show.fitCallbacks({name: 'Training Performance'}, ['loss'])
  });
  console.log('Final loss:', history.history.loss[0]);
}

// Make predictions based on user input
// Make predictions based on user input
async function predict() {
  const season = document.getElementById('season').value;
  const state = document.getElementById('state').value;
  const soilType = document.getElementById('soilType').value;
  const ph = parseFloat(document.getElementById('ph').value);
  const rainfall = parseFloat(document.getElementById('rainfall').value);

  // Create input tensor from user input
  const input = [season, state, soilType, ph, rainfall];
  const inputTensor = tf.tensor2d([input]);

  // Make prediction using the model
  const output = model.predict(inputTensor);
  const prediction = Array.from(output.dataSync());
  console.log('Prediction:', prediction);
}
    
}
