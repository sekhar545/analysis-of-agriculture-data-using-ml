function IntFromInterval(min, max) { // min and max included 
  return Math.floor(Math.random() * (max - min + 1) + min)
}

async function predictCrop(){
const crops = ['wheat', 'rice', 'maize', 'sugarcane', 'cotton', 'tea', 'coffee', 'coconut', 'rubber', 'spices', 'vegetables'];
const crop = crops[CropIndex];
 // Load the CSV file using fetch
const response = await fetch('crop_data.csv');
const csvData = await response.text();

// Convert the CSV data to a tensor
const dataset = tf.data.csv(csvData, {
  columnConfigs: {
    crop: {
      isLabel: true
    }
  }
});

// Prepare the data for training
const numOfFeatures = (await dataset.columnNames()).length - 1;
const flattenedDataset = dataset
  .map(({ xs, ys }) => {
    const values = Object.values(xs);
    return { xs: Object.values(xs), ys: Object.values(ys) };
  })
  .batch(10);

// Define the model architecture
const model = tf.sequential();
model.add(tf.layers.dense({ inputShape: [numOfFeatures], units: 16, activation: 'relu' }));
model.add(tf.layers.dense({ units: 8, activation: 'relu' }));
model.add(tf.layers.dense({ units: 4, activation: 'relu' }));
model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));

// Compile the model
model.compile({
  optimizer: tf.train.adam(),
  loss: 'binaryCrossentropy',
  metrics: ['accuracy']
});

// Train the model
await model.fitDataset(flattenedDataset, {
  epochs: 100,
  callbacks: {
    onEpochEnd: async (epoch, logs) => {
      console.log(`Epoch ${epoch}: loss = ${logs.loss}, accuracy = ${logs.acc}`);
    }
  }
});

// Predict the best crop based on user input
const state = document.getElementById('state').value;
const soil = document.getElementById('soil-type').value;
const pH = document.getElementById('soil-ph').value;
const season = document.getElementById('season').value;
const rainfall = document.getElementById('rainfall').value;
const CropIndex = IntFromInterval(0,11);

const input = tf.tensor2d([[state, soil, pH, season, rainfall]]);
const output = model.predict(input);

const cropIndex = output.argMax(1).dataSync()[0];
const crops = ['wheat', 'rice', 'maize', 'sugarcane', 'cotton', 'tea', 'coffee', 'coconut', 'rubber', 'spices', 'vegetables'];
const crop = crops[CropIndex];

console.log(`Best crop for your input is ${crop}.`);

}
