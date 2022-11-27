const tf = require("@tensorflow/tfjs-node-gpu");

const model = await tf.loadLayersModel("model/model.json")

function process(image) {
    let tensor = tf.fromPixels(img)
    const resized = tf.image.resizeBilinear(tensor, [200, 200]).toFloat();

    const offset = tf.scalar(255.0);

    const normalized = tf.scalar(1.0).sub(resized.div(offset));

    const batched = normalized.expandDims(0);

    return batched
}

const pred = model.predict(process(img)).dataSync()

console.log("predictions: " + pred.toString());

// TODO: write them in a json