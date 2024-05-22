const tf = require('@tensorflow/tfjs-node');
const InputError = require('../exceptions/InputError');
 
async function predictClassification(model, image) {
    try {
        const tensor = tf.node
            .decodeJpeg(image)
            .resizeNearestNeighbor([224, 224])
            .expandDims()
            .toFloat()
 
        const classes = ['Cancer', 'Non-cancer'];
 
        const prediction = model.predict(tensor);
        const score = await prediction.dataSync();
        const confidenceScore = Math.max(...score) * 100;
 
        let label, suggestion;
 
        if(confidenceScore > 50) {
            label = classes[0];
            suggestion = "Segera periksa ke dokter !";
        } else {
            label = classes[1];
            suggestion = "Anda tidak perlu khawatir, namun tetap periksa ke dokter untuk memastikan";
        }

        return { label, suggestion };
    } catch (error) {
        throw new InputError(`Terjadi kesalahan dalam melakukan prediksi`);
    }
}
 
module.exports = predictClassification;