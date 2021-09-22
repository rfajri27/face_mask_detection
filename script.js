let model;
const webcam = new Webcam(document.getElementById('wc'));
const MODEL_URL = 'http://127.0.0.1:8887/tfjs_model/model.json';

let isPredicting = false;

async function predict() {
  while (isPredicting) {
    const predictedClass = tf.tidy(() => {
      const img = webcam.capture();
      const predictions = model.predict(img);
      return predictions.as1D().argMax();
    });
    const classId = (await predictedClass.data())[0];
    var predictionText = "";
    switch(classId){
		case 0:
			predictionText = "Mask";
			break;
		case 1:
			predictionText = "No Mask";
			break;
	}
	document.getElementById("prediction").innerText = predictionText;


    predictedClass.dispose();
    await tf.nextFrame();
  }
}

function startPredicting(){
	isPredicting = true;
	predict();
}

function stopPredicting(){
	isPredicting = false;
	predict();
}

async function init(){
	await webcam.setup();
	model = await tf.loadLayersModel(MODEL_URL);
	tf.tidy(() => model.predict(webcam.capture()));
}

init();