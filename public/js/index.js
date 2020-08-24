'use strict'
//Importacion de librearia
const tf = require('@tensorflow/tfjs');
const mobilnet =require('@tensorflow-models/mobilenet');

import LinealRegresionModel from './model/LinealRegresionModel';
const data = require ('../data/data.json') ;

//Importacion de datos
const $stoptrainig = document.getElementById('stopTrainig');
const $reloadTrainig = document.getElementById('reloadTrainig');
const $save = document.getElementById('save');
const $loadModel = document.getElementById('loadModel');
const $generatePrediction = document.getElementById('generatePrediction');
const $createModel = document.getElementById('createModel');

/**
 * IMPLEMENTACION DE MODELO PARA REGRECION LINEAL
 */
//Limpieza y filtrados se datos
const dataCleanN1 = data.map( ({NumeroDeCuartosPromedio,Precio}) => ({precio:Precio ,cuartos:NumeroDeCuartosPromedio}));
//eliminacion de datos nulos
const dataCleanN2 = dataCleanN1.filter( ({precio,cuartos}) => (precio != null && cuartos != null));
const dataPrepared = dataCleanN2.map(({precio,cuartos}) => [cuartos,precio] )
const linearModel = new LinealRegresionModel(dataPrepared,'Cuartos','Presio');

$createModel.addEventListener('click',()=>{
  linearModel.visualizeInitialData();
  linearModel.createModel();
})



$stoptrainig.addEventListener('click',()=>{
  linearModel.stopTraining();
})

//guardado de modelo
$save.addEventListener('click',async () =>{
  linearModel.save()
})

$loadModel.addEventListener('click',async ()=>{
  linearModel.loadModelbyFile('jsonModel','binModel')
})

//entrenar modelo
$reloadTrainig.addEventListener('click',async ()=>{
  await linearModel.trainModel();
})

//cargar modelo guardado
$generatePrediction.addEventListener('click',async()=>{
  linearModel.viewInferenceCurve()
})

/**
 * IMPLEMENTACION DE MODELO PARA RECONOCIMIENTO DE IMAGES
 */

let net;
const image =document.getElementById('img');
const $newImage = document.getElementById('newImage');
const $description = document.getElementById('decription');
const $camera = document.getElementById('camera');
const $pauserCamCapture = document.getElementById('pauserCamCapture');
const $imageCaptureDescription = document.getElementById('imageCaptureDescription');
let cameraActive = false;
image.src ='https://i.imgur.com/JlUvsxa.jpg';  
$pauserCamCapture.addEventListener('click',()=>{
  cameraActive = !cameraActive; 
 
  if(cameraActive){
    $pauserCamCapture.innerText = 'Desactivar Captura';
  }else{
    $pauserCamCapture.innerText = 'Activar Captura';
  }
  run();
})
const run = async() => {
  try {
    net = await mobilnet.load()
    
    const imageclasification =await net.classify(image) ;  
    $description.innerHTML = `<p>${JSON.stringify(imageclasification)}</p>`;
    const video = await tf.data.webcam($camera); 
    while(cameraActive){
      const imageCapture =await video.capture()
      const imagecaptureClasification =await net.classify(imageCapture) ;  
      $imageCaptureDescription.innerHTML = `<p>Prediccion: ${imagecaptureClasification[0].className}</p>
                                            <p>probabilidad: ${imagecaptureClasification[0].probability}</hp>`
      imageCapture.dispose();
      //espera el procesamiento del frame para provesar el sigiente 
      await tf.nextFrame();
    }
  } catch (error) {
    console.log(error);
    
  }

}

 image.onload = async ()=>{
  run()
}

$newImage.addEventListener('click',()=>{
  const number = Math.floor(Math.random()*1000);
  image.src = `https://picsum.photos/200/300?random=${number}`;
})
 


