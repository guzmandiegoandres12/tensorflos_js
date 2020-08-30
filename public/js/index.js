'use strict'
//Importacion de librearia
const tf = require('@tensorflow/tfjs');
const mobilnet =require('@tensorflow-models/mobilenet');
const knn = require('@tensorflow-models/knn-classifier')
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
  linearModel.loadModelbyFile('jsonModel','binModel');
})

//entrenar modelo
$reloadTrainig.addEventListener('click',async ()=>{
  await linearModel.trainModel();
})

//cargar modelo guardado
$generatePrediction.addEventListener('click',async()=>{
  linearModel.viewInferenceCurve(); 
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
const  $clssgato = document.getElementById('clssgato'); 
const  $clssdino = document.getElementById('clssdino');
const  $clssdiego = document.getElementById('clssdiego');
const  $clssok = document.getElementById('clssok');
const  $clssrock =document.getElementById('clssrock');
const  $console2 =document.getElementById('console2');

let video;
let cameraActive = false;
const classifier = knn.create();

$clssgato.addEventListener('click',()=>{
  addExample(0);
})
$clssdino.addEventListener('click',()=>{
  addExample(1);
})
$clssdiego.addEventListener('click',()=>{
  addExample(2);
})
$clssok.addEventListener('click',()=>{
  addExample(3);
})
$clssrock.addEventListener('click',()=>{
  addExample(4);
})

image.src ='https://i.imgur.com/JlUvsxa.jpg';

const run = async() => {
  try {
    console.log("run");
    net = await mobilnet.load()
    console.log("run2");
    const imageclasification =await net.classify(image) ;  
    $description.innerHTML = `<p>${JSON.stringify(imageclasification)}</p>`;
    video = await tf.data.webcam($camera);
    console.log("run3");
    
    while(cameraActive){
      console.log("Inivio");
      
      const imageCapture =await video.capture()
      const imagecaptureClasification =await net.classify(imageCapture) ; 
      console.log(imagecaptureClasification);
       
      const activation = net.infer(imageCapture,"conv_preds")
      let resul2;
      try {
        resul2 = await classifier.predictClass(activation);
        const elemnts = ['gato','dino','diego','ok','rock']; 
        console.log(elemnts[resul2.label]);
        
      } catch (error) {
        console.log(error);
        
      }
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

$pauserCamCapture.addEventListener('click',async ()=>{
  cameraActive = !cameraActive; 
  console.log('dasdasdada');
  
  if(cameraActive){
    $pauserCamCapture.innerText = 'Desactivar Captura';
    console.log('eeeeeee');
    
    await run();
  }else{
    $pauserCamCapture.innerText = 'Activar Captura';
  }
})

const  addExample =async (idclass) => {
  console.log(idclass);

  const img = await video.capture();
  const activacion= net.infer(img,true);
  classifier.addExample(activacion,idclass)
  img.dispose()
}
 image.onload = async ()=>{
  run()
}

$newImage.addEventListener('click',()=>{
  const number = Math.floor(Math.random()*1000);
  image.src = `https://picsum.photos/200/300?random=${number}`;
})
 


