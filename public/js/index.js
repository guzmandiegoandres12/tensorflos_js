'use strict'
//Importacion de librearia
import LinealRegresionModel from './model/LinealRegresionModel';
const data = require ('../data/data.json') ;

//Importacion de datos
const $stoptrainig = document.getElementById('stopTrainig');
const $reloadTrainig = document.getElementById('reloadTrainig');
const $save = document.getElementById('save');
const $loadModel = document.getElementById('loadModel');
const $generatePrediction = document.getElementById('generatePrediction');

/**
 * IMPLEMENTACION DE MODELO PARA REGRECION LINEAL
 */
//Limpieza y filtrados se datos
const dataCleanN1 = data.map( ({NumeroDeCuartosPromedio,Precio}) => ({precio:Precio ,cuartos:NumeroDeCuartosPromedio}));
//eliminacion de datos nulos
const dataCleanN2 = dataCleanN1.filter( ({precio,cuartos}) => (precio != null && cuartos != null));
const dataPrepared = dataCleanN2.map(({precio,cuartos}) => [cuartos,precio] )
const linearModel = new LinealRegresionModel(dataPrepared,'Cuartos','Presio');

linearModel.visualizeInitialData();
linearModel.createModel();


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

 

