'use strict'
//Importacion de librearia
const tf = require('@tensorflow/tfjs');
const tf_vis = require('@tensorflow/tfjs-vis');
//Importacion de datos
const data = require ('../data/data.json') ;

const optimizer = tf.train.adam();
const functionLoss = tf.losses.meanSquaredError;
const metrics = ['mse'];

//Limpieza y filtrados se datos
const dataCleanN1 = data.map( ({NumeroDeCuartosPromedio,Precio}) => ({precio:Precio ,cuartos:NumeroDeCuartosPromedio}));
//eliminacion de datos nulos
const dataCleanN2 = dataCleanN1.filter( ({precio,cuartos}) => (precio != null && cuartos != null));

//Funcion de visualizacion  con tensorflow.js
const visualizer = (data) => {
  //asapracion de data a formato de data para funcion de visualizacio
  const values  = data.map( ({precio,cuartos}) => ({x:cuartos,y:precio}));
  //funcion de renderizacion de graficas
  tf_vis.render.scatterplot(
    {name:'Cuatos vs Presio'},
    {values:values},
    {
      xLabel:'Cueartos',
      yLabel:'Precio',
      height:300
    }
    )
}
//creacion de modelo
const createModel = ()=> {
  
  //creacion de modelo secuencial
  const model = tf.sequential();
  
  //definicion de capa oculta con resepcio de 1 dato
  model.add(tf.layers.dense({
    inputShape:[1],
    units:1,
    useBias: true
  }))

  //definicion de capa  de salida una neurona
  model.add(tf.layers.dense({
    units:1,
    useBias:true
  }))

  return model
}
//convercion de datos a tensores
const factoryTotensorData = (data) =>{
  return tf.tidy(() => {
    //mesclado de elementos del arry
    tf.util.shuffle(data);
    //extaxion de datos  (regularizacion)
    const inputs = data.map(({cuartos}) => cuartos);
    const labels = data.map(({precio}) => precio);
    //creacion de tensores con datos extraidos
    const tensorsInputs = tf.tensor2d(inputs,[inputs.length,1]);
    const tensorLabels = tf.tensor2d(labels,[labels.length,1]);

    //valores para desregularizacion
    const inputsMax = tensorsInputs.max();
    const inputsMin = tensorsInputs.min();
    const labelMax = tensorLabels.max();
    const lableMin = tensorLabels.min();

    //creacion de entradas normalizadas (dato-min)/(max-min) primitivas de tensor flosw de tensores
    const inpustNormalize = tensorsInputs.sub(inputsMin).div(inputsMax.sub(inputsMin));
    const labelsNormalize = tensorLabels.sub(lableMin).div(labelMax.sub(lableMin));
    
    return {
      inpust:inpustNormalize,
      labels: labelsNormalize,
      inputsMax,
      inputsMin,
      labelMax,
      lableMin
    }
  })
}

const trainModel = async (model,inpust,labels) => {
  //definicion de optimizador, funcion de perdida y meticas
  model.compile({
    optimizer: optimizer,
    loss: functionLoss,
    metrics: metrics
  })
  const surface = {name:'show.history live',tab:'Training'};
  const tamanioBatch = 28;
  const epochs = 50;
  const history = [];
  return await model.fit(inpust,labels,{
  tamanioBatch,
  epochs,
  shuffle:true,
  callbacks:tf_vis.show.fitCallbacks(
      {name:'training penformance'},
      ['loss','mse'],
      {
        height:200,
        callbacks:['onEpochEnd']
      }
    
  )
  })
}
const run = async(data) =>{
  visualizer(data)
  const model = createModel();
  const tensorsData = factoryTotensorData(data);
  const {labels,inpust} = tensorsData;
  await trainModel(model,inpust,labels)
  
}

run(dataCleanN2);

