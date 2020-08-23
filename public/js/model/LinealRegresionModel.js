'use strict'
const tf = require('@tensorflow/tfjs');
const tf_vis = require('@tensorflow/tfjs-vis');

export default class LinealRegresionModel{
  /**
   * Metodo constructor de clase
   * @param {Array} data array de array con datos [[x1,y1],[x2,y2]....,[xn,yn]]
   * @param {String} labelX valor de las etiquetas para X 
   * @param {String} labelY valor de las etiquetas para Y
   */
  constructor(data,labelX,labelY){
    this.data = data;
    this.stoptraining = false;
    this.labelX = labelX;
    this.labelY = labelY; 
  }
  /**
   * Metodo para cncelacion de proceso de entrenamiento
   */
  stopTraining(){
    this.stoptraining = true;
  }
  /**
   * Metodo para seteo de superparametros de entrenamiento
   */
  setSuperparams(){
    this.optimizer = tf.train.adam();
    this.functionLoss = tf.losses.meanSquaredError;
    this.metrics = ['mse'];
  } 
  /**
   * metodo para visualizacion de grafica  data inicial
   */
  visualizeInitialData(){
    const data = this.data.map( value => ({x:value[0],y:value[1]}));
    tf_vis.render.scatterplot(
      {name:`${this.labelX} vs ${this.labelY}` },
      {values:data},
      {
        xLabel:`${this.labelX}`,
        yLabel:`${this.labelY}`,
        height:300
      }
      )
  }
  /**
   * metodo para la creacion del modelo
   */
  createModel(){
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
    this.model = model;
  }
  /**
   * metodo para entrenamiento del modelo
   */
  async trainModel(){
     //definicion de optimizador, funcion de perdida y meticas
    this.stoptraining = false;
    this.factoryTotensorData();
    this.setSuperparams();
    this.model.compile({
      optimizer: this.optimizer,
      loss: this.functionLoss,
      metrics: this.metrics
    });
    
    const surface = {name:'show.history live',tab:'Training'};
    const tamanioBatch = 28;
    const epochs = 50;
    const history = [];
    return await this.model.fit(this.inputs,this.labels,{
      tamanioBatch,
      epochs,
      shuffle:true,
      callbacks:{
        onEpochEnd:(epoch,log)=>{
          history.push(log)
          tf_vis.show.history(surface,history,['loss','mse']) 
          if(this.stoptraining){
            this.model.stopTraining = true
          }  
        }
      }
    })
  }
  /**
   * Metodo convercion de datos a tensores
   */
  factoryTotensorData () {
    const {
      inputs,
      labels,
      inputsMax,
      inputsMin,
      labelMax,
      lableMin
    } =  tf.tidy(() => {
      //mesclado de elementos del arry
      tf.util.shuffle(this.data);
      //extaxion de datos  (regularizacion)
      const inputs = this.data.map(value => value[0]);
      const labels = this.data.map(value =>  value[1]);
      
      //creacion de tensores con datos extraidos
      const tensorsInputs = tf.tensor2d(inputs,[inputs.length,1]);
      const tensorLabels = tf.tensor2d(labels,[labels.length,1]);
  
      //valores para desregularizacion
      
      const inputsMax = tensorsInputs.max();
      const inputsMin = tensorsInputs.min();
      const labelMax = tensorLabels.max();
      const lableMin = tensorLabels.min();

      //creacion de entradas normalizadas (dato-min)/(max-min) primitivas de tensor flosw de tensores
      const inputsOut = tensorsInputs.sub(inputsMin).div(inputsMax.sub(inputsMin));
      const labelsOut = tensorLabels.sub(lableMin).div(labelMax.sub(lableMin));
      return {
        inputs:inputsOut,
        labels:labelsOut,
        inputsMax,
        inputsMin,
        labelMax,
        lableMin
      };
    })
    this.inputs = inputs;
    this.labels = labels;
    this.inputsMax = inputsMax;
    this.inputsMin = inputsMin;
    this.labelMax = labelMax;
    this.lableMin = lableMin;
  }
  /**
   * Metodo para calculo y visualizacion de inferencia
   */
  async viewInferenceCurve(){
    const [xs,preds] = tf.tidy(() => {
      const xs =tf.linspace(0,1,100);
      const preds = this.model.predict(xs.reshape([100,1]));
      const desnormX = xs
          .mul(this.inputsMax.sub(this.inputsMin))
          .add(this.inputsMin);
      const desnormY = preds.mul(this.labelMax.sub(this.lableMin))
          .add(this.lableMin)

      return [desnormX.dataSync(),desnormY.dataSync()]
    })
    const pointspredict = Array.from(xs).map((val,i)=>{
      return {x:val,y:preds[i]}
    })
 
    const poitsOrigins = this.data.map( value => ({x:value[0],y:value[1]}));
    tf_vis.render.scatterplot(
      {name:'origins vs prediction'},
      {
        values:[poitsOrigins,pointspredict],
        series:['originales','prediccion']
      },
      {
        xLabel:`${this.labelX}`,
        yLabel:`${this.labelY}`,
        height:300
      }
    )
  }
  /**
   * Metodo para descarga de  modelo
   */
  async save(){
     if(this.model){
      await this.model.save('downloads://modelo-refgrecion')
     }
  }
  /**
   * Metodo para carha de modelo desde dico
   * @param {String} jsonFileId id del file HTML object para archivo json
   * @param {String} binfileId id del file HTML object para archivo Bin
   */
  async loadModelbyFile(jsonFileId,binfileId){
    const $fileJson = document.getElementById(jsonFileId);
    const $fileBin = document.getElementById(binfileId);
    const model = await tf.loadLayersModel(tf.io.browserFiles([
      $fileJson.files[0],
      $fileBin.files[0]
    ])) 
    this.model = model;
    this.factoryTotensorData();
    console.log('====================================');
    console.log("MODEL LOADED");
    console.log('====================================');
  }
}