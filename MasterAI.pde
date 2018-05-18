//Neural Network class and documentation writen by Mark Skinner on may 17, 2018

//a prefabricated Renforcement Learning Model that is strong and flexable

//HOW TO USE THIS PROGRAM
/*
  peramiters for the constructor:
  
    1: int[] : defines the structure of the neural net. <HYPERPERAMETER>
               the number of elements in the array are the ammount of layers (including input and output).
               the numbers reperesent how many neurons in that layer
    2: bool  : wether the fitness should go up, or it should progress downwards. <HYPERPERAMETER>
               use true/false OR INCREASING / DECREASING (they are defined constants) increasing = true;
    3: int   : population size (caps at 1000). How many individuals are created to choose from for the next gen. <HYPERPERAMETER>
    4: int   : mutation rate -> the ammount the Genetic Algorithm mutates its individuals (0-100)%
    
  EXAMPLE:
  myNet = new NeuralNet(new int[]{2,3,1}, DECREASING, 100, 2);

  make sure you use this code propperly by:
  
  1.setting the individual: myNet.setIndividualNumber(i); (where i is and int between 0 and myNet.populationSize)
  2.setting the inputs: myNet.setInput(index, value);
  3.calculating the network: myNet.calculate();
  4.recieving the output: println(myNet.getOutput(index)); (returns a float)
  5.grading the individual with your own method
  6.setting the fitness for the individual: myNet.setFitness(value);
  7.go back to step 1 until all individuals have been graded
  8.train by adding this line of code in: myNet.update(); (this will create new individuals and keep the best 2 from the last gen)
 
 
  -------------ALL FUNCTIONS AND WHAT THEY DO-----------------
  calculateUpdatedNetwork(); //calculates the most sucsessfull intelegence
  changeFitness(float value);
  setFitness(float value);
  converge(float desired, float value); //gives distance beteen target number and current number (returns float)
  getGeneration(); //returns the current generation as an int
  update(); //genetic algorithm processes. ONLY USE WHEN ALL INDIVIDUALS HAVE BEEN TESTED AND GRADED
  setInput(float value);
  getOutput(int index, float value);
  calculate();//calculates neural network for individual set by the setIndividual() function
  setIndividualNumber(int num);
  setMutationRate(float value);
*/
static boolean INCREASING = true;
static boolean DECREASING = false;

class NeuralNet {
  //all gloabal variables for ANN
  int GN = 0;
  int weightCount = 0;
  float[][] weights;
  float[] in, out;
  int[] HLI = new int[]{2,3,1}; //hidden layer input
  
  //all global variables for GA
  int populationSize = 5;
  float mr = 5;
  boolean fitnessMode = true;//means better -> increasing
  int SN = 0;
  float[] fitness = new float[5];
  float bestFitness = 0;
  int[] best2 = new int[]{0,0};
  
  //FM: is the fitness increasing(true) or decreasing(false)  (fitness mode) 
  //(100% debuged and working)
  NeuralNet(int[] items, boolean FM, int popsize, float mutationRate){
    HLI = items;
    fitness = new float[popsize];
    //claculate number of weights
    for(int i = 0; i < items.length-1; i++) weightCount += items[i] * items[i+1] + items[i+1];
    //assign random values to each element of the 2d array
    weights = new float[popsize][weightCount];
    for(int x = 0; x < popsize; x++)for(int y = 0; y < weightCount; y++)weights[x][y] = random(-1,1);
    //setup inputs and outputs
    in = new float[items[0]];
    for(int i = 0; i < items[0]; i++) in[i] = 0;
    out = new float[items[items.length-1]];
    for(int i = 0; i < items[items.length-1]; i++) out[i] = 0;
    //set some global constants
    fitnessMode = FM;
    populationSize = constrain(popsize, 5, 1000);//keep it reasonable
    mr = constrain(mutationRate, 0, 100);
  }
  
  //computes the outputs but does not return them (the fun part)
  //a is for agent number (0 -- pop size-1);
  //(100% debuged but calculations untested)
  void calculate(){
    float v = 0;
    int wi = 0;
    if(HLI.length == 2){//if the network has no hidden layers
      for(int t = 0; t < out.length; t++){
        v=0;
        for(int i = 0; i < in.length; i++){
          v += in[i] * weights[SN][wi]; wi++;
        }
        out[t] = AF(v+weights[SN][wi]); wi++; //add bias :)
      }
    }
    else{
      //input -> HL
      float[] HL;
      HL = new float[HLI[1]];
      for(int t = 0; t < HLI[1]; t++){
        v = 0;
        for(int i = 0; i < in.length; i++){
          v += in[i] * weights[SN][wi]; wi++;
        }
        v += weights[SN][wi]; wi++;
        HL[t] = AF(v);
      }
      
      //code block for 2 consecutive hidden layers
      //float[]HL = new float[HLI[1]];
      for(int e = 2; e < HLI.length-1; e++){
        float[] HL2 = new float[HLI[e]];
        for(int t = 0; t < HLI[e]; t++){
          v = 0;
          for(int i = 0; i < HL.length; i++){
            v += HL[i] * weights[SN][wi]; wi++;
          }
          v += weights[SN][wi]; wi++;
          HL2[t] = AF(v);
        }
        HL = new float[HL2.length];
        HL = HL2;
      }
      
      //HL -> output
      for(int t = 0; t < out.length; t++){
        v = 0;
        for(int i = 0; i < HL.length; i++){
          v += HL[i] * weights[SN][wi]; wi++;
        }
        v += weights[SN][wi]; wi++;
        out[t] = AF(v);
      }
    }
  }
  
  //returns a computed output based on the index
  //(100% debuged and working)
  float getOutput(int index){
    return(out[index]);
  }
  
  //index starts at 0
  //(100% debuged and working)
  void setInput(int index, float value){
    in[index] = value;
  }
  
  //(100% debuged and working)
  void setIndividualNumber(int num){
    SN = num;
  }
  
  //the massive training function, increases generation number by 1
  void update(){
    GN++;
    float[][] temp = new float[populationSize][weightCount];
    //find the best 2 of generation GN
    best2 = findBest2(fitnessMode);
    //create new population based on the best 2 species
    for(int i = 0; i < weightCount; i++){
      temp[0][i] = weights[best2[0]][i];
      temp[1][i] = weights[best2[1]][i];
      temp[2][i] = (weights[best2[0]][i] + weights[best2[1]][i])/random(1.9,2.1);
      
      for(int x = 3; x < populationSize; x++){
        if(mr <= random(100)){
          if(random(0,100) < 50){
            temp[x][i] = weights[best2[0]][i] + random(-0.01, 0.01);
          }
          else{
            temp[x][i] = weights[best2[1]][i] + random(-0.01, 0.01);
          }
        }
        else{
          temp[x][i] = random(-1, 1);
        }
      }
    }
    weights = temp;
  }
  
  int getGeneration(){
    return GN;
  }
  
  int[] findBest2(boolean type){
    best2 = new int[]{0,0};
    if(type){
      float mocValue = -1000000000;
      for(int i = 0; i < fitness.length; i++){
        if(fitness[i] >= mocValue){
          mocValue = fitness[i];
          best2[0] = i;
        }
      }
      mocValue = -1000000000;
      for(int i = 0; i < fitness.length; i++){
        if(fitness[i] >= mocValue && best2[0] != i){
          mocValue = fitness[i];
          best2[1] = i;
        }
      }
    }
    else{
      float mocValue = 1000000000;
      for(int i = 0; i < fitness.length; i++){
        if(fitness[i] <= mocValue){
          mocValue = fitness[i];
          best2[0] = i;
        }
      }
      mocValue = 1000000000;
      for(int i = 0; i < fitness.length; i++){
        if(fitness[i] <= mocValue && best2[0] != i){
          mocValue = fitness[i];
          best2[1] = i;
        }
      }
    }
    bestFitness = fitness[best2[0]];
    return best2;
  }
  
  float converge(float desired, float value){
    return abs(desired-value);
  }
  
  //sets the fitness to a exact value 
  //(for use at the end of a testing session with your own fitness gradient)
  void setFitness(float value){
    fitness[SN] = value;
  }
  void changeFitness(float value){
    fitness[SN] += value;
  }
  void setMutationRate(float value){
    mr = value;
  }
  //activation
  float AF(float iv){
    return ((float)Math.tanh(iv));
  }
  void calculateUpdatedNetwork(){
    setIndividualNumber(best2[0]);
    calculate();
  }
}