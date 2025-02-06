/*
 * @author Sanaa Bhorkar
 * @version 01/30/2024
 * 
 * This A-B-1 perceptron and is built based off of the design document "1-Minimization the Single Output Error Function" and the 
 * previous 2-2-1 spreadsheet project. The A-B-1 network can train a network based off of randomly assigned weights or simply run
 * when given weights from a user. The network trains using simple gradient descent. 
 * 
 * List of methods:
 *    - main(): method to run the network
 *    - setTruthTable(): sets the truth table that the network will use to train
 *    - configureNetwork(): sets the parameters for the network
 *    - echoParams(): printing all the parameters
 *    - allocateMemory(): instantiating all the arrays
 *    - populate(): randomizing or fixing the weight values
 *    - randomize(): returns a random number in between a given range
 *    - train(): calculates the change in weights, applies them, runs the network, and calculates the error
 *    - runNetwork(): calculates the weights based off of the value of the input/hidden neurons and the corresponding weights
 *    - results(): prints the truth table and output neurons for each test case
 *    - activate(): plugs in a value into the sigmoid activation function
 *    - derivActivate(): plugs in a value into the derivative of the sigmoid activation function
 *    - calculateDeltaWeights(): calls the calculate weights methods for the inputToHidden weights and the hiddenToOutput weights
 *    - calculateWeightsInputToHidden(): uses gradient descent to calculate the change in weights that will minimize the error
 *    - calculateWeightsHiddenToOutput(): uses gradient descent to calculate the change in weights that will minimize the error
 *    - applyDeltaWeights(): applies the calculated change in weights on the current weight arrays
 */
public class AB1Network 
{
   public static final int N_OUTPUTS = 1;    // This value is fixed at 1 because this is an A-B-1 network.

   public int nInputs;
   public int nHidden;                  
   public int testCases;
   public int maxIterations;
   public double lambda;
   public double errorCutoff;
   public double randMin;
   public double randMax;
   public boolean training;
   public boolean randomize;
   public double[] inputNeurons;
   public double[] hiddenNeurons;
   public double[] outputNeurons;
   public double[] upperThetaHidden;
   public double[] upperOmega;
   public double[] upperPsi;
   public double[][] inputToHiddenWeights;
   public double[][] hiddenToOutputWeights;
   public double[][] inputToHiddenDeltaWeights;
   public double[][] hiddenToOutputDeltaWeights;
   public double[][] truthTableInputs;
   public double[][] truthTableExpectedOutputs;
   public double[][] truthTableActualOutputs;

/*
 * The main() method creates an object of the AB1Network class and runs its 6 major methods.
 */
   public static void main(String[] args)
   {
      System.out.println(); // Printing clear line before all output to make it easier to read in the terminal.

      AB1Network network = new AB1Network();

      network.configureNetwork();
      network.echoParams();
      network.allocateMemory();
      network.populate();
      network.train();
      network.results();
   } // public static void main(String[] args)

/*
 * The configureNetwork() method is where the user can edit certain parameters of the network such as the configuration, 
 * maxIterations cutoff, error threshold, whether it is training or running, the random number range, and the number of
 * test cases.
 */
   public void configureNetwork() 
   {
      training = true;  // false = running; true = training
      randomize = true; // false = set the weights manually; true = randomize the weights
      lambda = 0.3;
      nInputs = 2;
      nHidden = 5;
      maxIterations = 100000;
      errorCutoff = 2E-4;
      randMin = -1.5;
      randMax = 1.5;
      testCases = 4;
   } // public void configureNetwork() 

/*
 * The setTruthTable() allows the user to edit the truth table that the network is going to use to train.
 */
   public void setTruthTable()
   {
      truthTableInputs = new double[testCases][nInputs];
      truthTableExpectedOutputs = new double[testCases][N_OUTPUTS];
      truthTableActualOutputs = new double[testCases][N_OUTPUTS];

      truthTableInputs[0][0] = 0.0;
      truthTableInputs[0][1] = 0.0;
      truthTableExpectedOutputs[0][0] = 0.0;

      truthTableInputs[1][0] = 0.0;
      truthTableInputs[1][1] = 1.0;
      truthTableExpectedOutputs[1][0] = 1.0;

      truthTableInputs[2][0] = 1.0;
      truthTableInputs[2][1] = 0.0;
      truthTableExpectedOutputs[2][0] = 1.0;

      truthTableInputs[3][0] = 1.0;
      truthTableInputs[3][1] = 1.0;
      truthTableExpectedOutputs[3][0] = 0.0;
   } // public void setTruthTable()

/*
 * The echoParams() method prints the network configuration and all other relevant parameters.
 */
   public void echoParams() 
   {
      System.out.println("Network configuration: " + nInputs + "-" + nHidden + "-" + N_OUTPUTS);
      System.out.println("Training?: " + training);
      System.out.println("Randomize weights?: " + randomize);

      if (training)
      {
         System.out.println("Random number range: " + randMin + " to " + randMax);
         System.out.println("Maximum iterations: " + maxIterations);
         System.out.println("Error threshold: " + errorCutoff);
         System.out.println("Learning factor: " + lambda);
         System.out.println();
      } // if (training)
   } // public void echoParams() 

/*
 * The allocateMemory() method instantiates all the arrays that the network is going to use. Examples of the arrays are the
 * input, hidden, and output activations and all the weights.
 */
   public void allocateMemory() 
   {
      inputNeurons = new double[nInputs];
      hiddenNeurons = new double[nHidden];
      outputNeurons = new double[N_OUTPUTS];
      inputToHiddenWeights = new double[nInputs][nHidden];
      hiddenToOutputWeights = new double[nHidden][N_OUTPUTS];

      if (training)
      {
         inputToHiddenDeltaWeights = new double[nInputs][nHidden];
         hiddenToOutputDeltaWeights = new double[nHidden][N_OUTPUTS];
         upperThetaHidden = new double[nHidden];
         upperOmega = new double[nHidden];
         upperPsi = new double[nHidden];
      } // if (training)
   } // public void allocateMemory() 

/*
 * The populate() method populates the weight arrays with either fixed weight values or randomized values, depending on whether 
 * or not the network is training. We assume that "training" the network will use randomized weights while simply running the 
 * network will used fixed weights.
 */
   public void populate() 
   {
      setTruthTable();

      if (!randomize)
      {
         inputToHiddenWeights[0][0] = 0.75;
         inputToHiddenWeights[1][0] = 0.6;
         inputToHiddenWeights[0][1] = 0.25;
         inputToHiddenWeights[1][1] = 0.9;

         hiddenToOutputWeights[0][0] = 0.5;
         hiddenToOutputWeights[1][0] = 0.05;
      } // if (!randomize)
      else
      {
         int k, j;

         for (k = 0; k < nInputs; k++)
         {
            for (j = 0; j < nHidden; j++)
            {
               inputToHiddenWeights[k][j] = randomize(randMin, randMax);
            }
         } // for (k = 0; k < nInputs; k++)

         for (j = 0; j < nHidden; j++)
         {
            hiddenToOutputWeights[j][0] = randomize(randMin, randMax);
         }
      } // else
   } // public void populate() 

/*
 * The randomize() method takes in the low and high numbers in the random range and calculates a random number. This method is 
 * used when calculating random weight values.
 */
   public double randomize(double low, double high)
   {
      return low + (high - low) * Math.random();
   }

/*
 * The train() method goes through the whole training process for the network. It goes through each test case, sets the input 
 * neurons, calculates the delta weights, applies the delta weights, runs the network, and calculates the average error. When  
 * the average error over all test cases is less than the error threshold or the amount of iterations exceeds the maximum, the
 * network stops the training process. If the network is simply running, it goes through the test cases and runs.
 */
   public void train() 
   {
      double totalError = 0.0;
      double averageError = 0.0;

      if (training)
      {
         int iterations = 0;
         String reason = "";
         boolean done = false;

         while (!done)
         {
            int truthTableIndex, k;

            totalError = 0.0;
            averageError = 0.0;

            for (truthTableIndex = 0; truthTableIndex < testCases; truthTableIndex++)
            {
               for (k = 0; k < nInputs; k++)
               {
                  inputNeurons[k] = truthTableInputs[truthTableIndex][k];
               }

               runNetwork();
               calculateDeltaWeights(truthTableIndex);
               applyDeltaWeights();
               runNetwork();

               double omega = truthTableExpectedOutputs[truthTableIndex][0] - outputNeurons[0]; // 0 because only 1 output neuron

               totalError += 0.5 * omega * omega;
               truthTableActualOutputs[truthTableIndex][0] = outputNeurons[0];
            } // for (truthTableIndex = 0; truthTableIndex < testCases; truthTableIndex++)

            averageError = totalError / ((double) testCases);

            if (averageError < errorCutoff)
            {
               done = true;
               reason = "Average error under the error cutoff (" + errorCutoff + ").";
            }

            iterations++;

            if (iterations > maxIterations)
            {
               done = true;
               reason = "Over the max iterations cutoff (" + maxIterations + ").";
            }
         } // while (!done)

         System.out.println("Training Exit Information");
         System.out.println("------------------------------------");
         System.out.println("Reason for end of training: " + reason);
         System.out.println("Iterations reached: " + iterations);
         System.out.println("Error reached: " + averageError);
         System.out.println();
      } // if (!training)
      else
      {
         int truthTableIndex, k;

         totalError = 0.0;
         averageError = 0.0;

         for (truthTableIndex = 0; truthTableIndex < testCases; truthTableIndex++)
         {
            for (k = 0; k < nInputs; k++)
            {
               inputNeurons[k] = truthTableInputs[truthTableIndex][k];
            }

            runNetwork();

            truthTableActualOutputs[truthTableIndex][0] = outputNeurons[0];
         } // for (truthTableIndex = 0; truthTableIndex < testCases; truthTableIndex++)
      } // else
   } // public void train() 

/*
 * The runNetwork() calculates the hidden and output neurons based on the current values in the neurons and the weights. 
 */
   public void runNetwork()
   {
      int j, k;
      double thetaHidden, thetaOutput;

      for (j = 0; j < nHidden; j++)
      {
         thetaHidden = 0.0;

         for (k = 0; k < nInputs; k++)
         {
            thetaHidden += inputNeurons[k] * inputToHiddenWeights[k][j];
         }

         hiddenNeurons[j] = activate(thetaHidden);
      } // for (j = 0; j < nHidden; j++)

      thetaOutput = 0.0;

      for (j = 0; j < nHidden; j++) 
      {
         thetaOutput += hiddenNeurons[j] * hiddenToOutputWeights[j][0];
      }

      outputNeurons[0] = activate(thetaOutput);
   } // public void runNetwork()

/*
 * The results() method prints out the truth table and the output values of the network.
 */
   public void results()
   {
      int rowTruthTable, k;

      System.out.println("Truth Table");
      System.out.println("------------------------------------");
      System.out.println("a10\ta11\tT\tF");

      for (rowTruthTable = 0; rowTruthTable < testCases; rowTruthTable++)
      {
         String truthTableOutput = "";

         for (k = 0; k < nInputs; k++)
         {
            truthTableOutput += truthTableInputs[rowTruthTable][k] + "\t";
         }

         truthTableOutput += truthTableExpectedOutputs[rowTruthTable][0] + "\t";
         truthTableOutput += truthTableActualOutputs[rowTruthTable][0];

         System.out.println(truthTableOutput);
      } // for (rowTruthTable = 0; rowTruthTable < testCases; rowTruthTable++)
   } // public void results()

/*
 * The activate() method plugs in a given x value into the sigmoid activation function and returns the result.
 */
   public double activate(double x)
   {
      return 1.0/(1.0 + Math.exp(-x));
   }

/*
 * The derivActivate() method plugs in a given x into the derivative of the sigmoid activation function and returns the result.
 */
   public double derivActivate(double x)
   {
      double activationFunction = activate(x);

      return activationFunction * (1.0 - activationFunction);
   } // public double derivActivate(double x)

/*
 * The calculateDeltaWeights() method calculates the best weights for the inputToHidden weights and the hiddenToOutput weights.
 * It takes in the test case that the program is training on. 
 */
   public void calculateDeltaWeights(int testCase)
   {
      int k, j;               

      for (k = 0; k < nInputs; k++)
      {
         for (j = 0; j < nHidden; j++)
         {
            calculateWeightsInputToHidden(k, j, testCase);
            calculateWeightsHiddenToOutput(j, testCase);
         }
      } // for (k = 0; k < nInputs; k++)
   } // public void calculateDeltaWeights(int testCase)

/*
 * The calculateWeightsInputToHidden() method calculates the change in weights for the inputToHidden weights. It calculates 
 * this based off of the gradient descent method described in further detail in the design document. 
 */
   public void calculateWeightsInputToHidden(int k, int j, int testCase)
   {
      int K, J;
      double lowerOmega = truthTableExpectedOutputs[testCase][0] - outputNeurons[0];
      double upperTheta0, lowerPsi, partialEOverPartialWeightsInputToHidden;

      upperThetaHidden[j] = 0.0;

      for (K = 0; K < nInputs; K++)   
      {
         upperThetaHidden[j] += inputNeurons[K] * inputToHiddenWeights[K][j];
      }

      hiddenNeurons[j] = activate(upperThetaHidden[j]);
      upperTheta0 = 0.0;

      for (J = 0; J < nHidden; J++) 
      {
         upperTheta0 += hiddenNeurons[J] * hiddenToOutputWeights[J][0];
      }

      lowerPsi = lowerOmega * derivActivate(upperTheta0);
      upperOmega[j] = lowerPsi * hiddenToOutputWeights[j][0];
      upperPsi[j] = upperOmega[j] * derivActivate(upperThetaHidden[j]);
      partialEOverPartialWeightsInputToHidden = -inputNeurons[k] * upperPsi[j];
      inputToHiddenDeltaWeights[k][j] = -lambda * partialEOverPartialWeightsInputToHidden;
   } // public void calculateWeightsInputToHidden(int k, int j, int testCase)

/*
 * The calculateWeightsHiddenToOutput() method calculates the change in weights for the hiddenToOutput weights. It calculates 
 * this based off of the gradient descent method described in further detail in the design document. 
 */
   public void calculateWeightsHiddenToOutput(int j, int testCase) 
   {
      double upperTheta0, lowerOmega, lowerPsi, partialEOverPartialWeightsHiddenToOutput;
      int J;

      upperTheta0 = 0.0;

      for (J = 0; J < nHidden; J++)
      {
         upperTheta0 += hiddenNeurons[J] * hiddenToOutputWeights[J][0];
      }

      outputNeurons[0] = activate(upperTheta0);
      lowerOmega = truthTableExpectedOutputs[testCase][0] - outputNeurons[0];
      lowerPsi = lowerOmega * derivActivate(upperTheta0);
      partialEOverPartialWeightsHiddenToOutput = -hiddenNeurons[j] * lowerPsi;
      hiddenToOutputDeltaWeights[j][0] = -lambda * partialEOverPartialWeightsHiddenToOutput;
   } // public void calculateWeightsHiddenToOutput(int j, int testCase) 

/*
 * The applyDeltaWeights() method applies the changes calculated in the calculateWeights methods to the inputToHidden and
 * hiddenToOutput weight arrays.
 */
   public void applyDeltaWeights()
   {
      int k, j;

      for (k = 0; k < nInputs; k++)
      {
         for (j = 0; j < nHidden; j++)
         {
            inputToHiddenWeights[k][j] += inputToHiddenDeltaWeights[k][j];
         }
      }

      for (j = 0; j < nHidden; j++)
      {
         hiddenToOutputWeights[j][0] += hiddenToOutputDeltaWeights[j][0];
      }
   } // public void applyDeltaWeights()
} // public class AB1Network 