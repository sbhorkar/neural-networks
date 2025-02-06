import java.io.*;

/*
 * @author Sanaa Bhorkar
 * @version 02/21/2024
 * 
 * This A-B-C perceptron and is built based off of the design document "2-Minimizing the Error Function" and the 
 * previous A-B-1 network. The A-B-C network can train a network based off of randomly assigned weights or simply run
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
 *    - trainOrRun(): goes through the training process or simply calls the runNetwork() method 
 *    - runNetwork(): calculates the weights based off of the value of the input/hidden neurons and the corresponding weights
 *    - results(): prints the truth table and output neurons for each test case
 *    - activate(): plugs in a value into the sigmoid activation function
 *    - derivActivate(): plugs in a value into the derivative of the sigmoid activation function
 *    - calculateDeltaWeights(): calls the calculate weights methods for the inputToHidden weights and the hiddenToOutput weights
 *    - calculateWeightsInputToHidden(): uses gradient descent to calculate the change in weights that will minimize the error
 *    - calculateWeightsHiddenToOutput(): uses gradient descent to calculate the change in weights that will minimize the error
 *    - applyDeltaWeights(): applies the calculated change in weights on the current weight arrays
 *    - saveWeights(): saves the current weights in an external file
 *    - loadWeights(): loads the weights from an external file
 */
public class ABCNetwork 
{
   public int nInputs;
   public int nHidden;       
   public int nOutputs;           
   public int testCases;
   public int maxIterations;
   public int iterations;
   public double lambda;
   public double errorCutoff;
   public double randMin;
   public double randMax;
   public double averageError;
   public boolean training;
   public boolean randomize;
   public boolean loadWeightsFromFile;
   public String reason;
   public String weightsFileName;
   public double[] a;
   public double[] h;
   public double[] F;
   public double[] upperThetaHidden;
   public double[] upperThetaOutput;
   public double[] upperOmega;
   public double[] lowerOmega;
   public double[] upperPsi;
   public double[] lowerPsi;
   public double[][] inputToHiddenWeights;
   public double[][] hiddenToOutputWeights;
   public double[][] inputToHiddenDeltaWeights;
   public double[][] hiddenToOutputDeltaWeights;
   public double[][] truthTableInputs;
   public double[][] truthTableExpectedOutputs;
   public double[][] truthTableActualOutputs;

/*
 * The main() method creates an object of the ABCNetwork class and runs its 6 major methods.
 */
   public static void main(String[] args)
   {
      System.out.println(); // Printing clear line before all output to make it easier to read in the terminal.

      ABCNetwork network = new ABCNetwork();

      network.configureNetwork();
      network.echoParams();
      network.allocateMemory();
      network.populate();
      network.trainOrRun();
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
      loadWeightsFromFile = false; 
      lambda = 0.3;
      nInputs = 2;
      nHidden = 1;
      nOutputs = 3;
      maxIterations = 100000;
      errorCutoff = 2E-4;
      randMin = 0.1;
      randMax = 1.5;
      testCases = 4;
      weightsFileName = "weights.txt"; // if weightsFileName is empty, the network assumes that it shouldn't save the weights
   } // public void configureNetwork() 

/*
 * The setTruthTable() allows the user to edit the truth table that the network is going to use to train.
 */
   public void setTruthTable()
   {
      truthTableInputs[0][0] = 0.0;
      truthTableInputs[0][1] = 0.0;
      truthTableExpectedOutputs[0][0] = 0.0;

      truthTableInputs[1][0] = 0.0;
      truthTableInputs[1][1] = 1.0;
      truthTableExpectedOutputs[1][0] = 0.0;

      truthTableInputs[2][0] = 1.0;
      truthTableInputs[2][1] = 0.0;
      truthTableExpectedOutputs[2][0] = 0.0;

      truthTableInputs[3][0] = 1.0;
      truthTableInputs[3][1] = 1.0;
      truthTableExpectedOutputs[3][0] = 1.0;

/*
 * The block of code below can be uncommented for trying a A-B-3 network.
 */
      truthTableExpectedOutputs[0][1] = 0.0;
      truthTableExpectedOutputs[0][2] = 0.0;

      truthTableExpectedOutputs[1][1] = 1.0;
      truthTableExpectedOutputs[1][2] = 1.0;

      truthTableExpectedOutputs[2][1] = 1.0;
      truthTableExpectedOutputs[2][2] = 1.0;

      truthTableExpectedOutputs[3][1] = 1.0;
      truthTableExpectedOutputs[3][2] = 0.0;

   } // public void setTruthTable()

/*
 * The echoParams() method prints the network configuration and all other relevant parameters.
 */
   public void echoParams() 
   {
      System.out.println("Network configuration: " + nInputs + "-" + nHidden + "-" + nOutputs);
      System.out.println("Training?: " + training);
      System.out.println("Randomize weights?: " + randomize);
      System.out.println("Load weights from file?: " + loadWeightsFromFile);

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
      a = new double[nInputs];
      h = new double[nHidden];
      F = new double[nOutputs];
      inputToHiddenWeights = new double[nInputs][nHidden];
      hiddenToOutputWeights = new double[nHidden][nOutputs];
      truthTableInputs = new double[testCases][nInputs];
      truthTableExpectedOutputs = new double[testCases][nOutputs];
      truthTableActualOutputs = new double[testCases][nOutputs];

      if (training)
      {
         inputToHiddenDeltaWeights = new double[nInputs][nHidden];
         hiddenToOutputDeltaWeights = new double[nHidden][nOutputs];
         upperThetaHidden = new double[nHidden];
         upperThetaOutput = new double[nOutputs];
         upperOmega = new double[nHidden];
         lowerOmega = new double[nOutputs];
         upperPsi = new double[nHidden];
         lowerPsi = new double[nOutputs];
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

      if (loadWeightsFromFile)
      {
         loadWeights();
      }
      else if (!randomize && !loadWeightsFromFile)
      {
         inputToHiddenWeights[0][0] = 0.75;
         inputToHiddenWeights[1][0] = 0.6;

         hiddenToOutputWeights[0][0] = 0.5;
         hiddenToOutputWeights[0][1] = 0.75;
         hiddenToOutputWeights[0][2] = 0.35;
      } // else if (!randomize && !loadWeightsFromFile)
      else
      {
         int k, j, i;

         for (k = 0; k < nInputs; k++)
         {
            for (j = 0; j < nHidden; j++)
            {
               inputToHiddenWeights[k][j] = randomize(randMin, randMax);
            }
         } // for (k = 0; k < nInputs; k++)

         for (j = 0; j < nHidden; j++)
         {
            for (i = 0; i < nOutputs; i++)
            {
               hiddenToOutputWeights[j][i] = randomize(randMin, randMax);
            }
         } // for (j = 0; j < nHidden; j++)
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
 * The train() method goes through the whole training process for the network if the network is training. It goes through each test
 * case, sets the input neurons, calculates the delta weights, applies the delta weights, runs the network, and calculates the 
 * average error. When the average error over all test cases is less than the error threshold or the amount of iterations exceeds 
 * the maximum, the network stops the training process. If the network is simply running, it goes through the test cases and runs.
 */
   public void trainOrRun() 
   {
      double totalError = 0.0;
      
      averageError = 0.0;

      if (training)
      {
         boolean done = false;
       
         iterations = 0;
         reason = "";

         while (!done)
         {
            int truthTableIndex, k, I;

            totalError = 0.0;
            averageError = 0.0;

            for (truthTableIndex = 0; truthTableIndex < testCases; truthTableIndex++)
            {
               for (k = 0; k < nInputs; k++)
               {
                  a[k] = truthTableInputs[truthTableIndex][k];
               }

               runNetwork();
               calculateDeltaWeights(truthTableIndex);
               applyDeltaWeights();
               runNetwork();

               for (I = 0; I < nOutputs; I++)
               {
                  lowerOmega[I] = truthTableExpectedOutputs[truthTableIndex][I] - F[I];
                  totalError += 0.5 * lowerOmega[I] * lowerOmega[I];
                  truthTableActualOutputs[truthTableIndex][I] = F[I];
               } // for (I = 0; I < nOutputs; I++)
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

         if (!weightsFileName.equals(""))
         {
            saveWeights();
         }
      } // if (training)
      else
      {
         int truthTableIndex, k, i;

         for (truthTableIndex = 0; truthTableIndex < testCases; truthTableIndex++)
         {
            for (k = 0; k < nInputs; k++)
            {
               a[k] = truthTableInputs[truthTableIndex][k];
            }

            runNetwork();

            for (i = 0; i < nOutputs; i++)
            {
               truthTableActualOutputs[truthTableIndex][i] = F[i];
            }
         } // for (truthTableIndex = 0; truthTableIndex < testCases; truthTableIndex++)
      } // else
   } // public void train() 

/*
 * The runNetwork() calculates the hidden and output neurons based on the current values in the neurons and the weights. 
 */
   public void runNetwork()
   {
      int j, i, K, J;

      for (j = 0; j < nHidden; j++)
      {
         double thetaHidden = 0.0;

         for (K = 0; K < nInputs; K++)
         {
            thetaHidden += a[K] * inputToHiddenWeights[K][j];
         }

         h[j] = activate(thetaHidden);
      } // for (j = 0; j < nHidden; j++)

      for (i = 0; i < nOutputs; i++)
      {
         double thetaOutput = 0.0;

         for (J = 0; J < nHidden; J++) 
         {
            thetaOutput += h[J] * hiddenToOutputWeights[J][i];
         }

         F[i] = activate(thetaOutput);
      } // for (i = 0; i < nOutputs; i++)
   } // public void runNetwork()

/*
 * The results() method prints out the truth table and the output values of the network.
 */
   public void results()
   {
      int rowTruthTable, k, i;

      if (training)
      {
         System.out.println("Training Exit Information");
         System.out.println("------------------------------------");
         System.out.println("Reason for end of training: " + reason);
         System.out.println("Iterations reached: " + iterations);
         System.out.println("Error reached: " + averageError);
         System.out.println();
      } // if (training)

      System.out.println("Truth Table");
      System.out.println("------------------------------------");

      for (rowTruthTable = 0; rowTruthTable < testCases; rowTruthTable++)
      {
         String truthTableOutput = "";

         for (k = 0; k < nInputs; k++)
         {
            truthTableOutput += truthTableInputs[rowTruthTable][k] + "\t";
         }

         truthTableOutput += "|\t";

         for (i = 0; i < nOutputs; i++)
         {
            truthTableOutput += truthTableExpectedOutputs[rowTruthTable][i] + "\t";
         }

         truthTableOutput += "|\t";

         for (i = 0; i < nOutputs; i++)
         {
            truthTableOutput += truthTableActualOutputs[rowTruthTable][i] + "\t";
         }

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
      int k, j, i;         

      for (k = 0; k < nInputs; k++)
      {
         for (j = 0; j < nHidden; j++)
         {
            for (i = 0; i < nOutputs; i++)
            {
               calculateWeightsInputToHidden(k, j, testCase);
               calculateWeightsHiddenToOutput(j, i, testCase);
            }
         } // for (j = 0; j < nHidden; j++)
      } // for (k = 0; k < nInputs; k++)
   } // public void calculateDeltaWeights(int testCase)

/*
 * The calculateWeightsInputToHidden() method calculates the change in weights for the inputToHidden weights. It calculates 
 * this based off of the gradient descent method described in further detail in the design document. 
 */
   public void calculateWeightsInputToHidden(int k, int j, int testCase)
   {
      int K, I;
      double partialEOverPartialWeightsInputToHidden;

      upperThetaHidden[j] = 0.0;

      for (K = 0; K < nInputs; K++)   
      {
         upperThetaHidden[j] += a[K] * inputToHiddenWeights[K][j];
      }

      upperOmega[j] = 0.0;
      
      for (I = 0; I < nOutputs; I++)
      {
         upperOmega[j] += lowerPsi[I] * hiddenToOutputWeights[j][I]; 
      }

      upperPsi[j] = upperOmega[j] * derivActivate(upperThetaHidden[j]);
      partialEOverPartialWeightsInputToHidden = -a[k] * upperPsi[j];
      inputToHiddenDeltaWeights[k][j] = -lambda * partialEOverPartialWeightsInputToHidden;
   } // public void calculateWeightsInputToHidden(int k, int j, int testCase)

/*
 * The calculateWeightsHiddenToOutput() method calculates the change in weights for the hiddenToOutput weights. It calculates 
 * this based off of the gradient descent method described in further detail in the design document. 
 */
   public void calculateWeightsHiddenToOutput(int j, int i, int testCase) 
   {
      int J;
      double partialEOverPartialWeightsHiddenToOutput;

      upperThetaOutput[i] = 0.0;

      for (J = 0; J < nHidden; J++)
      {
         upperThetaOutput[i] += h[J] * hiddenToOutputWeights[J][i];
      }

      F[i] = activate(upperThetaOutput[i]);
      lowerOmega[i] = truthTableExpectedOutputs[testCase][i] - F[i];
      lowerPsi[i] = lowerOmega[i] * derivActivate(upperThetaOutput[i]);
      partialEOverPartialWeightsHiddenToOutput = -h[j] * lowerPsi[i];
      hiddenToOutputDeltaWeights[j][i] = -lambda * partialEOverPartialWeightsHiddenToOutput;
   } // public void calculateWeightsHiddenToOutput(int j, int i, int testCase)

/*
 * The applyDeltaWeights() method applies the changes calculated in the calculateWeights methods to the inputToHidden and
 * hiddenToOutput weight arrays.
 */
   public void applyDeltaWeights()
   {
      int k, j, i;

      for (k = 0; k < nInputs; k++)
      {
         for (j = 0; j < nHidden; j++)
         {
            inputToHiddenWeights[k][j] += inputToHiddenDeltaWeights[k][j];
         }
      } // for (k = 0; k < nInputs; k++)

      for (j = 0; j < nHidden; j++)
      {
         for (i = 0; i < nOutputs; i++)
         {
            hiddenToOutputWeights[j][i] += hiddenToOutputDeltaWeights[j][i];
         }
      } // for (j = 0; j < nHidden; j++)
   } // public void applyDeltaWeights()

/*
 * The saveWeights() method saves the current weights in an external file.
 */
   public void saveWeights()
   {
      try 
      {
         int k, j, i;

         PrintWriter pw = new PrintWriter(new FileOutputStream(new File(weightsFileName), false));

         for (k = 0; k < nInputs; k++)
         {
            for (j = 0; j < nHidden; j++)
            {
               pw.println(inputToHiddenWeights[k][j]);
            }
         } // for (k = 0; k < nInputs; k++)

         for (j = 0; j < nHidden; j++)
         {
            for (i = 0; i < nOutputs; i++)
            {
               pw.println(hiddenToOutputWeights[j][i]);
            }
         } // for (j = 0; j < nHidden; j++)

         pw.close();
      } // try
      catch (Exception e) 
      {
         System.out.println("Exception" + e.toString());
      }
   } // public void saveWeights()

/*
 * The loadWeights() method loads in weights from an external file.
 */
   public void loadWeights()
   {
      int k, j, i;

      try
      {
         BufferedReader reader = new BufferedReader(new FileReader(weightsFileName));
         String line = reader.readLine();
         for (k = 0; k < nInputs; k++)
         {
            for (j = 0; j < nHidden; j++)
            {
               double weight = Double.parseDouble(line);
               inputToHiddenWeights[k][j] = weight; 
               line = reader.readLine();
            } // for (j = 0; j < nHidden; j++)
         } // for (k = 0; k < nInputs; k++)

         for (j = 0; j < nHidden; j++)
         {
            for (i = 0; i < nOutputs; i++)
            {
               double weight = Double.parseDouble(line);
               hiddenToOutputWeights[j][i] = weight; 
               line = reader.readLine();
            } // for (i = 0; i < nOutputs; i++)
         } // for (j = 0; j < nHidden; j++)
      } // try
      catch (IOException e)
      {
         System.out.println("Exception " + e.toString());
      }
   } // public void loadWeights()
} // public class ABCNetwork 