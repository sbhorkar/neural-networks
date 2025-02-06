import java.io.*;
import java.util.*;

/*
 * @author Sanaa Bhorkar
 * @version 03/08/2024
 * 
 * This A-B-C perceptron and is built based off of the design document "3-Minimizing and Optimizing the Error Function." 
 * The A-B-C network can train a network based off of randomly assigned weights or simply run when given weights from a user. The
 * network trains using gradient descent and back propagation. The program uses a DataOutputStream/DataInputStream to save the
 * weights in a binary format or load them from the binary format. It also uses a BufferedReader and StringTokenizer to read the
 * configuration file.
 * 
 * Format of Configuration File:
 * T                  // R for running, T for training
 * R                  // R for randomized initial weights, P for preloaded initial weights, M for manual inputs in allocateMemory()
 * 2 5 3              // number of units in each layer 
 * 4                  // number of test cases
 * 1 1 1 1 0          // truth table
 * 1 0 0 1 1
 * 0 1 0 1 1
 * 0 0 0 0 0
 * 100000             // maximum number of iterations 
 * 2E-4               // minimum error
 * 0.3                // learning rate (lambda)  
 * 0.1 1.5            // random number range
 * weights.txt        // the output file that weights will be saved to (can be omitted to not save weights)
 * initialWeights.txt // the file that the weights will be read from (will be omitted when not loading weights)
 */
public class ABCBackprop 
{
   public static String saveWeightsFile;
   public static int nInputs;
   public static int nHidden;       
   public static int nOutputs;           
   public static int testCases;
   public static int maxIterations;
   public static int iterations;
   public static double lambda;
   public static double errorCutoff;
   public static double randMin;
   public static double randMax;
   public static double totalError;
   public static double averageError;
   public static double startTime;
   public static double endTime;
   public static boolean training;
   public static boolean randomize;
   public static boolean configNameGiven;
   public static boolean loadWeightsFromFile;
   public static boolean saveWeightsToFile;
   public static String initialWeightsFile;
   public static String configFileName;
   public static double[] a;
   public static double[] h;
   public static double[] F;
   public static double[] upperThetaHidden;
   public static double[] upperThetaOutput;
   public static double[] lowerPsi;
   public static double[][] inputToHiddenWeights;
   public static double[][] hiddenToOutputWeights;
   public static double[][] truthTableInputs;
   public static double[][] truthTableExpectedOutputs;
   public static double[][] truthTableActualOutputs;

/*
 * The main() method creates an object of the ABCNetwork class and runs its 6 major methods.
 */
   public static void main(String[] args)
   {
      System.out.println(); // Printing clear line before all output to make it easier to read in the terminal.

      ABCBackprop network = new ABCBackprop();

      if (args.length > 0)
      {
         network.configureNetwork(args[0]);
      }
      else
      {
         network.configureNetwork(null);
      }

      network.echoParams();
      network.allocateMemory();
      network.populate();

      if (training)
      {
         network.trainNetwork();
         if (saveWeightsFile != null && !saveWeightsFile.equals(""))
         {
            saveWeights();
         }
      }
      
      network.runNetwork();
      network.results();
   } // public static void main(String[] args)

/*
 * The configureNetwork() method is where the user can edit certain parameters of the network such as the configuration, 
 * maxIterations cutoff, error threshold, whether it is training or running, the random number range, and the number of
 * test cases.
 */
   public void configureNetwork(String argGiven) 
   {
      if (argGiven == null)
      {
         configFileName = "configuration.txt"; // default name if no other name is given
      }
      else
      {
         configFileName = argGiven;
      }

      try
      {
         readConfigFile();
      }
      catch (Exception e)
      {
         System.out.println("Exception: " + e);
      }
   } // public void configureNetwork(String argGiven)

/*
 * The setTruthTable() method reads the configuration file to set the truth table.
 */
   public void setTruthTable()
   {
      int k, i, testCase;
      
      try
      {
         BufferedReader reader = new BufferedReader(new FileReader(configFileName));
         String line = reader.readLine();

/*
 * Need to read the first four lines of the configuration file that have already been read in readConfigFile().
 */
         line = reader.readLine();
         line = reader.readLine();
         line = reader.readLine();
         line = reader.readLine();

         StringTokenizer tokenizer = new StringTokenizer(line);

         for (testCase = 0; testCase < testCases; testCase++)
         {
            for (k = 0; k < nInputs; k++)
            {
               truthTableInputs[testCase][k] = Double.parseDouble(tokenizer.nextToken());
            }

            for (i = 0; i < nOutputs; i++)
            {
               truthTableExpectedOutputs[testCase][i] = Double.parseDouble(tokenizer.nextToken());
            }

            line = reader.readLine();
            tokenizer = new StringTokenizer(line);
         } // for (testCase = 0; testCase < testCases; testCase++)
      } // try
      catch (Exception e)
      {
         System.out.println("Exception " + e.toString());
      }
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
      
      if (!randomize && !loadWeightsFromFile)
      {
         System.out.println("Using manually inputted weights");
      }

      if (saveWeightsFile != null && !saveWeightsFile.equals(""))
      {
         System.out.println("Saving weights to file: " + saveWeightsFile);
      }

      if (loadWeightsFromFile)
      {
         System.out.println("Loading weights from file: " + initialWeightsFile);
      }

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
         upperThetaHidden = new double[nHidden];
         upperThetaOutput = new double[nOutputs];
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
         try
         {
            loadWeights();
         }
         catch (Exception e)
         {
            System.out.println("Exception: " + e);
         }
      } // if (loadWeightsFromFile)
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
 * The runNetwork() method goes through the running process for the network. It goes through each
 * test case, sets the input neurons and runs the network. It also calculates the time spent in running.
 */
   public void runNetwork() 
   {
      int truthTableIndex, k, i;

      startTime = System.currentTimeMillis();

      for (truthTableIndex = 0; truthTableIndex < testCases; truthTableIndex++)
      {
         for (k = 0; k < nInputs; k++)
         {
            a[k] = truthTableInputs[truthTableIndex][k];
         }

         run();

         for (i = 0; i < nOutputs; i++)
         {
            truthTableActualOutputs[truthTableIndex][i] = F[i];
         }
      } // for (truthTableIndex = 0; truthTableIndex < testCases; truthTableIndex++)

      endTime = System.currentTimeMillis();
   } // public void runNetwork() 

/*
 * The trainNetwork() method goes through the training process for the network. It goes through each test case, sets the input
 * neurons, runs the network, finds the new weights with backpropagation, runs the network to apply the new weights, calculates
 * the error, and calculates the total time taken for running.
 */
   public void trainNetwork() 
   {
      totalError = 0.0;
      averageError = 0.0;
      iterations = 0;

      boolean done = false;
      
      startTime = System.currentTimeMillis();

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

            runForTraining(truthTableIndex);
            trainBackprop();
            run();

            for (I = 0; I < nOutputs; I++)
            {
               double omega = truthTableExpectedOutputs[truthTableIndex][I] - F[I];
               
               totalError += 0.5 * omega * omega;
               truthTableActualOutputs[truthTableIndex][I] = F[I];
            } // for (I = 0; I < nOutputs; I++)
         } // for (truthTableIndex = 0; truthTableIndex < testCases; truthTableIndex++)

         averageError = totalError / ((double) testCases);

         iterations++;

         if (averageError < errorCutoff || iterations > maxIterations)
         {
            done = true;
         }
      } // while (!done)

      endTime = System.currentTimeMillis();
   } // public void trainNetwork() 

/*
 * The runForTraining() method runs the network while saving the theta and psi values, so they are only calculated once.
 */  
   public void runForTraining(int testCase)
   {
      int j, K, J, i;

      for (j = 0; j < nHidden; j++)
      {
         upperThetaHidden[j] = 0.0;

         for (K = 0; K < nInputs; K++)
         {
            upperThetaHidden[j] += a[K] * inputToHiddenWeights[K][j];
         }

         h[j] = activate(upperThetaHidden[j]);
      } // for (j = 0; j < nHidden; j++)

      for (i = 0; i < nOutputs; i++)
      {
         upperThetaOutput[i] = 0.0;

         for (J = 0; J < nHidden; J++)
         {
            upperThetaOutput[i] += h[J] * hiddenToOutputWeights[J][i];
         }

         F[i] = activate(upperThetaOutput[i]);

         double lowerOmega = truthTableExpectedOutputs[testCase][i] - F[i];

         lowerPsi[i] = lowerOmega * derivActivate(upperThetaOutput[i]);
      } // for (i = 0; i < nOutputs; i++)
   } // public void runForTraining(int testCase)

/*
 * The trainBackprop() method finds the new weights by training with backpropagation.
 */
   public void trainBackprop()
   {
      int k, j, i;
      double upperOmega, deltaHiddenToOutputWeight, upperPsi, deltaInputToHiddenWeight;

      for (j = 0; j < nHidden; j++)
      {
         upperOmega = 0.0;

         for (i = 0; i < nOutputs; i++)
         {
            upperOmega += lowerPsi[i] * hiddenToOutputWeights[j][i];

            deltaHiddenToOutputWeight = lambda * h[j] * lowerPsi[i];

            hiddenToOutputWeights[j][i] += deltaHiddenToOutputWeight;
         } // for (i = 0; i < nOutputs; i++)

         upperPsi = upperOmega * derivActivate(upperThetaHidden[j]);

         for (k = 0; k < nInputs; k++)
         {
            deltaInputToHiddenWeight = lambda * a[k] * upperPsi;

            inputToHiddenWeights[k][j] += deltaInputToHiddenWeight;
         } // for (k = 0; k < nInputs; k++)
      } // for (j = 0; j < nHidden; j++)
   } // public void trainBackprop()

/*
 * The run() method calculates the hidden and output activations based on the current values in the activations and the weights. 
 */
   public void run()
   {
      int j, K, i, J;
      double thetaHidden, thetaOutput;

      for (j = 0; j < nHidden; j++)
      {
         thetaHidden = 0.0;

         for (K = 0; K < nInputs; K++)
         {
            thetaHidden += a[K] * inputToHiddenWeights[K][j];
         }

         h[j] = activate(thetaHidden);
      } // for (j = 0; j < nHidden; j++)

      for (i = 0; i < nOutputs; i++)
      {
         thetaOutput = 0.0;

         for (J = 0; J < nHidden; J++) 
         {
            thetaOutput += h[J] * hiddenToOutputWeights[J][i];
         }

         F[i] = activate(thetaOutput);
      } // for (i = 0; i < nOutputs; i++)
   } // public void runNetwork()

/*
 * The results() method prints out the training exit info, truth table, and the output values of the network.
 */
   public void results()
   {
      int rowTruthTable, k, i;

      if (training)
      {
         System.out.println("Training Exit Information");
         System.out.println("------------------------------------");

         if (iterations > maxIterations)
         {
            System.out.println("Reason for end of training: Over the max iterations cutoff (" + maxIterations + ").");
         }
         
         if (averageError < errorCutoff)
         {
            System.out.println("Reason for end of training: Average error under the error cutoff (" + errorCutoff + ").");
         }

         System.out.println("Iterations reached: " + iterations);
         System.out.println("Error reached: " + averageError);
      } // if (training)

      System.out.println("Time taken: " + (endTime - startTime) + "ms");
      System.out.println();

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
 * The activate() method plugs in a given x value into the activation function and returns the result.
 */
   public double activate(double x)
   {
      return 1.0/(1.0 + Math.exp(-x));
   }

/*
 * The derivActivate() method plugs in a given x into the derivative of the activation function and returns the result.
 */
   public double derivActivate(double x)
   {
      double activationFunction = activate(x);

      return activationFunction * (1.0 - activationFunction);
   } // public double derivActivate(double x)

/*
 * The saveWeights() method saves the current weights in an external file.
 */
   public static void saveWeights()
   {
      int k, j, i;

      try 
      {
         FileOutputStream fos = new FileOutputStream(saveWeightsFile);
         DataOutputStream dos = new DataOutputStream(fos);

         for (k = 0; k < nInputs; k++)
         {
            for (j = 0; j < nHidden; j++)
            {
               dos.writeDouble(inputToHiddenWeights[k][j]);
            }
         } // for (k = 0; k < nInputs; k++)

         for (j = 0; j < nHidden; j++)
         {
            for (i = 0; i < nOutputs; i++)
            {
               dos.writeDouble(hiddenToOutputWeights[j][i]);
            }
         } // for (j = 0; j < nHidden; j++)
      } // try
      catch (Exception e) 
      {
         System.out.println("Exception" + e.toString());
      }
   } // public void saveWeights()

/*
 * The loadWeights() method loads in weights from an external file.
 */
   public void loadWeights() throws Exception
   {
      int k, j, i;

      try 
      {
         FileInputStream fis = new FileInputStream(initialWeightsFile);
         DataInputStream dis = new DataInputStream(fis);

         for (k = 0; k < nInputs; k++)
         {
            for (j = 0; j < nHidden; j++)
            {
               inputToHiddenWeights[k][j] = dis.readDouble();
               // System.out.println(inputToHiddenWeights[k][j]);
            }
         } // for (k = 0; k < nInputs; k++)

         for (j = 0; j < nHidden; j++)
         {
            for (i = 0; i < nOutputs; i++)
            {
               hiddenToOutputWeights[j][i] = dis.readDouble();
            }
         } // for (j = 0; j < nHidden; j++)
      } // try
      catch (IOException e)
      {
         System.out.println("Exception " + e.toString());
      }
   } // public void loadWeights()

/*
 * 
 */
   public void readConfigFile() throws Exception
   {
      int testCase;

      try (BufferedReader reader = new BufferedReader(new FileReader(configFileName)))
      {
         String line = reader.readLine();
         StringTokenizer tokenizer = new StringTokenizer(line);
         String token = tokenizer.nextToken();

         if (token.equals("T"))
         {
            training = true;
         }
         else if (token.equals("R"))
         {
            training = false;
         }
         else 
         {
            throw new Exception("Configuration file does not follow the format.");   
         }

         line = reader.readLine();
         tokenizer = new StringTokenizer(line);
         token = tokenizer.nextToken();

         if (token.equals("R"))
         {
            randomize = true;
         }
         else if (token.equals("P"))
         {
            randomize = false;
            loadWeightsFromFile = true;
         }
         else if (tokenizer.nextToken().equals("M"))
         {
            randomize = false;
            loadWeightsFromFile = false;
         }
         else 
         {
            throw new Exception("Configuration file does not follow the format.");   
         }

         line = reader.readLine();
         tokenizer = new StringTokenizer(line);

         nInputs = Integer.parseInt(tokenizer.nextToken());
         nHidden = Integer.parseInt(tokenizer.nextToken());
         nOutputs = Integer.parseInt(tokenizer.nextToken());

         line = reader.readLine();
         tokenizer = new StringTokenizer(line);

         testCases = Integer.parseInt(tokenizer.nextToken());

         line = reader.readLine();
         tokenizer = new StringTokenizer(line);

         for (testCase = 0; testCase < testCases; testCase++)
         {
            line = reader.readLine(); // need to read over truth table lines as it is going to be dealt with in allocateMemory().
         }

         tokenizer = new StringTokenizer(line);
         maxIterations = Integer.parseInt(tokenizer.nextToken());

         line = reader.readLine();
         tokenizer = new StringTokenizer(line);
         errorCutoff = Double.parseDouble(tokenizer.nextToken());

         line = reader.readLine();
         tokenizer = new StringTokenizer(line);
         lambda = Double.parseDouble(tokenizer.nextToken());

         line = reader.readLine();
         tokenizer = new StringTokenizer(line);
         randMin = Double.parseDouble(tokenizer.nextToken());
         randMax = Double.parseDouble(tokenizer.nextToken());
         
         line = reader.readLine();
         if (line != null)
         {
            saveWeightsToFile = true;
            saveWeightsFile = line;
            line = reader.readLine();
            if (line != null && loadWeightsFromFile)
            {
               initialWeightsFile = line;
            }
            else if (loadWeightsFromFile && line == null)
            {
/*
 * If no initial weights file is given when preloading weights, it will preload from the saved weights file.
 */
               initialWeightsFile = saveWeightsFile;
            } // else if (loadWeightsFromFile && line == null)
         } // if (line != null)
      } // try (BufferedReader reader = new BufferedReader(new FileReader(configFileName)))
      catch (Exception e)
      {
         System.out.println("Exception " + e.toString());
      }
   } // public void readConfigFile()
} // public class ABCBackprop