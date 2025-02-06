import java.io.*;
import java.util.*;

/*
 * @author Sanaa Bhorkar
 * @version 04/16/2024
 * 
 * This N-Layer perceptron and is built based off of the design document "5-N-Layer." 
 * The network can train a network based off of randomly assigned weights or simply run when given weights from a user. The
 * network trains using gradient descent and back propagation. The program uses a DataOutputStream/DataInputStream to save the
 * weights in a binary format or load them from the binary format. It also uses a BufferedReader and StringTokenizer to read the
 * configuration file.
 * 
 * Format of configuration file if training:
 * T                  // T for training
 * R                  // R for randomized initial weights, P for preloaded initial weights, M for manual inputs in allocateMemory()
 * 4                  // number of total layers (needs to include the output layer)
 * 2 1 1 3            // number of units in each layer 
 * 4                  // number of test cases
 * testCase.txt       // name of the file with the input activation values
 * 0 0 0              // truth table
 * 0 1 1
 * 0 1 1
 * 1 1 0
 * 100000             // maximum number of iterations 
 * 10000              // # of iterations for keepAlive messages
 * 2E-4               // error cutoff
 * 0.3                // learning rate (lambda)  
 * 0.1 1.5            // random number range
 * weights.txt        // the output file that weights will be saved to (can be omitted to not save weights)
 * initialWeights.txt // the file that the weights will be read from (will be omitted when not loading weights)
 * 
 * Format of configuration file if running:
 * R                  // R for running
 * P                  // R for randomized initial weights, P for preloaded initial weights, M for manual inputs in allocateMemory()
 * 4                  // number of layers (needs to include the output layer)
 * 2 1 1 1            // number of units in each layer 
 * 4                  // number of test cases
 * testCase.txt       // name of the file with the test case activation values
 * T                  // T for including a truth table, F for not
 * 0 0 0              // truth table if included
 * 1 1 1
 * 2 2 2
 * 3 1 2
 * weights.txt        // the output file that weights will be saved to (can be omitted to not save weights)
 * initialWeights.txt // the file that the weights will be read from (will be omitted when not loading weights)
 * 
 * Table of Contents:
 * - public static void main(String[] args)
 * - public void configureNetwork(String argGiven) 
 * - public void setTruthTable()
 * - public void echoParams()
 * - public void allocateMemory() 
 * - public void populate() 
 * - public double randomize(double low, double high)
 * - public void runNetwork() 
 * - public void calculateErrorForRunning()
 * - public void trainNetwork()
 * - public void runForTraining(int testCase)
 * - public void trainBackprop()
 * - public void run()
 * - public void results()
 * - public double activate(double theta)
 * - public double derivActivate(double theta)
 * - public double sigmoid(double x)
 * - public double derivSigmoid(double x)
 * - public void saveWeights()
 * - public void loadWeights() throws Exception
 * - public void readConfigFile() throws Exception
 * - public void readTestCaseFile()
 */
public class NLayer 
{
   public static final int INDEX_INPUT_LAYER = 0;
   public static final int INDEX_FIRST_HIDDEN_LAYER = 1;

   public static int nLayers;          
   public static int testCases;
   public static int maxIterations;
   public static int keepAlive;
   public static int iterations;
   public static int deepestLayer;
   public static int indexOutputLayer;
   public static int indexLastHiddenLayer;
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
   public static boolean stopForWeights;
   public static boolean configNameGiven;
   public static boolean loadWeightsFromFile;
   public static boolean saveWeightsToFile;
   public static boolean usingTruthTable;
   public static String initialWeightsFile;
   public static String saveWeightsFile;
   public static String configFileName;
   public static String testCaseFileName;
   public static int[] layerSizes;
   public static double[][] a;
   public static double[][] upperTheta;
   public static double[][] psi;
   public static double[][][] weights;
   public static double[][] truthTableInputs;
   public static double[][] truthTableExpectedOutputs;
   public static double[][] truthTableActualOutputs;

/*
 * The main() method creates an object of the NLayer class and runs its 6 major methods.
 */
   public static void main(String[] args)
   {
      System.out.println(); // Printing clear line before all output to make it easier to read in the terminal.

      NLayer network = new NLayer();

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

      if (!stopForWeights)  // stopForWeights is true if the configuration in the weights file does not match
      {
         if (training)
         {
            network.trainNetwork();
            if (saveWeightsFile != null && !saveWeightsFile.equals(""))
            {
               saveWeights();
            }
         } // if (training)
      
         network.runNetwork();
         network.results();
      } // if (!stopForWeights)
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
         configFileName = "configuration.txt";     // default name if no other name is given
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

      indexOutputLayer = nLayers - 1;              // nLayers - 1 because I am using 0-index
      indexLastHiddenLayer = indexOutputLayer - 1; // the last hidden layer is the layer to the left of the output layer
      deepestLayer = 0;                            // deepestLayer will contain the size of the layer with the most nodes

      int n;

      for (n = 0; n < nLayers; n++)
      {
         if (layerSizes[n] > deepestLayer)
         {
            deepestLayer = layerSizes[n];
         }
      } // for (n = 0; n < nLayers; n++)
   } // public void configureNetwork(String argGiven)

/*
 * The setTruthTable() method reads the configuration file to set the truth table.
 */
   public void setTruthTable()
   {
      int n, i, testCase;
      
      try
      {
         BufferedReader reader = new BufferedReader(new FileReader(configFileName));
         String line = reader.readLine();

/*
 * Need to read the first six lines of the configuration file that have already been read in readConfigFile().
 */
         line = reader.readLine();
         line = reader.readLine();
         line = reader.readLine();
         line = reader.readLine();
         line = reader.readLine();
         line = reader.readLine();

         if (!training) // takes care of extra line for a running configuration that checks if a truth table is given
         {
            line = reader.readLine();
         }

         StringTokenizer tokenizer = new StringTokenizer(line);

         n = indexOutputLayer;

         for (testCase = 0; testCase < testCases; testCase++)
         {
            
            for (i = 0; i < layerSizes[n]; i++)
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
      int n;

      System.out.print("Network configuration: ");
    
      for (n = 0; n < nLayers; n++)
      {
         System.out.print(layerSizes[n] + " ");
      }

      System.out.println();
      System.out.println("Training?: " + training);
      System.out.println("Randomize weights?: " + randomize);
      
      if (!randomize && !loadWeightsFromFile)
      {
         System.out.println("Using manually inputted weights");
      }

      System.out.println("Reading configuration from file: " + configFileName);

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
         
         if (keepAlive != 0)
         {
            System.out.println("# of iterations for keep alive messages: " + keepAlive);
         }
      } // if (training)

      System.out.println("Using truth table?: " + usingTruthTable);
      System.out.println();
   } // public void echoParams() 

/*
 * The allocateMemory() method instantiates all the arrays that the network is going to use. Examples of the arrays are the
 * input, hidden, and output activations and all the weights.
 */
   public void allocateMemory() 
   {
      int n;
      
      a = new double[nLayers][deepestLayer];
      weights = new double[nLayers][deepestLayer][deepestLayer];

      n = INDEX_INPUT_LAYER;
      truthTableInputs = new double[testCases][layerSizes[n]];

      n = indexOutputLayer;
      truthTableExpectedOutputs = new double[testCases][layerSizes[n]];
      truthTableActualOutputs = new double[testCases][layerSizes[n]];

      if (training)
      {
         upperTheta = new double[nLayers][deepestLayer];
         psi = new double[nLayers][deepestLayer];
      } // if (training)
   } // public void allocateMemory() 

/*
 * The populate() method populates the weight arrays with either fixed weight values or randomized values, depending on whether 
 * or not the network is training. We assume that "training" the network will use randomized weights while simply running the 
 * network will used fixed weights.
 */
   public void populate() 
   {
      stopForWeights = false;

      readTestCaseFile();

      if (usingTruthTable)
      {
         setTruthTable();
      }

      int n;

      if (loadWeightsFromFile)
      {
         try
         {
            loadWeights();
         }
         catch (IOException e)
         {
            System.out.println(e);
            stopForWeights = true;
         }
         catch (Exception e)
         {
            System.out.println("Exception: " + e);
         }
      } // if (loadWeightsFromFile)
      else if (!randomize && !loadWeightsFromFile)
      {
         n = INDEX_INPUT_LAYER;
         weights[n][0][0] = 0.75;
         weights[n][1][0] = 0.6;
      } // else if (!randomize && !loadWeightsFromFile)
      else
      {
         int parentIndex, childIndex;

         for (n = 0; n < indexOutputLayer; n++)
         {
            for (parentIndex = 0; parentIndex < layerSizes[n]; parentIndex++)
            {
               for (childIndex = 0; childIndex < layerSizes[n + 1]; childIndex++)
               {
                  weights[n][parentIndex][childIndex] = randomize(randMin, randMax);
               }
            } // for (parentIndex = 0; parentIndex < layerSizes[n]; parentIndex++)
         } // for (n = 0; n < indexOutputLayer; n++)
      } // else
   } // public void populate() 

/*
 * The randomize() method takes in the low and high numbers in the random range and calculates a random number. This method
 * is used when calculating random weight values.
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
      int truthTableIndex, n, x, i;

      startTime = System.currentTimeMillis();

      for (truthTableIndex = 0; truthTableIndex < testCases; truthTableIndex++)
      {
         n = INDEX_INPUT_LAYER;

         for (x = 0; x < layerSizes[n]; x++)
         {
            a[n][x] = truthTableInputs[truthTableIndex][x];
         }

         run();

         n = indexOutputLayer;

         for (i = 0; i < layerSizes[n]; i++)
         {
            truthTableActualOutputs[truthTableIndex][i] = a[n][i];
         }
      } // for (truthTableIndex = 0; truthTableIndex < testCases; truthTableIndex++)

      endTime = System.currentTimeMillis();

      if (!training && usingTruthTable) // calculates the error if the user wants it whlie just running
      {
         calculateErrorForRunning();
      }
   } // public void runNetwork() 

/*
 * The calculateErrorForRunning() method calculates the error when running the network only if the user gave the network
 * a truth table.
 */
   public void calculateErrorForRunning()
   {
      int truthTableIndex, n, I;
      double omega;

      n = indexOutputLayer;

      totalError = 0.0;

      for (truthTableIndex = 0; truthTableIndex < testCases; truthTableIndex++)
      {
         for (I = 0; I < layerSizes[n]; I++)
         {
            omega = truthTableExpectedOutputs[truthTableIndex][I] - truthTableActualOutputs[truthTableIndex][I];
            
            totalError += 0.5 * omega * omega;
         } // for (I = 0; I < layerSizes[n]; I++)
      } // for (truthTableIndex = 0; truthTableIndex < testCases; truthTableIndex++)

      averageError = totalError / ((double) testCases);
   } // public void calculateErrorForRunning()

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
         int truthTableIndex, n, x, I;
         double omega;

         totalError = 0.0;
         averageError = 0.0;

         for (truthTableIndex = 0; truthTableIndex < testCases; truthTableIndex++)
         {
            n = INDEX_INPUT_LAYER;

            for (x = 0; x < layerSizes[n]; x++)
            {
               a[n][x] = truthTableInputs[truthTableIndex][x];
            }

            runForTraining(truthTableIndex);
            trainBackprop();
            run();

            n = indexOutputLayer;

            for (I = 0; I < layerSizes[n]; I++)
            {
               omega = truthTableExpectedOutputs[truthTableIndex][I] - a[n][I];
               
               totalError += 0.5 * omega * omega;
               truthTableActualOutputs[truthTableIndex][I] = a[n][I];
            } // for (I = 0; I < layerSizes[n]; I++)
         } // for (truthTableIndex = 0; truthTableIndex < testCases; truthTableIndex++)

         averageError = totalError / ((double) testCases);

         iterations++;

         if (averageError < errorCutoff || iterations > maxIterations)
         {
            done = true;
         }

         if (keepAlive != 0 && (iterations % keepAlive == 0))
         {
            System.out.println("Iteration " + iterations + ", Error " + averageError);
         }
      } // while (!done)

      endTime = System.currentTimeMillis();
   } // public void trainNetwork() 

/*
 * The runForTraining() method runs the network while saving the theta and psi values, so they are only calculated once.
 */  
   public void runForTraining(int testCase)
   {
      int n, J, i, parentIndex, childIndex;
      double upperThetaI;

      for (n = INDEX_FIRST_HIDDEN_LAYER; n < indexOutputLayer; n++) // loop over hidden layers
      {
         for (parentIndex = 0; parentIndex < layerSizes[n]; parentIndex++)
         {
            upperTheta[n][parentIndex] = 0.0;

            for (childIndex = 0; childIndex < layerSizes[n - 1]; childIndex++)
            {
               upperTheta[n][parentIndex] += a[n - 1][childIndex] * weights[n - 1][childIndex][parentIndex];
            }

            a[n][parentIndex] = activate(upperTheta[n][parentIndex]);
         } // for (parentIndex = 0; parentIndex < layerSizes[n]; parentIndex++)
      } // for (n = INDEX_FIRST_HIDDEN_LAYER; n < indexOutputLayer; n++)

      n = indexOutputLayer;

      for (i = 0; i < layerSizes[n]; i++)
      {
         upperThetaI = 0.0;

         for (J = 0; J < layerSizes[n - 1]; J++)
         {
            upperThetaI += a[n - 1][J] * weights[n - 1][J][i];
         }

         a[n][i] = activate(upperThetaI);
         psi[n][i] = (truthTableExpectedOutputs[testCase][i] - a[n][i]) * derivActivate(upperThetaI);
      } // for (i = 0; i < layerSizes[n]; i++)
   } // public void runForTraining(int testCase)

/*
 * The trainBackprop() method finds the new weights by training with backpropagation.
 */
   public void trainBackprop()
   {
      int n, x, m, K, parentIndex, childIndex;
      double upperOmega, upperPsi;

      for (n = indexLastHiddenLayer; n > INDEX_FIRST_HIDDEN_LAYER; n--) // need to loop over all hidden layers except for first one
      {
         for (parentIndex = 0; parentIndex < layerSizes[n]; parentIndex++)
         {
            upperOmega = 0.0;

            for (childIndex = 0; childIndex < layerSizes[n + 1]; childIndex++)
            {
               upperOmega += psi[n + 1][childIndex] * weights[n][parentIndex][childIndex];
               weights[n][parentIndex][childIndex] += lambda * a[n][parentIndex] * psi[n + 1][childIndex];
            }

            psi[n][parentIndex] = upperOmega * derivActivate(upperTheta[n][parentIndex]);
         } // for (parentIndex = 0; parentIndex < layerSizes[n]; parentIndex++)
      } // for (n = indexLastHiddenLayer; n > INDEX_FIRST_HIDDEN_LAYER; n--)

      n = INDEX_FIRST_HIDDEN_LAYER;

      for (m = 0; m < layerSizes[n]; m++)
      {
         upperOmega = 0.0;

         for (K = 0; K < layerSizes[n + 1]; K++)
         {
            upperOmega += psi[n + 1][K] * weights[n][m][K];
            weights[n][m][K] += lambda * a[n][m] * psi[n + 1][K];
         }

         upperPsi = upperOmega * derivActivate(upperTheta[n][m]);

         for (x = 0; x < layerSizes[n - 1]; x++)
         {
            weights[n - 1][x][m] += lambda * a[n - 1][x] * upperPsi;
         }
      } // for (m = 0; m < layerSizes[n]; m++)
   } // public void trainBackprop()

/*
 * The run() method calculates the hidden and output activations based on the current values in the activations and the weights. 
 */
   public void run()
   {
      int n, parentIndex, childIndex;
      double theta;

      for (n = INDEX_FIRST_HIDDEN_LAYER; n < nLayers; n++)
      {
         for (parentIndex = 0; parentIndex < layerSizes[n]; parentIndex++)
         {
            theta = 0.0;

            for (childIndex = 0; childIndex < layerSizes[n - 1]; childIndex++)
            {
               theta += a[n - 1][childIndex] * weights[n - 1][childIndex][parentIndex];
            }

            a[n][parentIndex] = activate(theta);
         } // for (parentIndex = 0; parentIndex < layerSizes[n]; parentIndex++)
      } // for (n = INDEX_FIRST_HIDDEN_LAYER; n < nLayers; n++)
   } // public void runNetwork()

/*
 * The results() method prints out the training exit info, truth table, and the output values of the network.
 */
   public void results()
   {
      int rowTruthTable, n, x, i;

      System.out.println();

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
      } // if (training)

      if (usingTruthTable)
      {
         System.out.println("Error reached: " + averageError);
      }

      System.out.println("Time taken: " + (endTime - startTime) + "ms");
      System.out.println();

      System.out.println("Truth Table");
      System.out.println("------------------------------------");

      String truthTableOutput = "";

      n = indexOutputLayer;

      // if (usingTruthTable)
      // {
      //    System.out.println();
      //    System.out.println("Expected Output");

      //    for (rowTruthTable = 0; rowTruthTable < testCases; rowTruthTable++)
      //    {
      //       for (i = 0; i < layerSizes[n]; i++)
      //       {
      //          truthTableOutput += truthTableExpectedOutputs[rowTruthTable][i] + "\t";
      //       }

      //       System.out.println(truthTableOutput);
      //       truthTableOutput = "";
      //    } // for (rowTruthTable = 0; rowTruthTable < testCases; rowTruthTable++)
      // } // if (usingTruthTable)

      System.out.println();
      System.out.println("Actual Output");

      for (rowTruthTable = 0; rowTruthTable < testCases; rowTruthTable++)
      {
         for (i = 0; i < layerSizes[n]; i++)
         {
            truthTableOutput += truthTableActualOutputs[rowTruthTable][i] + "\t";
         }

         System.out.println(truthTableOutput);
         truthTableOutput = "";
      } // for (rowTruthTable = 0; rowTruthTable < testCases; rowTruthTable++)
   } // public void results()

/*
 * The activate() method plugs in a given x value into the activation function and returns the result.
 */
   public double activate(double theta)
   {
      return sigmoid(theta);
   }

/*
 * The derivActivate() method plugs in a given x into the derivative of the activation function and returns the result.
 */
   public double derivActivate(double theta)
   {
      return derivSigmoid(theta);
   }

/*
 * The sigmoid() method is one funnction that can be used as the activation function.
 */
   public double sigmoid(double x)
   {
      return 1.0/(1.0 + Math.exp(-x));
   }

/*
 * The derivSigmoid() method is the associated derivative with the sigmoid activation function.
 */
   public double derivSigmoid(double x)
   {
      double sigmoid = sigmoid(x);

      return sigmoid * (1.0 - sigmoid);  
   } // public double derivSigmoid(double x)

/*
 * The saveWeights() method saves the current weights in an external file.
 */
   public static void saveWeights()
   {
      int n, parentIndex, childIndex;

      FileOutputStream fos = null;
      DataOutputStream dos = null;

      try 
      {
         fos = new FileOutputStream(saveWeightsFile);
         dos = new DataOutputStream(fos);

         for (n = 0; n < nLayers; n++)
         {
            dos.writeInt(layerSizes[n]);
         }

         for (n = 0; n < indexOutputLayer; n++)
         {
            for (parentIndex = 0; parentIndex < layerSizes[n]; parentIndex++)
            {
               for (childIndex = 0; childIndex < layerSizes[n + 1]; childIndex++)
               {
                  dos.writeDouble(weights[n][parentIndex][childIndex]);
               }
            } // for (parentIndex = 0; parentIndex < layerSizes[n]; parentIndex++)
         } // for (n = 0; n < indexOutputLayer; n++)
      } // try
      catch (Exception e) 
      {
         System.out.println("Exception" + e.toString());
      }
      finally
      {
         try 
         {
            if (dos != null)
            {
               dos.close();
            }
         } // try 
         catch (IOException e) 
         {
            System.out.println("Exception " + e.toString());
         }
      } // finally
   } // public void saveWeights()

/*
 * The loadWeights() method loads in weights from an external file.
 */
   public void loadWeights() throws Exception
   {
      int n, parentIndex, childIndex;

      FileInputStream fis = null;
      DataInputStream dis = null;

      try 
      {
         fis = new FileInputStream(initialWeightsFile);
         dis = new DataInputStream(fis);

         for (n = 0; n < nLayers; n++)
         {
            if (dis.readInt() != layerSizes[n])
            {
               dis.close();
               throw new IOException("Loading weights from file with the incorrect number of activations in a layer");
            }
         } // for (n = 0; n < nLayers; n++)

         for (n = 0; n < indexOutputLayer; n++)
         {
            for (parentIndex = 0; parentIndex < layerSizes[n]; parentIndex++)
            {
               for (childIndex = 0; childIndex < layerSizes[n + 1]; childIndex++)
               {
                  weights[n][parentIndex][childIndex] = dis.readDouble();
               }
            } // for (parentIndex = 0; parentIndex < layerSizes[n]; parentIndex++)
         } // for (n = 0; n < indexOutputLayer; n++)
      } // try
      catch (IOException e)
      {
         System.out.println("Exception " + e.toString());
         stopForWeights = true;
      }
      catch (Exception e)
      {
         System.out.println("Exception " + e.toString());
      }
      finally
      {
         try 
         {
            if (dis != null)
            {
               dis.close();
            }
         } // try
         catch (IOException e) 
         {
            System.out.println("Exception " + e.toString());
         }
      } // finally
   } // public void loadWeights()

/*
 * The readConfigFile() method reads the given, or default if not given, configuration file and sets all relevant configuration
 * parameters.
 */
   public void readConfigFile() throws Exception
   {
      int n, testCase;

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

         nLayers = Integer.parseInt(tokenizer.nextToken());

         layerSizes = new int[nLayers];

         line = reader.readLine();
         tokenizer = new StringTokenizer(line);

         for (n = 0; n < nLayers; n++)
         {
            layerSizes[n] = Integer.parseInt(tokenizer.nextToken());

            if (layerSizes[n] == 0)
            {
               throw new IOException("Network cannot have 0 activations in a layer.");
            }
         } // for (n = 0; n < nLayers; n++)

         line = reader.readLine();
         tokenizer = new StringTokenizer(line);

         testCases = Integer.parseInt(tokenizer.nextToken());

         line = reader.readLine();

         testCaseFileName = line;

         line = reader.readLine();
         tokenizer = new StringTokenizer(line);

         if (training || tokenizer.nextToken().equals("T"))
         {
            usingTruthTable = true;

            for (testCase = 0; testCase < testCases; testCase++)
            {
               line = reader.readLine(); // need to read over truth table lines as it is going to be dealt with in populate().
            }
         } // if (training || tokenizer.nextToken().equals("T"))
         
         if (training)
         {
            tokenizer = new StringTokenizer(line);
            maxIterations = Integer.parseInt(tokenizer.nextToken());

            line = reader.readLine();
            tokenizer = new StringTokenizer(line);
            keepAlive = Integer.parseInt(tokenizer.nextToken());

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
         } // if (training)
         
         line = reader.readLine();

         if (line != null)
         {
            saveWeightsToFile = !line.equals("");
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

/*
 * The readTestCaseFile() reads the file with the test case activation values.
 */
   public void readTestCaseFile() 
   {
      int n, x, testCase;

      try 
      {
         DataInputStream in = new DataInputStream(new FileInputStream(testCaseFileName));

         n = INDEX_INPUT_LAYER;

         for (testCase = 0; testCase < testCases; testCase++) 
         {
            for (x = 0; x < layerSizes[n]; x++) 
            {
               if (in.available() > 0)
               {
                  truthTableInputs[testCase][x] = in.readDouble();
               } 
               else 
               {
                  System.out.println("No more data available in the file, but not all inputs have been read.");
                  return;
               }
            }
         } // for (testCase = 0; testCase < testCases; testCase++)

         in.close();
      } // try
      catch (IOException e) 
      {
         System.out.println("IOException " + e.toString());
      }
   } // public void readTestCaseFile()
} // public class NLayer