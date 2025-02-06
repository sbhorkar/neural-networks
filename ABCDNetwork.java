import java.io.*;
import java.util.*;

/*
 * @author Sanaa Bhorkar
 * @version 03/21/2024
 * 
 * This A-B-C-D perceptron and is built based off of the design document "4-Three Layer Network." 
 * The A-B-C-D network can train a network based off of randomly assigned weights or simply run when given weights from a user. The
 * network trains using gradient descent and back propagation. The program uses a DataOutputStream/DataInputStream to save the
 * weights in a binary format or load them from the binary format. It also uses a BufferedReader and StringTokenizer to read the
 * configuration file.
 * 
 * Format of configuration file if training:
 * T                  // T for training
 * R                  // R for randomized initial weights, P for preloaded initial weights, M for manual inputs in allocateMemory()
 * 2 1 1 3            // number of units in each layer 
 * 4                  // number of test cases
 * testCase.txt       // name of the file with the test case activation values
 * 0 0 0 0 0          // truth table
 * 0 1 0 1 1
 * 1 0 0 1 1
 * 1 1 1 1 0
 * 100000             // maximum number of iterations 
 * 2E-4               // minimum error
 * 0.3                // learning rate (lambda)  
 * 0.1 1.5            // random number range
 * weights.txt        // the output file that weights will be saved to (can be omitted to not save weights)
 * initialWeights.txt // the file that the weights will be read from (will be omitted when not loading weights)
 * 
 * Format of configuration file if running:
 * R                  // R for running
 * P                  // R for randomized initial weights, P for preloaded initial weights, M for manual inputs in allocateMemory()
 * 2 1 1 3            // number of units in each layer 
 * 4                  // number of test cases
 * testCase.txt       // name of the file with the test case activation values
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
public class ABCDNetwork 
{
   public static final int N_LAYERS = 4;          // A-B-C-D network will always have 4 layers
   public static final int INDEX_INPUT_LAYER = 0; // starting at 0 as Java starts at 0 and thought it would be easier to use that
   public static final int INDEX_HIDDEN_LAYER_1 = 1;
   public static final int INDEX_HIDDEN_LAYER_2 = 2;
   public static final int INDEX_OUTPUT_LAYER = 3;

   public static int nInputs;
   public static int nHidden1;
   public static int nHidden2;       
   public static int nOutputs;           
   public static int testCases;
   public static int maxIterations;
   public static int iterations;
   public static int deepestLayer;
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
   public static String initialWeightsFile;
   public static String saveWeightsFile;
   public static String configFileName;
   public static String testCaseFileName;
   public static double[][] a;
   public static double[][] upperTheta;
   public static double[][] psi;
   public static double[][][] weights;
   public static double[][] truthTableInputs;
   public static double[][] truthTableExpectedOutputs;
   public static double[][] truthTableActualOutputs;

/*
 * The main() method creates an object of the ABCDNetwork class and runs its 6 major methods.
 */
   public static void main(String[] args)
   {
      System.out.println(); // Printing clear line before all output to make it easier to read in the terminal.

      ABCDNetwork network = new ABCDNetwork();

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

      if (!stopForWeights)            // stopForWeights is true if the configuration in the weights file does not match
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

      deepestLayer = 0;                        // deepestLayer will contain the size of the layer with the most nodes

      if (nInputs > deepestLayer)
      {
         deepestLayer = nInputs;
      }
      if (nHidden1 > deepestLayer)
      {
         deepestLayer = nHidden1;
      }
      if (nHidden2 > deepestLayer)
      {
         deepestLayer = nHidden2;
      }
      if (nOutputs > deepestLayer)
      {
         deepestLayer = nOutputs;
      }
   } // public void configureNetwork(String argGiven)

/*
 * The setTruthTable() method reads the configuration file to set the truth table.
 */
   public void setTruthTable()
   {
      int i, testCase;
      
      try
      {
         BufferedReader reader = new BufferedReader(new FileReader(configFileName));
         String line = reader.readLine();

/*
 * Need to read the first five lines of the configuration file that have already been read in readConfigFile().
 */
         line = reader.readLine();
         line = reader.readLine();
         line = reader.readLine();
         line = reader.readLine();
         line = reader.readLine();

         StringTokenizer tokenizer = new StringTokenizer(line);

         for (testCase = 0; testCase < testCases; testCase++)
         {
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
      System.out.println("Network configuration: " + nInputs + "-" + nHidden1 + "-" + nHidden2 + "-" + nOutputs);
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
         System.out.println();
      } // if (training)
   } // public void echoParams() 

/*
 * The allocateMemory() method instantiates all the arrays that the network is going to use. Examples of the arrays are the
 * input, hidden, and output activations and all the weights.
 */
   public void allocateMemory() 
   {
      a = new double[N_LAYERS][deepestLayer];
      weights = new double[N_LAYERS][deepestLayer][deepestLayer];
      truthTableInputs = new double[testCases][nInputs];
      truthTableExpectedOutputs = new double[testCases][nOutputs];
      truthTableActualOutputs = new double[testCases][nOutputs];

      if (training)
      {
         upperTheta = new double[N_LAYERS][deepestLayer];
         psi = new double[N_LAYERS][deepestLayer];
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

      if (training)
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

         n = INDEX_HIDDEN_LAYER_1;
         weights[n][0][0] = 0.75;
         weights[n][1][0] = 0.6;

         n = INDEX_HIDDEN_LAYER_2;
         weights[n][0][0] = 0.5;
         weights[n][0][1] = 0.75;
         weights[n][0][2] = 0.35;
      } // else if (!randomize && !loadWeightsFromFile)
      else
      {
         int m, k, j, i;

         n = INDEX_INPUT_LAYER;

         for (m = 0; m < nInputs; m++)
         {
            for (k = 0; k < nHidden1; k++)
            {
               weights[n][m][k] = randomize(randMin, randMax);
            }
         } // for (m = 0; m < nInputs; m++)

         n = INDEX_HIDDEN_LAYER_1;

         for (k = 0; k < nHidden1; k++)
         {
            for (j = 0; j < nHidden2; j++)
            {
               weights[n][k][j] = randomize(randMin, randMax);
            }
         } // for (k = 0; k < nHidden1; k++)

         n = INDEX_HIDDEN_LAYER_2;

         for (j = 0; j < nHidden2; j++)
         {
            for (i = 0; i < nOutputs; i++)
            {
               weights[n][j][i] = randomize(randMin, randMax);
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
      int truthTableIndex, n, m, i;

      startTime = System.currentTimeMillis();

      for (truthTableIndex = 0; truthTableIndex < testCases; truthTableIndex++)
      {
         n = INDEX_INPUT_LAYER;

         for (m = 0; m < nInputs; m++)
         {
            a[n][m] = truthTableInputs[truthTableIndex][m];
         }

         run();

         n = INDEX_OUTPUT_LAYER;

         for (i = 0; i < nOutputs; i++)
         {
            truthTableActualOutputs[truthTableIndex][i] = a[n][i];
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
         int truthTableIndex, n, m, I;

         totalError = 0.0;
         averageError = 0.0;

         for (truthTableIndex = 0; truthTableIndex < testCases; truthTableIndex++)
         {
            n = INDEX_INPUT_LAYER;

            for (m = 0; m < nInputs; m++)
            {
               a[n][m] = truthTableInputs[truthTableIndex][m];
            }

            runForTraining(truthTableIndex);
            trainBackprop();
            run();

            n = INDEX_OUTPUT_LAYER;

            for (I = 0; I < nOutputs; I++)
            {
               double omega = truthTableExpectedOutputs[truthTableIndex][I] - a[n][I];
               
               totalError += 0.5 * omega * omega;
               truthTableActualOutputs[truthTableIndex][I] = a[n][I];
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
      int n, M, k, K, j, J, i;

      for (k = 0; k < nHidden1; k++)
      {
         n = INDEX_INPUT_LAYER;

         upperTheta[n + 1][k] = 0.0;

         for (M = 0; M < nInputs; M++)
         {
            upperTheta[n + 1][k] += a[n][M] * weights[n][M][k];
         }

         n = INDEX_HIDDEN_LAYER_1;
         a[n][k] = activate(upperTheta[n][k]);
      } // for (k = 0; k < nHidden1; k++)

      for (j = 0; j < nHidden2; j++)
      {
         n = INDEX_HIDDEN_LAYER_1;

         upperTheta[n + 1][j] = 0.0;

         for (K = 0; K < nHidden1; K++)
         {
            upperTheta[n + 1][j] += a[n][K] * weights[n][K][j];
         }

         n = INDEX_HIDDEN_LAYER_2;
         a[n][j] = activate(upperTheta[n][j]);
      } // for (j = 0; j < nHidden2; j++)

      for (i = 0; i < nOutputs; i++)
      {
         n = INDEX_HIDDEN_LAYER_2;

         double upperThetaI = 0.0;

         for (J = 0; J < nHidden2; J++)
         {
            upperThetaI += a[n][J] * weights[n][J][i];
         }

         n = INDEX_OUTPUT_LAYER;

         a[n][i] = activate(upperThetaI);

         double lowerOmega = truthTableExpectedOutputs[testCase][i] - a[n][i];

         psi[n][i] = lowerOmega * derivActivate(upperThetaI);
      } // for (i = 0; i < nOutputs; i++)
   } // public void runForTraining(int testCase)

/*
 * The trainBackprop() method finds the new weights by training with backpropagation.
 */
   public void trainBackprop()
   {
      int n, m, k, j, J, I;
      double jUpperOmega, kUpperOmega, kUpperPsi;

      for (j = 0; j < nHidden2; j++)
      {
         n = INDEX_HIDDEN_LAYER_2;

         jUpperOmega = 0.0;

         for (I = 0; I < nOutputs; I++)
         {
            jUpperOmega += psi[n + 1][I] * weights[n][j][I];
            weights[n][j][I] += lambda * a[n][j] * psi[n + 1][I];
         }

         psi[n][j] = jUpperOmega * derivActivate(upperTheta[n][j]);
      } // for (j = 0; j < nHidden2; j++)

      for (k = 0; k < nHidden1; k++)
      {
         n = INDEX_HIDDEN_LAYER_1;

         kUpperOmega = 0.0;

         for (J = 0; J < nHidden2; J++)
         {
            kUpperOmega += psi[n + 1][J] * weights[n][k][J];
            weights[n][k][J] += lambda * a[n][k] * psi[n + 1][J];
         }

         kUpperPsi = kUpperOmega * derivActivate(upperTheta[n][k]);
         n = INDEX_INPUT_LAYER;

         for (m = 0; m < nInputs; m++)
         {
            weights[n][m][k] += lambda * a[n][m] * kUpperPsi;
         }
      } // for (k = 0; k < nHidden1; k++)
   } // public void trainBackprop()

/*
 * The run() method calculates the hidden and output activations based on the current values in the activations and the weights. 
 */
   public void run()
   {
      int n, M, k, K, j, J, i;

      double theta1, theta2, thetaOutput; 

      for (k = 0; k < nHidden1; k++)
      {
         n = INDEX_INPUT_LAYER;

         theta1 = 0.0;

         for (M = 0; M < nInputs; M++)
         {
            theta1 += a[n][M] * weights[n][M][k];
         }

         n = INDEX_HIDDEN_LAYER_1;
         a[n][k] = activate(theta1);
      } // for (k = 0; k < nHidden1; k++)

      for (j = 0; j < nHidden2; j++)
      {
         n = INDEX_HIDDEN_LAYER_1;

         theta2 = 0.0;

         for (K = 0; K < nHidden1; K++)
         {
            theta2 += a[n][K] * weights[n][K][j];
         }

         n = INDEX_HIDDEN_LAYER_2;
         a[n][j] = activate(theta2);
      } // for (j = 0; j < nHidden2; j++)

      for (i = 0; i < nOutputs; i++)
      {
         n = INDEX_HIDDEN_LAYER_2;

         thetaOutput = 0.0;

         for (J = 0; J < nHidden2; J++)
         {
            thetaOutput += a[n][J] * weights[n][J][i];
         }

         n = INDEX_OUTPUT_LAYER;

         a[n][i] = activate(thetaOutput);
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

         if (training) // expected outputs given only when training
         {

            for (i = 0; i < nOutputs; i++)
            {
               truthTableOutput += truthTableExpectedOutputs[rowTruthTable][i] + "\t";
            }

            truthTableOutput += "|\t";

         } // if (training)

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
      int n, m, k, j, i;

      try 
      {
         FileOutputStream fos = new FileOutputStream(saveWeightsFile);
         DataOutputStream dos = new DataOutputStream(fos);

         dos.writeInt(nInputs);
         dos.writeInt(nHidden1);
         dos.writeInt(nHidden2);
         dos.writeInt(nOutputs);

         n = INDEX_INPUT_LAYER;

         for (m = 0; m < nInputs; m++)
         {
            for (k = 0; k < nHidden1; k++)
            {
               dos.writeDouble(weights[n][m][k]);
            }
         } // for (m = 0; m < nInputs; m++)

         n = INDEX_HIDDEN_LAYER_1;

         for (k = 0; k < nHidden1; k++)
         {
            for (j = 0; j < nHidden2; j++)
            {
               dos.writeDouble(weights[n][k][j]);
            }
         } // for (k = 0; k < nHidden1; k++)

         n = INDEX_HIDDEN_LAYER_2;

         for (j = 0; j < nHidden2; j++)
         {
            for (i = 0; i < nOutputs; i++)
            {
               dos.writeDouble(weights[n][j][i]);
            }
         } // for (j = 0; j < nHidden2; j++)
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
      int n, m, k, j, i;

      try 
      {
         FileInputStream fis = new FileInputStream(initialWeightsFile);
         DataInputStream dis = new DataInputStream(fis);

         int inputConfig = dis.readInt();
         int h1Config = dis.readInt();
         int h2Config = dis.readInt();
         int outputConfig = dis.readInt();

         if (inputConfig != nInputs || h1Config != nHidden1 || h2Config != nHidden2 ||outputConfig != nOutputs)
         {
            dis.close();
            throw new IOException("Loading weights from file with the incorrect number of activations in a layer");
         }

         n = INDEX_INPUT_LAYER;

         for (m = 0; m < nInputs; m++)
         {
            for (k = 0; k < nHidden1; k++)
            {
               weights[n][m][k] = dis.readDouble();
            }
         } // for (m = 0; m < nInputs; m++)

         n = INDEX_HIDDEN_LAYER_1;

         for (k = 0; k < nHidden1; k++)
         {
            for (j = 0; j < nHidden2; j++)
            {
               weights[n][k][j] = dis.readDouble();
            }
         } // for (k = 0; k < nHidden1; k++)

         n = INDEX_HIDDEN_LAYER_2;

         for (j = 0; j < nHidden2; j++)
         {
            for (i = 0; i < nOutputs; i++)
            {
               weights[n][j][i] = dis.readDouble();
            }
         } // for (j = 0; j < nHidden2; j++)
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
   } // public void loadWeights()

/*
 * The readConfigFile() method reads the given, or default if not given, configuration file and sets all relevant configuration
 * parameters.
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
         nHidden1 = Integer.parseInt(tokenizer.nextToken());
         nHidden2 = Integer.parseInt(tokenizer.nextToken());
         nOutputs = Integer.parseInt(tokenizer.nextToken());

         line = reader.readLine();
         tokenizer = new StringTokenizer(line);

         testCases = Integer.parseInt(tokenizer.nextToken());

         line = reader.readLine();

         testCaseFileName = line;
         
         if (training)
         {
            line = reader.readLine();

            for (testCase = 0; testCase < testCases; testCase++)
            {
               line = reader.readLine(); // need to read over truth table lines as it is going to be dealt with in populate().
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
      int m, testCase;
      
      try
      {
         BufferedReader reader = new BufferedReader(new FileReader(testCaseFileName));
         String line = reader.readLine();
         StringTokenizer tokenizer = new StringTokenizer(line);

         for (testCase = 0; testCase < testCases; testCase++)
         {
            for (m = 0; m < nInputs; m++)
            {
               truthTableInputs[testCase][m] = Double.parseDouble(tokenizer.nextToken());
            }

            line = reader.readLine();
            if (line != null)
            {
               tokenizer = new StringTokenizer(line);
            }
         } // for (testCase = 0; testCase < testCases; testCase++)
      } // try
      catch (Exception e)
      {
         System.out.println("Exception " + e.toString());
      }
   } // public void readTestCaseFile()
} // public class ABCBackprop