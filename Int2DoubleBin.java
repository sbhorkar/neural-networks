import java.io.*;

public class Int2DoubleBin {
    public static void main(String[] args) {
        // Check if the command line arguments are provided
        if (args.length != 2) {
            System.out.println("Usage: java ByteToDoubleScaler <input file> <output file>");
            return;
        }

        String inputFile = args[0];
        String outputFile = args[1];

        try (DataInputStream in = new DataInputStream(new FileInputStream(inputFile));
             DataOutputStream out = new DataOutputStream(new FileOutputStream(outputFile))) {

            // Read each byte as unsigned, scale it, and write the result as a double
            while (in.available() > 0) {
                int byteValue = in.readUnsignedByte(); // Read one byte and treat it as unsigned
                double scaledValue = ((double) byteValue) / 255.0; // Scale to 0-1
                out.writeDouble(scaledValue); // Write scaled value as double
            }

            System.out.println("Processing completed successfully.");
        } catch (IOException e) {
            System.err.println("Error processing the file: " + e.getMessage());
        }
    }
}
